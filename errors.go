package fantasy

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"

	"github.com/charmbracelet/x/exp/slice"
	"golang.org/x/net/http2"
)

// Error is a custom error type for the fantasy package.
type Error struct {
	Message string
	Title   string
	Cause   error
}

func (err *Error) Error() string {
	if err.Title == "" {
		return err.Message
	}
	return fmt.Sprintf("%s: %s", err.Title, err.Message)
}

func (err Error) Unwrap() error {
	return err.Cause
}

// ProviderError represents an error returned by an external provider.
type ProviderError struct {
	Message string
	Title   string
	Cause   error

	URL             string
	StatusCode      int
	RequestBody     []byte
	ResponseHeaders map[string]string
	ResponseBody    []byte

	ContextUsedTokens  int
	ContextMaxTokens   int
	ContextTooLargeErr bool

	// AuthError marks the error as an authentication failure a provider
	// flagged as resolvable by refreshing credentials (e.g. re-running an
	// interactive login). It covers auth failures that do not carry an HTTP
	// 401 status, so a caller-supplied OnAuthRefresh hook still engages.
	AuthError bool

	// TransientError marks a temporary server-side failure worth retrying
	// despite carrying no retryable HTTP status code. Mid-stream SSE error
	// events ride inside an already-successful 200 response, so the status
	// code alone cannot signal that a retry may succeed.
	TransientError bool
}

func (m *ProviderError) Error() string {
	if m.Title == "" {
		return m.Message
	}
	return fmt.Sprintf("%s: %s", m.Title, m.Message)
}

// Unwrap returns the underlying cause so errors.Is and errors.As can
// inspect the wrapped error (e.g. an HTTP/2 transport error).
func (m *ProviderError) Unwrap() error {
	return m.Cause
}

// IsRetryable reports whether the error should be retried.
// It returns true if the error is flagged as transient, if the underlying
// cause is io.ErrUnexpectedEOF, if the "x-should-retry" response header
// evaluates to true, if the HTTP status code indicates a retryable
// condition (408, 409, 429, or any 5xx), or if the cause is a transient
// HTTP/2 transport error.
func (m *ProviderError) IsRetryable() bool {
	if m.TransientError {
		return true
	}
	// We're mostly mimicking OpenAI's Go SDK here:
	// https://github.com/openai/openai-go/blob/b9d280a37149430982e9dfeed16c41d27d45cfc5/internal/requestconfig/requestconfig.go#L244
	if errors.Is(m.Cause, io.ErrUnexpectedEOF) {
		return true
	}
	if IsTransportError(m.Cause) {
		return true
	}
	if m.shouldRetryHeader() {
		return true
	}
	return m.StatusCode == http.StatusRequestTimeout ||
		m.StatusCode == http.StatusConflict ||
		m.StatusCode == http.StatusTooManyRequests ||
		m.StatusCode >= http.StatusInternalServerError
}

func (m *ProviderError) shouldRetryHeader() bool {
	if m.ResponseHeaders == nil {
		return false
	}
	for k, v := range m.ResponseHeaders {
		if strings.EqualFold(k, "x-should-retry") {
			b, _ := strconv.ParseBool(v)
			return b
		}
	}
	return false
}

// IsContextTooLarge checks if the error is due to the context exceeding the model's limit.
func (m *ProviderError) IsContextTooLarge() bool {
	return m.ContextTooLargeErr || m.ContextMaxTokens > 0 || m.ContextUsedTokens > 0
}

// NewIncompleteStreamError returns a retryable ProviderError indicating that
// an upstream stream closed cleanly without delivering its terminal signal
// (finish_reason, stop_reason, response.completed, candidate.finishReason,
// etc.). The cause is io.ErrUnexpectedEOF so ProviderError.IsRetryable()
// engages and the retry middleware re-runs the step.
func NewIncompleteStreamError() *ProviderError {
	return &ProviderError{
		Title:   "stream transport error",
		Message: io.ErrUnexpectedEOF.Error(),
		Cause:   io.ErrUnexpectedEOF,
	}
}

// http2TransportErrorFragments are message fragments that identify a
// transient HTTP/2 transport failure. Go's standard library carries its own
// copy of the http2 package, since Go 1.27 at net/http/internal/http2, and
// an internal package's types cannot be named by an importer however they
// are declared, so errors.As against x/net/http2 never matches one. We fall
// back to matching these stable fragments, which both copies use. The list
// is kept tight to avoid misclassifying application-level errors as
// transport failures.
//
// Every fragment here names the framing of a stream- or connection-level
// protocol error, "…: stream ID N; CODE" and the like, which is the shape
// cleanHTTP2ErrorMessage knows how to trim. Messages reporting that the
// connection itself died belong in http2ConnectionLostMessages instead.
var http2TransportErrorFragments = []string{
	"stream error:",     // RST_STREAM: INTERNAL_ERROR, REFUSED_STREAM, CANCEL, etc.
	"connection error:", // connection-level protocol error
}

// http2ConnectionLostMessages identify an HTTP/2 connection that died under
// a request, rather than a protocol error reported over a live one. Both
// are worth retrying, since a fresh connection is exactly the remedy, but
// these arrive as plain errors carrying no type to match and no shared
// framing to parse, so they are listed in full.
//
// The first is what a failed health-check ping produces. A transport
// configured with HTTP2Config.SendPingTimeout pings a connection that has
// gone quiet and closes it when the ping goes unanswered, which is how a
// connection killed while the machine slept gets noticed at all: nothing
// else in the stack tells a dead peer from a slow one. Leaving it
// unclassified turns that recovery into an immediate hard failure, because
// the error is not a net.Error either.
//
// The second is a server's GOAWAY. x/net's GoAwayError is matched by type
// above; the standard library's cannot be, so it needs a string.
var http2ConnectionLostMessages = []string{
	"http2: client connection lost",
	"http2: server sent GOAWAY and closed the connection",
}

// IsTransportError reports whether err or any error in its chain is a
// transient transport-level failure that is safe to retry on a fresh
// connection. In practice these are HTTP/2 stream resets, connection
// errors, and GOAWAY frames, which originate from the transport rather
// than the application.
//
// x/net/http2 error types are matched by type. The standard library's copy
// lives in an internal package, so its equivalents are matched by message
// instead.
func IsTransportError(err error) bool {
	if err == nil {
		return false
	}
	var (
		streamErr http2.StreamError
		connErr   http2.ConnectionError
		goAwayErr http2.GoAwayError
	)
	if errors.As(err, &streamErr) ||
		errors.As(err, &connErr) ||
		errors.As(err, &goAwayErr) {
		return true
	}
	// Wrapped errors embed the inner message, so scanning the top-level
	// string covers the whole chain.
	msg := err.Error()
	for _, fragment := range http2TransportErrorFragments {
		if strings.Contains(msg, fragment) {
			return true
		}
	}
	for _, lost := range http2ConnectionLostMessages {
		if strings.Contains(msg, lost) {
			return true
		}
	}
	return false
}

// NewTransportError wraps a transient transport error into a retryable
// ProviderError with a human-friendly title and message.
func NewTransportError(err error) *ProviderError {
	return &ProviderError{
		Title:   "stream transport error",
		Message: extractHTTP2ErrorMessage(err),
		Cause:   err,
	}
}

// TransientStreamErrorTypes are provider error "type" (or "code") values
// that name a temporary server-side condition worth retrying. Mid-stream
// SSE error events ride inside an already-successful 200 response, so the
// HTTP status code cannot signal retryability; providers classify the
// payload against this set and set ProviderError.TransientError.
//
// This is the canonical list. Providers parse their SDK-specific error
// shapes but defer the transient/permanent policy decision here.
var TransientStreamErrorTypes = map[string]bool{
	"server_error":     true,
	"internal_error":   true,
	"overloaded_error": true,
	"api_error":        true,
	"rate_limit_error": true,
}

// WrapTransportError wraps a transient transport failure in a retryable
// ProviderError so callers get a clean message and .IsRetryable() reports
// true. It recognizes an unexpected mid-stream EOF and HTTP/2 stream,
// connection, and GOAWAY resets. Any other error is returned unchanged.
//
// This is the canonical entry point for provider error handlers: they can
// hand off whatever the transport surfaced without re-encoding which
// failures count as transient.
func WrapTransportError(err error) error {
	switch {
	case errors.Is(err, io.ErrUnexpectedEOF):
		return &ProviderError{
			Title:   "stream transport error",
			Message: err.Error(),
			Cause:   err,
		}
	case IsTransportError(err):
		return NewTransportError(err)
	default:
		return err
	}
}

// extractHTTP2ErrorMessage locates the HTTP/2 error fragment within a
// possibly-wrapped error message and returns a concise, cleaned form for
// display. It falls back to the full message when no fragment is found.
//
//	"stream error: stream ID 27; INTERNAL_ERROR; received from peer" → "INTERNAL_ERROR (received from peer)"
//	"stream error: stream ID 5; REFUSED_STREAM"                      → "REFUSED_STREAM"
//	"http2: connection error: INTERNAL_ERROR"                        → "INTERNAL_ERROR"
//	"http2: client connection lost"                                  → "client connection lost"
//	"http2: server sent GOAWAY ...; LastStreamID=5, ErrCode=NO_ERROR" → "server sent GOAWAY and closed the connection"
func extractHTTP2ErrorMessage(err error) string {
	msg := err.Error()
	// Checked first, and returned rather than cleaned. These messages are
	// already a whole sentence, and GOAWAY carries trailing "; LastStreamID=…"
	// detail that cleanHTTP2ErrorMessage would mistake for the "; CODE" of a
	// stream error and return on its own.
	for _, lost := range http2ConnectionLostMessages {
		if strings.Contains(msg, lost) {
			return strings.TrimPrefix(lost, "http2: ")
		}
	}
	for _, fragment := range http2TransportErrorFragments {
		if i := strings.Index(msg, fragment); i != -1 {
			return cleanHTTP2ErrorMessage(msg[i:])
		}
	}
	return msg
}

// cleanHTTP2ErrorMessage trims the verbose framing from an HTTP/2 error
// string that begins at a known fragment. "stream error: stream ID N; CODE"
// collapses to "CODE" (with any trailing cause in parentheses), and
// "connection error: CODE" collapses to "CODE".
func cleanHTTP2ErrorMessage(msg string) string {
	// "stream error: stream ID N; CODE[; cause]".
	if idx := strings.Index(msg, "; "); idx != -1 {
		rest := msg[idx+2:]
		code, cause, hasCause := strings.Cut(rest, "; ")
		if hasCause {
			return fmt.Sprintf("%s (%s)", code, cause)
		}
		return code
	}
	// "connection error: CODE".
	if _, code, ok := strings.Cut(msg, ": "); ok {
		return code
	}
	return msg
}

// RetryError represents an error that occurred during retry operations.
type RetryError struct {
	Errors []error
}

func (e *RetryError) Error() string {
	if err, ok := slice.Last(e.Errors); ok {
		return fmt.Sprintf("retry error: %v", err)
	}
	return "retry error: no underlying errors"
}

func (e RetryError) Unwrap() error {
	if err, ok := slice.Last(e.Errors); ok {
		return err
	}
	return nil
}

// ToolExecutionError is returned by the agent when a tool's Run function
// returns a Go error. The failure ends the step, and it is never retried:
// the error is local to the tool, so re-running the step would only repeat
// the model request and re-execute every tool in it. Callers can use
// errors.As to find it and errors.Unwrap (or errors.Is/As) to reach the
// tool's own error.
type ToolExecutionError struct {
	ToolName   string
	ToolCallID string
	Err        error
}

func (e *ToolExecutionError) Error() string {
	return fmt.Sprintf("tool %q failed: %v", e.ToolName, e.Err)
}

// Unwrap returns the error the tool returned.
func (e *ToolExecutionError) Unwrap() error {
	return e.Err
}

// ErrorTitleForStatusCode returns a human-readable title for a given HTTP status code.
func ErrorTitleForStatusCode(statusCode int) string {
	return strings.ToLower(http.StatusText(statusCode))
}

// NoObjectGeneratedError is returned when object generation fails
// due to parsing errors, validation errors, or model failures.
type NoObjectGeneratedError struct {
	RawText         string
	ParseError      error
	ValidationError error
	Usage           Usage
	FinishReason    FinishReason
}

// Error implements the error interface.
func (e *NoObjectGeneratedError) Error() string {
	if e.ValidationError != nil {
		return fmt.Sprintf("object validation failed: %v", e.ValidationError)
	}
	if e.ParseError != nil {
		return fmt.Sprintf("failed to parse object: %v", e.ParseError)
	}
	return "failed to generate object"
}

// IsNoObjectGeneratedError checks if an error is of type NoObjectGeneratedError.
func IsNoObjectGeneratedError(err error) bool {
	var target *NoObjectGeneratedError
	return errors.As(err, &target)
}
