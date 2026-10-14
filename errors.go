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

	// ErrorType is the provider's own name for the failure, e.g.
	// "overloaded_error", "rate_limit_exceeded", "RESOURCE_EXHAUSTED".
	// Providers already parse it to classify transient failures, so carrying
	// it spares callers from reading it back out of Message, which is
	// formatted for people rather than for parsing.
	//
	// Empty when the response named no type, and always empty for a failure
	// a provider recognises by message alone rather than from a classified
	// payload. Treat it as a hint worth acting on when present rather than a
	// field every error carries.
	ErrorType string
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
// copy of the http2 package, since Go 1.27 at net/http/internal/http2, whose
// types an importer cannot name, so errors.As against x/net/http2 never
// matches one and only the message is left to go on. The list is kept tight
// to avoid misclassifying application-level errors as transport failures.
//
// These name protocol errors shaped "...: stream ID N; CODE", which is what
// cleanHTTP2ErrorMessage trims. A dead connection goes in
// http2ConnectionLostMessages instead.
var http2TransportErrorFragments = []string{
	"stream error:",     // RST_STREAM: INTERNAL_ERROR, REFUSED_STREAM, CANCEL, etc.
	"connection error:", // connection-level protocol error
}

// http2ConnectionLostMessages report an HTTP/2 connection that died under a
// request, rather than a protocol error carried over a live one. Retrying is
// the remedy, since a retry dials afresh, but they arrive as plain errors
// that are not net.Error and carry no type to match, so without this they
// read as permanent.
//
// Matched in full: they share no framing to parse, and the trailing frame
// detail on GOAWAY would misread as the "; CODE" of a stream error.
var http2ConnectionLostMessages = []string{
	"http2: client connection lost",                       // health-check ping went unanswered
	"http2: server sent GOAWAY and closed the connection", // GoAwayError, from either http2 copy
}

// matchConnectionLost returns the entry of http2ConnectionLostMessages that
// msg contains, so a caller can both detect one and name it.
func matchConnectionLost(msg string) (string, bool) {
	for _, lost := range http2ConnectionLostMessages {
		if strings.Contains(msg, lost) {
			return lost, true
		}
	}
	return "", false
}

// IsTransportError reports whether err or any error in its chain is a
// transient transport-level failure that is safe to retry on a fresh
// connection. In practice these are HTTP/2 stream resets, connection
// errors, and GOAWAY frames, which originate from the transport rather
// than the application.
//
// Stream and connection errors are matched by type against x/net/http2.
// GOAWAY is matched by message instead: x/net deprecated GoAwayError
// without offering a replacement, and the standard library's copy of the
// package is internal, so neither one's type can be named here for long.
// Both spell the failure the same way, so one fragment covers them.
func IsTransportError(err error) bool {
	if err == nil {
		return false
	}
	var (
		streamErr http2.StreamError
		connErr   http2.ConnectionError
	)
	if errors.As(err, &streamErr) || errors.As(err, &connErr) {
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
	_, lost := matchConnectionLost(msg)
	return lost
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
	// Returned whole: these are already a sentence, and cleaning GOAWAY
	// would strip it down to its trailing frame detail.
	if lost, ok := matchConnectionLost(msg); ok {
		return strings.TrimPrefix(lost, "http2: ")
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
