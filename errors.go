package fantasy

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"
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

	// TransientError marks the error as a temporary failure a provider
	// flagged as worth retrying even though it carries no retryable HTTP
	// status code. Errors delivered inside an already-successful response,
	// such as a mid-stream SSE error event, are reported with the status of
	// the response that carried them, which says nothing about whether a
	// second attempt can succeed.
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
// transient HTTP/2 transport failure. Go's standard library bundles its
// own copy of the http2 package whose error types are unexported, so they
// cannot be matched with errors.As. We fall back to matching these stable
// fragments, which both the stdlib and x/net/http2 use. The list is kept
// tight to avoid misclassifying application-level errors as transport
// failures.
var http2TransportErrorFragments = []string{
	"stream error:",     // RST_STREAM: INTERNAL_ERROR, REFUSED_STREAM, CANCEL, etc.
	"connection error:", // connection-level protocol error
}

// IsTransportError reports whether err or any error in its chain is a
// transient transport-level failure that is safe to retry on a fresh
// connection. In practice these are HTTP/2 stream resets, connection
// errors, and GOAWAY frames, which originate from the transport rather
// than the application.
//
// x/net/http2 error types are matched by type; Go's stdlib-bundled http2
// uses unexported types, so those are matched by their message fragments.
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

// streamEventErrorPrefix is the message prefix the Anthropic and OpenAI Go
// SDKs use when a stream that opened successfully delivers an `error` event
// instead of completing. The HTTP response carrying such an event is a 200,
// so the status code cannot tell us whether a retry is worthwhile; the
// payload that follows the prefix can.
const streamEventErrorPrefix = "received error while streaming:"

// transientStreamErrorTypes are provider error `type` values that name a
// temporary, server-side condition, meaning the same request may well
// succeed on a later attempt. Anthropic reports shed capacity as
// overloaded_error and internal faults as api_error; OpenAI-compatible
// providers use server_error or internal_error. Either family can report a
// rate limit mid-stream. The list is kept tight so permanent failures
// (invalid_request_error, authentication_error, and the like) are never
// retried.
var transientStreamErrorTypes = []string{
	"overloaded_error",
	"api_error",
	"server_error",
	"internal_error",
	"rate_limit_error",
}

// streamErrorEvent is the payload of an SSE `error` event. Anthropic nests
// the detail under "error" while the OpenAI SDK hands over the already
// unwrapped inner object, so both shapes are accepted.
type streamErrorEvent struct {
	Type    string            `json:"type"`
	Message string            `json:"message"`
	Error   *streamErrorEvent `json:"error"`
}

// WrapStreamError wraps an error event delivered mid-stream in a
// ProviderError, marking it transient when the payload names a temporary
// server-side condition such as Anthropic's overloaded_error. Any other
// error is returned unchanged.
//
// Providers should call this before WrapTransportError: these events ride
// inside a successful HTTP response, so without classification they reach
// callers as an opaque, non-retryable blob of JSON and a saturated provider
// looks like a hard failure.
func WrapStreamError(err error) error {
	if err == nil {
		return nil
	}
	msg := err.Error()
	i := strings.Index(msg, streamEventErrorPrefix)
	if i == -1 {
		return err
	}
	payload := strings.TrimSpace(msg[i+len(streamEventErrorPrefix):])
	errType, message := parseStreamErrorEvent(payload)
	if errType == "" && message == "" {
		// Nothing recognizable to report: leave the error as it came so no
		// detail is lost.
		return err
	}
	return &ProviderError{
		Title:          streamErrorTitle(errType),
		Message:        cmp.Or(message, payload),
		Cause:          err,
		ResponseBody:   []byte(payload),
		TransientError: slices.Contains(transientStreamErrorTypes, errType),
	}
}

// parseStreamErrorEvent extracts the error type and message from an SSE
// error event payload, unwrapping the nested "error" object when present.
// A payload that is not JSON, or that is JSON without useful fields, still
// yields a type when it names a known transient condition, since a provider
// that mangles its own envelope is exactly the sort we want to retry.
func parseStreamErrorEvent(payload string) (errType, message string) {
	var event streamErrorEvent
	if err := json.Unmarshal([]byte(payload), &event); err == nil {
		// The outer envelope's type is always "error"; the useful value is
		// nested when a provider wraps the detail.
		if event.Error != nil {
			errType = cmp.Or(event.Error.Type, event.Type)
			message = cmp.Or(event.Error.Message, event.Message)
		} else {
			errType, message = event.Type, event.Message
		}
		if errType != "error" {
			return errType, message
		}
	}
	for _, candidate := range transientStreamErrorTypes {
		if strings.Contains(payload, candidate) {
			return candidate, message
		}
	}
	return "", message
}

// streamErrorTitle renders a provider error type as a human-readable title.
func streamErrorTitle(errType string) string {
	switch errType {
	case "overloaded_error":
		return "provider overloaded"
	case "rate_limit_error":
		return "rate limit exceeded"
	case "":
		return "provider stream error"
	default:
		return strings.ReplaceAll(errType, "_", " ")
	}
}

// extractHTTP2ErrorMessage locates the HTTP/2 error fragment within a
// possibly-wrapped error message and returns a concise, cleaned form for
// display. It falls back to the full message when no fragment is found.
//
//	"stream error: stream ID 27; INTERNAL_ERROR; received from peer" → "INTERNAL_ERROR (received from peer)"
//	"stream error: stream ID 5; REFUSED_STREAM"                      → "REFUSED_STREAM"
//	"http2: connection error: INTERNAL_ERROR"                        → "INTERNAL_ERROR"
func extractHTTP2ErrorMessage(err error) string {
	msg := err.Error()
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
