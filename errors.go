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
// It returns true if the underlying cause is io.ErrUnexpectedEOF, if the
// "x-should-retry" response header evaluates to true, if the HTTP status
// code indicates a retryable condition (408, 409, 429, or any 5xx), or
// if the cause is a transient HTTP/2 transport error.
func (m *ProviderError) IsRetryable() bool {
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
