package fantasy

import (
	"fmt"
	"testing"

	"golang.org/x/net/http2"
)

func TestIsTransportError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		err  error
		want bool
	}{
		{"nil error", nil, false},
		{"stream error with peer", newTestError("stream error: stream ID 27; INTERNAL_ERROR; received from peer"), true},
		{"stream error without peer", newTestError("stream error: stream ID 5; REFUSED_STREAM"), true},
		{"connection error", newTestError("connection error: INTERNAL_ERROR"), true},
		{"http2-prefixed connection error", newTestError("http2: connection error: PROTOCOL_ERROR: bad frame"), true},
		{"generic error", newTestError("something went wrong"), false},
		{"EOF", newTestError("EOF"), false},
		{"empty error", newTestError(""), false},
		{"wrapped stream error", fmt.Errorf("reading body: %w", newTestError("stream error: stream ID 3; INTERNAL_ERROR")), true},
		{"x/net StreamError", http2.StreamError{StreamID: 1, Code: http2.ErrCodeInternal}, true},
		{"x/net ConnectionError", http2.ConnectionError(http2.ErrCodeInternal), true},
		{"x/net GoAwayError", http2.GoAwayError{LastStreamID: 1, ErrCode: http2.ErrCodeInternal}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := IsTransportError(tt.err); got != tt.want {
				t.Errorf("IsTransportError(%v) = %v, want %v", tt.err, got, tt.want)
			}
		})
	}
}

func TestCleanHTTP2ErrorMessage(t *testing.T) {
	t.Parallel()

	tests := []struct {
		input string
		want  string
	}{
		{
			"stream error: stream ID 27; INTERNAL_ERROR; received from peer",
			"INTERNAL_ERROR (received from peer)",
		},
		{
			"stream error: stream ID 5; REFUSED_STREAM",
			"REFUSED_STREAM",
		},
		{
			"connection error: INTERNAL_ERROR",
			"INTERNAL_ERROR",
		},
		{
			"some other error",
			"some other error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			t.Parallel()
			if got := cleanHTTP2ErrorMessage(tt.input); got != tt.want {
				t.Errorf("cleanHTTP2ErrorMessage(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestNewTransportError(t *testing.T) {
	t.Parallel()

	rawErr := newTestError("stream error: stream ID 27; INTERNAL_ERROR; received from peer")
	err := NewTransportError(rawErr)

	if err.Title != "stream transport error" {
		t.Errorf("Title = %q, want %q", err.Title, "stream transport error")
	}
	if err.Message != "INTERNAL_ERROR (received from peer)" {
		t.Errorf("Message = %q, want %q", err.Message, "INTERNAL_ERROR (received from peer)")
	}
	if !err.IsRetryable() {
		t.Error("expected HTTP/2 transport error to be retryable")
	}
}

func TestNewTransportErrorWrapped(t *testing.T) {
	t.Parallel()

	rawErr := fmt.Errorf("reading response body: %w",
		newTestError("stream error: stream ID 12; REFUSED_STREAM"))
	err := NewTransportError(rawErr)

	if err.Message != "REFUSED_STREAM" {
		t.Errorf("Message = %q, want %q", err.Message, "REFUSED_STREAM")
	}
	if !err.IsRetryable() {
		t.Error("expected wrapped HTTP/2 transport error to be retryable")
	}
}

type testError struct{ msg string }

func (e *testError) Error() string { return e.msg }

func newTestError(msg string) error { return &testError{msg: msg} }
