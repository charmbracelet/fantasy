package fantasy

import (
	"errors"
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

func TestWrapStreamError(t *testing.T) {
	t.Parallel()

	const prefix = "received error while streaming: "

	tests := []struct {
		name        string
		err         error
		wantWrapped bool
		wantRetry   bool
		wantTitle   string
		wantMessage string
	}{
		{
			name:        "anthropic overloaded",
			err:         newTestError(prefix + `{"type":"error","error":{"details":null,"type":"overloaded_error","message":"Overloaded"}}`),
			wantWrapped: true,
			wantRetry:   true,
			wantTitle:   "provider overloaded",
			wantMessage: "Overloaded",
		},
		{
			name:        "anthropic api error",
			err:         newTestError(prefix + `{"type":"error","error":{"type":"api_error","message":"Internal server error"}}`),
			wantWrapped: true,
			wantRetry:   true,
			wantTitle:   "api error",
			wantMessage: "Internal server error",
		},
		{
			name:        "openai unwrapped server error",
			err:         newTestError(prefix + `{"message":"upstream unavailable","type":"server_error","code":null}`),
			wantWrapped: true,
			wantRetry:   true,
			wantTitle:   "server error",
			wantMessage: "upstream unavailable",
		},
		{
			name:        "rate limit",
			err:         newTestError(prefix + `{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`),
			wantWrapped: true,
			wantRetry:   true,
			wantTitle:   "rate limit exceeded",
			wantMessage: "slow down",
		},
		{
			name:        "permanent error is wrapped but not retried",
			err:         newTestError(prefix + `{"type":"error","error":{"type":"invalid_request_error","message":"bad tool schema"}}`),
			wantWrapped: true,
			wantRetry:   false,
			wantTitle:   "invalid request error",
			wantMessage: "bad tool schema",
		},
		{
			name:        "non-JSON payload naming a transient type",
			err:         newTestError(prefix + "overloaded_error: capacity exhausted"),
			wantWrapped: true,
			wantRetry:   true,
			wantTitle:   "provider overloaded",
			wantMessage: "overloaded_error: capacity exhausted",
		},
		{
			name:        "wrapped by an outer error",
			err:         fmt.Errorf("anthropic: %w", newTestError(prefix+`{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`)),
			wantWrapped: true,
			wantRetry:   true,
			wantTitle:   "provider overloaded",
			wantMessage: "Overloaded",
		},
		{
			name: "unrecognizable payload passes through",
			err:  newTestError(prefix + "{}"),
		},
		{
			name: "unrelated error passes through",
			err:  newTestError("something went wrong"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := WrapStreamError(tt.err)

			var providerErr *ProviderError
			if !errors.As(got, &providerErr) {
				if tt.wantWrapped {
					t.Fatalf("WrapStreamError(%v) did not wrap as *ProviderError (got %T)", tt.err, got)
				}
				if got != tt.err {
					t.Errorf("WrapStreamError mutated error: got %v, want %v", got, tt.err)
				}
				return
			}
			if !tt.wantWrapped {
				t.Fatalf("WrapStreamError(%v) wrapped an error it should have passed through", tt.err)
			}
			if providerErr.IsRetryable() != tt.wantRetry {
				t.Errorf("IsRetryable() = %v, want %v", providerErr.IsRetryable(), tt.wantRetry)
			}
			if providerErr.Title != tt.wantTitle {
				t.Errorf("Title = %q, want %q", providerErr.Title, tt.wantTitle)
			}
			if providerErr.Message != tt.wantMessage {
				t.Errorf("Message = %q, want %q", providerErr.Message, tt.wantMessage)
			}
			if !errors.Is(got, tt.err) {
				t.Error("wrapped error must retain the original error in its chain")
			}
		})
	}
}

func TestWrapStreamErrorNil(t *testing.T) {
	t.Parallel()

	if got := WrapStreamError(nil); got != nil {
		t.Errorf("WrapStreamError(nil) = %v, want nil", got)
	}
}

type testError struct{ msg string }

func (e *testError) Error() string { return e.msg }

func newTestError(msg string) error { return &testError{msg: msg} }
