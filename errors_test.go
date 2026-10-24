package fantasy

import (
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

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
		{"lost health-check ping", newTestError("http2: client connection lost"), true},
		{"wrapped lost ping", fmt.Errorf("reading body: %w", newTestError("http2: client connection lost")), true},
		{"stdlib GoAwayError", newTestError(`http2: server sent GOAWAY and closed the connection; LastStreamID=5, ErrCode=NO_ERROR, debug=""`), true},
		{"deliberate close is not transient", newTestError("http2: client connection force closed via ClientConn.Close"), false},
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

func TestExtractHTTP2ErrorMessage(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		err  error
		want string
	}{
		{
			"lost ping drops the http2 prefix",
			newTestError("http2: client connection lost"),
			"client connection lost",
		},
		{
			// The trailing "; LastStreamID=…" reads like the "; CODE" of a
			// stream error, so cleaning this one would return the frame
			// bookkeeping and discard the part that says what happened.
			"GOAWAY keeps the sentence, not the frame detail",
			newTestError(`http2: server sent GOAWAY and closed the connection; LastStreamID=5, ErrCode=NO_ERROR, debug=""`),
			"server sent GOAWAY and closed the connection",
		},
		{
			"stream errors still clean",
			newTestError("stream error: stream ID 27; INTERNAL_ERROR; received from peer"),
			"INTERNAL_ERROR (received from peer)",
		},
		{
			"unrecognised errors pass through",
			newTestError("something went wrong"),
			"something went wrong",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := extractHTTP2ErrorMessage(tt.err); got != tt.want {
				t.Errorf("extractHTTP2ErrorMessage(%v) = %q, want %q", tt.err, got, tt.want)
			}
		})
	}
}

// TestConnectionLostIsRetryable covers the path the retry middleware walks.
// A lost connection reaches it as a bare error that is not a net.Error and
// carries no type to match, so without the message check it reads as
// permanent and the request is never retried.
func TestConnectionLostIsRetryable(t *testing.T) {
	t.Parallel()

	// Spelled out rather than ranged over http2ConnectionLostMessages, so
	// emptying that list fails here instead of passing vacuously.
	for _, msg := range []string{
		"http2: client connection lost",
		"http2: server sent GOAWAY and closed the connection",
	} {
		t.Run(msg, func(t *testing.T) {
			t.Parallel()
			err := newTestError(msg)
			if !isRetryableError(err) {
				t.Error("expected a lost connection to be retryable")
			}
			if wrapped := WrapTransportError(err); !errors.Is(wrapped, err) {
				t.Errorf("WrapTransportError dropped the cause: %v", wrapped)
			}
		})
	}
}

func TestRetryAtFromHeaders(t *testing.T) {
	t.Parallel()

	t.Run("delay seconds", func(t *testing.T) {
		t.Parallel()

		retryAt := RetryAtFromHeaders(map[string]string{"retry-after": "30"})
		if wait := time.Until(retryAt); wait <= 0 || wait > 30*time.Second {
			t.Errorf("time.Until(RetryAt) = %v, want within (0s, 30s]", wait)
		}
	})

	t.Run("milliseconds take precedence", func(t *testing.T) {
		t.Parallel()

		retryAt := RetryAtFromHeaders(map[string]string{"retry-after-ms": "500", "retry-after": "30"})
		if wait := time.Until(retryAt); wait <= 0 || wait > 500*time.Millisecond {
			t.Errorf("time.Until(RetryAt) = %v, want within (0s, 500ms]", wait)
		}
	})

	t.Run("http date", func(t *testing.T) {
		t.Parallel()

		want := time.Now().UTC().Add(time.Hour).Truncate(time.Second)
		retryAt := RetryAtFromHeaders(map[string]string{"retry-after": want.Format(time.RFC1123)})
		if diff := retryAt.Sub(want); diff < -time.Second || diff > time.Second {
			t.Errorf("RetryAt = %v, want %v", retryAt, want)
		}
	})

	t.Run("case-insensitive header names", func(t *testing.T) {
		t.Parallel()

		if RetryAtFromHeaders(map[string]string{"Retry-After": "30"}).IsZero() {
			t.Error("RetryAt is zero for a canonical-cased Retry-After header")
		}
	})

	t.Run("missing, invalid, and non-positive values", func(t *testing.T) {
		t.Parallel()

		for name, headers := range map[string]map[string]string{
			"nil":        nil,
			"none":       {},
			"invalid":    {"retry-after": "invalid"},
			"zero":       {"retry-after": "0"},
			"negative":   {"retry-after": "-5"},
			"invalid-ms": {"retry-after-ms": "soon"},
			"past date":  {"retry-after": time.Now().Add(-time.Hour).UTC().Format(time.RFC1123)},
		} {
			t.Run(name, func(t *testing.T) {
				t.Parallel()

				if retryAt := RetryAtFromHeaders(headers); !retryAt.IsZero() {
					t.Errorf("RetryAtFromHeaders(%v) = %v, want the zero time", headers, retryAt)
				}
			})
		}
	})
}

func TestIsQuotaExhaustedMessage(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		message string
		want    bool
	}{
		{"synthetic subscription body", `{"error":"You've exceeded your subscription rate limits. Upgrade, or try again later."}`, true},
		{"case-insensitive", "you have EXCEEDED YOUR SUBSCRIPTION quota", true},
		{"plain rate limit", "Rate limit reached for requests", false},
		{"unrelated", "model overloaded", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			if got := IsQuotaExhaustedMessage(tc.message); got != tc.want {
				t.Errorf("IsQuotaExhaustedMessage(%q) = %v, want %v", tc.message, got, tc.want)
			}
		})
	}
}

func TestProviderErrorRetryDeadline(t *testing.T) {
	t.Parallel()

	t.Run("Error names a future deadline", func(t *testing.T) {
		t.Parallel()

		err := &ProviderError{
			Title:   "too many requests",
			Message: "quota gone",
			RetryAt: time.Now().Add(90 * time.Minute),
		}
		if msg := err.Error(); !strings.Contains(msg, "retry at ") || !strings.Contains(msg, "(in 1h30m") {
			t.Errorf("Error() = %q, want it to name the deadline and the wait", msg)
		}
	})

	t.Run("Error omits a passed deadline", func(t *testing.T) {
		t.Parallel()

		err := &ProviderError{Title: "too many requests", Message: "quota gone", RetryAt: time.Now().Add(-time.Minute)}
		if msg := err.Error(); strings.Contains(msg, "retry at") {
			t.Errorf("Error() = %q, want no deadline suffix once it has passed", msg)
		}
	})

	t.Run("quota exhaustion without a deadline is not retryable", func(t *testing.T) {
		t.Parallel()

		err := &ProviderError{StatusCode: http.StatusTooManyRequests, QuotaExhausted: true}
		if err.IsRetryable() {
			t.Error("quota exhaustion with no renewal deadline must not be retried")
		}
	})

	t.Run("quota exhaustion with a deadline stays retryable", func(t *testing.T) {
		t.Parallel()

		err := &ProviderError{StatusCode: http.StatusTooManyRequests, QuotaExhausted: true, RetryAt: time.Now().Add(time.Minute)}
		if !err.IsRetryable() {
			t.Error("quota exhaustion with a renewal deadline is retryable at that moment")
		}
	})
}
