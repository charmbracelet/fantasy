package anthropic

import (
	"errors"
	"fmt"
	"io"
	"testing"

	"charm.land/fantasy"
)

func TestToProviderErr_WrapsUnexpectedEOF(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		err  error
	}{
		{"direct", io.ErrUnexpectedEOF},
		{"wrapped", fmt.Errorf("read stream: %w", io.ErrUnexpectedEOF)},
		{"double_wrapped", fmt.Errorf("anthropic: %w", fmt.Errorf("sse: %w", io.ErrUnexpectedEOF))},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			got := toProviderErr(tc.err)

			var providerErr *fantasy.ProviderError
			if !errors.As(got, &providerErr) {
				t.Fatalf("toProviderErr did not wrap %v as *fantasy.ProviderError (got %T)", tc.err, got)
			}
			if !errors.Is(providerErr.Cause, io.ErrUnexpectedEOF) {
				t.Errorf("ProviderError.Cause = %v, want chain containing io.ErrUnexpectedEOF", providerErr.Cause)
			}
			if !providerErr.IsRetryable() {
				t.Error("wrapped io.ErrUnexpectedEOF must be retryable so retry.go engages")
			}
		})
	}
}

func TestToProviderErr_PassesThroughUnrelatedErrors(t *testing.T) {
	t.Parallel()

	err := errors.New("something unrelated")
	got := toProviderErr(err)
	if got != err {
		t.Errorf("toProviderErr mutated unrelated error: got %v, want %v", got, err)
	}
}

func TestToProviderErr_PassesThroughPlainEOF(t *testing.T) {
	t.Parallel()

	// A clean io.EOF at the end of a stream is not a failure — the streaming
	// handler in anthropic.go treats it as a normal terminator and never
	// calls toProviderErr with io.EOF. But if it ever did, we should not
	// wrap it: io.EOF is not "retryable" in the ProviderError sense.
	got := toProviderErr(io.EOF)
	var providerErr *fantasy.ProviderError
	if errors.As(got, &providerErr) {
		t.Errorf("toProviderErr wrapped io.EOF as ProviderError; should pass through")
	}
}

func TestToProviderErr_FlagsExpiredBedrockCredentials(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		err  error
	}{
		{"direct", errors.New("failed to refresh cached credentials")},
		{"wrapped", fmt.Errorf("operation error Bedrock: %w", errors.New("failed to refresh cached credentials"))},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			var providerErr *fantasy.ProviderError
			if !errors.As(toProviderErr(tc.err), &providerErr) {
				t.Fatalf("toProviderErr did not wrap %v as *fantasy.ProviderError", tc.err)
			}
			if !providerErr.AuthError {
				t.Error("expected AuthError flag so OnAuthRefresh engages")
			}
		})
	}
}

// A mid-stream SSE error event rides inside a 200 response, so only the
// payload marks the failure as temporary.
func TestToProviderErr_RetriesMidStreamOverload(t *testing.T) {
	t.Parallel()

	err := errors.New(`received error while streaming: {"type":"error","error":{"details":null,"type":"overloaded_error","message":"Overloaded"}}`)

	var providerErr *fantasy.ProviderError
	if !errors.As(toProviderErr(err), &providerErr) {
		t.Fatalf("toProviderErr did not wrap %v as *fantasy.ProviderError", err)
	}
	if !providerErr.IsRetryable() {
		t.Error("a mid-stream overload must be retryable so the step is re-run")
	}
	if !providerErr.TransientError {
		t.Error("TransientError must be set for a transient stream error")
	}
	if providerErr.StatusCode != 0 {
		t.Errorf("StatusCode = %d, want 0 (no HTTP status was returned)", providerErr.StatusCode)
	}
	if providerErr.Title != "provider overloaded" {
		t.Errorf("Title = %q, want %q", providerErr.Title, "provider overloaded")
	}
	if providerErr.Message != "Overloaded" {
		t.Errorf("Message = %q, want %q", providerErr.Message, "Overloaded")
	}
	if !errors.Is(providerErr.Cause, err) {
		t.Error("Cause chain must include the original error")
	}
}

func TestToProviderErr_StreamErrorTransientTypes(t *testing.T) {
	t.Parallel()

	for _, errType := range []string{"overloaded_error", "api_error", "server_error", "internal_error", "rate_limit_error"} {
		t.Run(errType, func(t *testing.T) {
			t.Parallel()

			err := errors.New(`received error while streaming: {"type":"error","error":{"type":"` + errType + `","message":"transient"}}`)

			var providerErr *fantasy.ProviderError
			if !errors.As(toProviderErr(err), &providerErr) {
				t.Fatalf("toProviderErr did not wrap %v as *fantasy.ProviderError", err)
			}
			if !providerErr.IsRetryable() {
				t.Errorf("%s stream failure must be retryable", errType)
			}
		})
	}
}

func TestToProviderErr_PermanentStreamErrorNotRetried(t *testing.T) {
	t.Parallel()

	err := errors.New(`received error while streaming: {"type":"error","error":{"type":"invalid_request_error","message":"bad tool schema"}}`)

	var providerErr *fantasy.ProviderError
	if !errors.As(toProviderErr(err), &providerErr) {
		t.Fatalf("toProviderErr did not wrap %v as *fantasy.ProviderError", err)
	}
	if providerErr.IsRetryable() {
		t.Error("a permanent stream error type must not be retryable")
	}
	if providerErr.Message != "bad tool schema" {
		t.Errorf("Message = %q, want %q", providerErr.Message, "bad tool schema")
	}
}

func TestToProviderErr_StreamErrorWrappedByOuterError(t *testing.T) {
	t.Parallel()

	inner := errors.New(`received error while streaming: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`)
	err := fmt.Errorf("anthropic: %w", inner)

	var providerErr *fantasy.ProviderError
	if !errors.As(toProviderErr(err), &providerErr) {
		t.Fatalf("toProviderErr did not wrap %v as *fantasy.ProviderError", err)
	}
	if !providerErr.IsRetryable() {
		t.Error("a wrapped mid-stream overload must still be retryable")
	}
}
