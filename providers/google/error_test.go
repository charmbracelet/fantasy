package google

import (
	"errors"
	"fmt"
	"io"
	"testing"

	"charm.land/fantasy"
	"google.golang.org/genai"
)

func TestToProviderErr_WrapsUnexpectedEOF(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		err  error
	}{
		{"direct", io.ErrUnexpectedEOF},
		{"wrapped", fmt.Errorf("read stream: %w", io.ErrUnexpectedEOF)},
		{"double_wrapped", fmt.Errorf("google: %w", fmt.Errorf("sse: %w", io.ErrUnexpectedEOF))},
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

	got := toProviderErr(io.EOF)
	var providerErr *fantasy.ProviderError
	if errors.As(got, &providerErr) {
		t.Errorf("toProviderErr wrapped io.EOF as ProviderError; should pass through")
	}
}

func TestToProviderErr_SurfacesRetryInfoDelay(t *testing.T) {
	t.Parallel()

	apiErr := genai.APIError{
		Code:    429,
		Message: "Resource has been exhausted",
		Status:  "RESOURCE_EXHAUSTED",
		Details: []map[string]any{
			{
				"@type":      "type.googleapis.com/google.rpc.RetryInfo",
				"retryDelay": "38s",
			},
		},
	}

	providerErr, ok := toProviderErr(apiErr).(*fantasy.ProviderError)
	if !ok {
		t.Fatalf("toProviderErr did not return *fantasy.ProviderError")
	}
	if providerErr.ResponseHeaders == nil {
		t.Fatalf("ResponseHeaders is nil, want a synthesized retry-after header")
	}
	if got := providerErr.ResponseHeaders["retry-after"]; got != "38" {
		t.Errorf("ResponseHeaders[retry-after] = %q, want %q", got, "38")
	}
}

func TestToProviderErr_NoRetryInfoLeavesHeadersNil(t *testing.T) {
	t.Parallel()

	apiErr := genai.APIError{Code: 500, Message: "internal error"}

	providerErr, ok := toProviderErr(apiErr).(*fantasy.ProviderError)
	if !ok {
		t.Fatalf("toProviderErr did not return *fantasy.ProviderError")
	}
	if providerErr.ResponseHeaders != nil {
		t.Errorf("ResponseHeaders = %v, want nil", providerErr.ResponseHeaders)
	}
}
