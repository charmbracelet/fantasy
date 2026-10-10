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

// ErrorType carries Google's own name for the failure, so a caller can tell a
// quota refusal from a bad request without matching on prose.
func TestToProviderErr_CarriesTheErrorType(t *testing.T) {
	t.Parallel()

	apiErr := genai.APIError{
		Code:    429,
		Status:  "RESOURCE_EXHAUSTED",
		Message: "Resource has been exhausted",
	}

	var providerErr *fantasy.ProviderError
	if !errors.As(toProviderErr(apiErr), &providerErr) {
		t.Fatalf("toProviderErr did not wrap %v as *fantasy.ProviderError", apiErr)
	}
	if providerErr.ErrorType != "RESOURCE_EXHAUSTED" {
		t.Errorf("ErrorType = %q, want %q", providerErr.ErrorType, "RESOURCE_EXHAUSTED")
	}
}

// Gemini's retry hint arrives as a RetryInfo detail rather than a header, and
// must come out as the lowercase retry-after header retry.go looks up.
func TestToProviderErr_SurfacesRetryInfoDelay(t *testing.T) {
	t.Parallel()

	apiErr := genai.APIError{
		Code:    429,
		Status:  "RESOURCE_EXHAUSTED",
		Message: "Resource has been exhausted",
		Details: []map[string]any{
			{
				"@type":      "type.googleapis.com/google.rpc.RetryInfo",
				"retryDelay": "38s",
			},
		},
	}

	var providerErr *fantasy.ProviderError
	if !errors.As(toProviderErr(apiErr), &providerErr) {
		t.Fatalf("toProviderErr did not wrap %v as *fantasy.ProviderError", apiErr)
	}
	if got := providerErr.ResponseHeaders["retry-after"]; got != "38" {
		t.Errorf(`ResponseHeaders["retry-after"] = %q, want %q`, got, "38")
	}
}

// Without a RetryInfo detail there is no hint to pass on, and an invented
// header would override retry.go's own backoff.
func TestToProviderErr_NoRetryInfoLeavesHeadersNil(t *testing.T) {
	t.Parallel()

	apiErr := genai.APIError{Code: 500, Message: "internal error"}

	var providerErr *fantasy.ProviderError
	if !errors.As(toProviderErr(apiErr), &providerErr) {
		t.Fatalf("toProviderErr did not wrap %v as *fantasy.ProviderError", apiErr)
	}
	if providerErr.ResponseHeaders != nil {
		t.Errorf("ResponseHeaders = %v, want nil", providerErr.ResponseHeaders)
	}
}
