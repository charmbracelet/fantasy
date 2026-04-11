package google

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
