package kronk

import (
	"context"
	"errors"
	"fmt"
	"io"
	"testing"

	"charm.land/fantasy"
	krn "github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
)

func TestToProviderErr(t *testing.T) {
	tests := []struct {
		name             string
		err              error
		wantProviderErr  bool
		wantRetryable    bool
		wantContextLarge bool
		wantUsedTokens   int
		wantMaxTokens    int
	}{
		{name: "nil"},
		{name: "canceled", err: context.Canceled},
		{name: "deadline", err: context.DeadlineExceeded},
		{name: "invalid", err: fmt.Errorf("chat: %w", model.ErrInvalidRequest), wantProviderErr: true},
		{
			name:             "context limit",
			err:              fmt.Errorf("chat: %w: input tokens [9000] exceed context window [8192]", model.ErrInvalidRequest),
			wantProviderErr:  true,
			wantContextLarge: true,
			wantUsedTokens:   9000,
			wantMaxTokens:    8192,
		},
		{name: "native context full", err: errors.New("unable to process request: the context window is full"), wantProviderErr: true, wantContextLarge: true},
		{name: "admission timeout", err: krn.ErrAdmissionTimeout, wantProviderErr: true, wantRetryable: true},
		{name: "transport", err: io.ErrUnexpectedEOF, wantProviderErr: true, wantRetryable: true},
		{name: "other", err: errors.New("model failed"), wantProviderErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := toProviderErr(tt.err)
			var providerErr *fantasy.ProviderError
			if gotProviderErr := errors.As(got, &providerErr); gotProviderErr != tt.wantProviderErr {
				t.Fatalf("ProviderError: got %t, want %t", gotProviderErr, tt.wantProviderErr)
			}
			if !tt.wantProviderErr {
				if !errors.Is(got, tt.err) {
					t.Errorf("error: got %v, want %v", got, tt.err)
				}
				return
			}
			if got := providerErr.IsRetryable(); got != tt.wantRetryable {
				t.Errorf("IsRetryable: got %t, want %t", got, tt.wantRetryable)
			}
			if got := providerErr.IsContextTooLarge(); got != tt.wantContextLarge {
				t.Errorf("IsContextTooLarge: got %t, want %t", got, tt.wantContextLarge)
			}
			if got := providerErr.ContextUsedTokens; got != tt.wantUsedTokens {
				t.Errorf("ContextUsedTokens: got %d, want %d", got, tt.wantUsedTokens)
			}
			if got := providerErr.ContextMaxTokens; got != tt.wantMaxTokens {
				t.Errorf("ContextMaxTokens: got %d, want %d", got, tt.wantMaxTokens)
			}
		})
	}
}

func TestToOperationErr(t *testing.T) {
	baseErr := errors.New("load failed")
	err := toOperationErr("kronk model loading failed", baseErr)

	var providerErr *fantasy.ProviderError
	if !errors.As(err, &providerErr) {
		t.Fatalf("error: got %T, want *fantasy.ProviderError", err)
	}
	if got, want := providerErr.Title, "kronk model loading failed"; got != want {
		t.Errorf("Title: got %q, want %q", got, want)
	}
	if !errors.Is(err, baseErr) {
		t.Errorf("Cause: got %v, want %v", err, baseErr)
	}
}
