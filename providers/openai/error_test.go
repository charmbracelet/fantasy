package openai

import (
	"errors"
	"fmt"
	"io"
	"testing"

	"charm.land/fantasy"
	"github.com/openai/openai-go/v3/packages/ssestream"
)

func TestToProviderErr_WrapsUnexpectedEOF(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		err  error
	}{
		{"direct", io.ErrUnexpectedEOF},
		{"wrapped", fmt.Errorf("read stream: %w", io.ErrUnexpectedEOF)},
		{"double_wrapped", fmt.Errorf("openai: %w", fmt.Errorf("sse: %w", io.ErrUnexpectedEOF))},
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

// Regression test for a real reported bug: a mid-stream in-band SSE error
// event (the OpenAI-compatible provider's own envelope, not an HTTP-level
// error) must be classified as retryable when it reports a transient
// server_error, otherwise it silently skips retry.go entirely instead of
// just being marked non-retryable.
func TestToProviderErr_StreamErrorServerErrorIsRetryable(t *testing.T) {
	t.Parallel()

	streamErr := &ssestream.StreamError{
		Message: `received error while streaming: {"message":"Streaming response failed","type":"server_error","param":null,"code":"server_error"}`,
		Event: ssestream.Event{
			Data: []byte(`{"error":{"message":"Streaming response failed","type":"server_error","param":null,"code":"server_error"}}`),
		},
	}

	got := toProviderErr(streamErr)

	var providerErr *fantasy.ProviderError
	if !errors.As(got, &providerErr) {
		t.Fatalf("toProviderErr did not wrap StreamError as *fantasy.ProviderError (got %T)", got)
	}
	if !providerErr.IsRetryable() {
		t.Error("server_error stream failure must be retryable so retry.go engages")
	}
	if providerErr.Message != "Streaming response failed" {
		t.Errorf("Message = %q, want the parsed error body message", providerErr.Message)
	}
	if !errors.Is(providerErr.Cause, streamErr) {
		t.Errorf("Cause chain must include the original *ssestream.StreamError")
	}
}

func TestToProviderErr_StreamErrorUnknownTypeIsNotRetryable(t *testing.T) {
	t.Parallel()

	streamErr := &ssestream.StreamError{
		Message: `received error while streaming: {"message":"bad request","type":"invalid_request_error","param":null,"code":"invalid_request_error"}`,
		Event: ssestream.Event{
			Data: []byte(`{"error":{"message":"bad request","type":"invalid_request_error","param":null,"code":"invalid_request_error"}}`),
		},
	}

	got := toProviderErr(streamErr)

	var providerErr *fantasy.ProviderError
	if !errors.As(got, &providerErr) {
		t.Fatalf("toProviderErr did not wrap StreamError as *fantasy.ProviderError (got %T)", got)
	}
	if providerErr.IsRetryable() {
		t.Error("a non-transient stream error type must not be retryable")
	}
}

func TestToProviderErr_StreamErrorMalformedBodyFallsBackToRawMessage(t *testing.T) {
	t.Parallel()

	streamErr := &ssestream.StreamError{
		Message: "received error while streaming: not json",
		Event:   ssestream.Event{Data: []byte("not json")},
	}

	got := toProviderErr(streamErr)

	var providerErr *fantasy.ProviderError
	if !errors.As(got, &providerErr) {
		t.Fatalf("toProviderErr did not wrap StreamError as *fantasy.ProviderError (got %T)", got)
	}
	if providerErr.Message != streamErr.Message {
		t.Errorf("Message = %q, want fallback to raw StreamError.Message %q", providerErr.Message, streamErr.Message)
	}
	if providerErr.IsRetryable() {
		t.Error("unparseable stream error body must not be assumed retryable")
	}
}
