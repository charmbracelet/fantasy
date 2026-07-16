package openai

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

// TestStreamHTTP2StreamReset proves the full chain end to end: a real
// HTTP/2 RST_STREAM sent mid-response surfaces as a retryable
// ProviderError so the agent's retry middleware re-runs the step.
//
// The server streams a partial SSE chunk, flushes, then aborts the
// handler. On an HTTP/2 connection this makes the transport send an
// RST_STREAM(INTERNAL_ERROR) to the client, which is exactly the
// "stream error: stream ID N; INTERNAL_ERROR; received from peer"
// failure observed in production.
func TestStreamHTTP2StreamReset(t *testing.T) {
	t.Parallel()

	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		// Stream a valid partial chunk so the client is mid-stream.
		fmt.Fprint(w, "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\","+
			"\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\n")
		w.(http.Flusher).Flush()
		// Abort mid-stream: on HTTP/2 this emits RST_STREAM to the peer.
		panic(http.ErrAbortHandler)
	}))
	srv.EnableHTTP2 = true
	srv.StartTLS()
	defer srv.Close()

	provider, err := New(
		WithAPIKey("test-api-key"),
		WithBaseURL(srv.URL),
		// srv.Client() trusts the test cert and negotiates HTTP/2.
		WithHTTPClient(srv.Client()),
	)
	require.NoError(t, err)

	model, err := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")
	require.NoError(t, err)

	stream, err := model.Stream(context.Background(), fantasy.Call{Prompt: testPrompt})
	require.NoError(t, err)

	var streamErr error
	for part := range stream {
		if part.Type == fantasy.StreamPartTypeError {
			streamErr = part.Error
		}
	}

	require.Error(t, streamErr, "expected a stream error from the RST_STREAM")

	// The error must be classified as a transient HTTP/2 transport error.
	require.True(t, fantasy.IsTransportError(streamErr),
		"error should be detected as HTTP/2 transport error, got: %v", streamErr)

	// It must be wrapped as a retryable ProviderError so the retry
	// middleware engages.
	var provErr *fantasy.ProviderError
	require.ErrorAs(t, streamErr, &provErr)
	require.True(t, provErr.IsRetryable(), "provider error should be retryable")
}
