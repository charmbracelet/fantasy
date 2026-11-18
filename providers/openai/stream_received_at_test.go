package openai

import (
	"testing"
	"time"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestStreamReceivedAtBeforeConsumerDelay(t *testing.T) {
	t.Parallel()
	server := newStreamingMockServer()
	defer server.close()
	server.prepareStreamResponse(map[string]any{
		"content":       []string{"Hello", " world"},
		"finish_reason": "stop",
	})
	provider, err := New(WithAPIKey("test-api-key"), WithBaseURL(server.server.URL))
	require.NoError(t, err)
	model, err := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")
	require.NoError(t, err)
	stream, err := model.Stream(t.Context(), fantasy.Call{Prompt: testPrompt})
	require.NoError(t, err)

	started := time.Now()
	var firstReceived, consumerResumed time.Time
	var deltas int
	for part := range stream {
		require.NotEqual(t, fantasy.StreamPartTypeError, part.Type)
		require.False(t, part.ReceivedAt.IsZero())
		require.False(t, part.ReceivedAt.Before(started))
		switch part.Type {
		case fantasy.StreamPartTypeTextStart:
			firstReceived = part.ReceivedAt
			time.Sleep(20 * time.Millisecond)
			consumerResumed = time.Now()
		case fantasy.StreamPartTypeTextDelta:
			require.Equal(t, firstReceived, part.ReceivedAt)
			require.True(t, part.ReceivedAt.Before(consumerResumed))
			deltas++
		}
	}
	require.Equal(t, 2, deltas)
}
