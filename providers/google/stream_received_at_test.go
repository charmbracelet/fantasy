package google

import (
	"testing"
	"time"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestStreamReceivedAtBeforeConsumerDelay(t *testing.T) {
	t.Parallel()
	server := newJSONModeServer(t, "STOP")
	defer server.Close()
	model := objectStreamModel(t, server.URL)
	stream, err := model.Stream(t.Context(), fantasy.Call{Prompt: objectStreamCall().Prompt})
	require.NoError(t, err)
	var firstReceived, consumerResumed time.Time
	var found bool
	for part := range stream {
		require.NotEqual(t, fantasy.StreamPartTypeError, part.Type)
		switch part.Type {
		case fantasy.StreamPartTypeTextStart:
			require.False(t, part.ReceivedAt.IsZero())
			firstReceived = part.ReceivedAt
			time.Sleep(20 * time.Millisecond)
			consumerResumed = time.Now()
		case fantasy.StreamPartTypeTextDelta:
			require.Equal(t, firstReceived, part.ReceivedAt)
			require.True(t, part.ReceivedAt.Before(consumerResumed))
			found = true
		}
	}
	require.True(t, found)
}
