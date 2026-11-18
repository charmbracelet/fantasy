package fantasy_test

import (
	"encoding/json"
	"testing"
	"time"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestStreamPartReceivedAtJSON(t *testing.T) {
	t.Parallel()
	t.Run("round trip", func(t *testing.T) {
		receivedAt := time.Date(2026, time.November, 10, 12, 0, 0, 123456789, time.UTC)
		part := fantasy.StreamPart{Type: fantasy.StreamPartTypeTextDelta, Delta: "hello", ReceivedAt: receivedAt}
		data, err := json.Marshal(part)
		require.NoError(t, err)
		var decoded fantasy.StreamPart
		require.NoError(t, json.Unmarshal(data, &decoded))
		require.Equal(t, part, decoded)
	})
	t.Run("zero omitted", func(t *testing.T) {
		data, err := json.Marshal(fantasy.StreamPart{Type: fantasy.StreamPartTypeTextStart})
		require.NoError(t, err)
		require.NotContains(t, string(data), "received_at")
	})
	t.Run("legacy payload", func(t *testing.T) {
		var part fantasy.StreamPart
		require.NoError(t, json.Unmarshal([]byte(`{"type":"text-delta","delta":"hello"}`), &part))
		require.True(t, part.ReceivedAt.IsZero())
	})
}
