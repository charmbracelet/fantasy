package google

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// newCumulativeUsageServer serves a Gemini stream whose usageMetadata is
// cumulative: every chunk repeats the running totals, the way the API reports
// them, so the last chunk holds the final count. Text chunks carry the given
// parts; the last one also carries finishReason STOP.
func newCumulativeUsageServer(t *testing.T, texts []string) *httptest.Server {
	t.Helper()

	candidates := []int{10, 20, 30}
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for i, text := range texts {
			candidate := map[string]any{
				"content": map[string]any{"role": "model", "parts": []map[string]any{{"text": text}}},
			}
			if i == len(texts)-1 {
				candidate["finishReason"] = "STOP"
			}
			body := map[string]any{
				"candidates": []map[string]any{candidate},
				"usageMetadata": map[string]any{
					"promptTokenCount":     100,
					"candidatesTokenCount": candidates[i],
					"totalTokenCount":      100 + candidates[i],
				},
			}
			encoded, err := json.Marshal(body)
			require.NoError(t, err)
			_, _ = w.Write([]byte("data: " + string(encoded) + "\r\n\r\n"))
		}
	}))
}

// Gemini repeats cumulative usageMetadata on every streamed chunk, so the
// final count is the last chunk's value; summing the chunks inflated
// OutputTokens (10+20+30 instead of 30).
func TestStreamUsageIsCumulativeNotSummed(t *testing.T) {
	t.Parallel()

	server := newCumulativeUsageServer(t, []string{`{"answer":`, `"hel`, `lo"}`})
	defer server.Close()

	stream, err := objectStreamModel(t, server.URL).Stream(t.Context(), fantasy.Call{
		Prompt: fantasy.Prompt{{
			Role:    fantasy.MessageRoleUser,
			Content: []fantasy.MessagePart{fantasy.TextPart{Text: "Hi"}},
		}},
	})
	require.NoError(t, err)

	var finishes []fantasy.StreamPart
	for part := range stream {
		switch part.Type {
		case fantasy.StreamPartTypeFinish:
			finishes = append(finishes, part)
		case fantasy.StreamPartTypeError:
			t.Fatalf("unexpected error part: %v", part.Error)
		}
	}

	require.Len(t, finishes, 1)
	assert.Equal(t, int64(100), finishes[0].Usage.InputTokens)
	assert.Equal(t, int64(30), finishes[0].Usage.OutputTokens)
	assert.Equal(t, int64(130), finishes[0].Usage.TotalTokens)
}

func TestStreamObjectUsageIsCumulativeNotSummed(t *testing.T) {
	t.Parallel()

	server := newCumulativeUsageServer(t, []string{`{"answer":`, `"hel`, `lo"}`})
	defer server.Close()

	stream, err := objectStreamModel(t, server.URL).StreamObject(t.Context(), objectStreamCall())
	require.NoError(t, err)

	var finishes []fantasy.ObjectStreamPart
	for part := range stream {
		switch part.Type {
		case fantasy.ObjectStreamPartTypeFinish:
			finishes = append(finishes, part)
		case fantasy.ObjectStreamPartTypeError:
			t.Fatalf("unexpected error part: %v", part.Error)
		}
	}

	require.Len(t, finishes, 1)
	assert.Equal(t, int64(100), finishes[0].Usage.InputTokens)
	assert.Equal(t, int64(30), finishes[0].Usage.OutputTokens)
}
