package atlascloud

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestProviderName(t *testing.T) {
	t.Parallel()

	provider, err := New(WithAPIKey("k"))
	require.NoError(t, err)
	model, err := provider.LanguageModel(t.Context(), "qwen/qwen3.5-flash")
	require.NoError(t, err)

	assert.Equal(t, Name, model.Provider())
}

func TestDefaultRequest(t *testing.T) {
	t.Parallel()

	var captured []map[string]string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h := make(map[string]string)
		for k, v := range r.Header {
			if len(v) > 0 {
				h[k] = v[0]
			}
		}
		captured = append(captured, h)

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":      "chatcmpl-test",
			"object":  "chat.completion",
			"created": 1711115037,
			"model":   "qwen/qwen3.5-flash",
			"choices": []map[string]any{
				{
					"index": 0,
					"message": map[string]any{
						"role":    "assistant",
						"content": "Hi there",
					},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]any{
				"prompt_tokens":     4,
				"completion_tokens": 2,
				"total_tokens":      6,
			},
		})
	}))
	defer server.Close()

	provider, err := New(WithAPIKey("k"), WithBaseURL(server.URL))
	require.NoError(t, err)

	model, err := provider.LanguageModel(t.Context(), "qwen/qwen3.5-flash")
	require.NoError(t, err)

	_, err = model.Generate(t.Context(), fantasy.Call{
		Prompt: fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Hi"},
				},
			},
		},
	})
	require.NoError(t, err)

	require.Len(t, captured, 1)
	assert.Equal(t, "Bearer k", captured[0]["Authorization"])
	assert.True(t, strings.HasPrefix(captured[0]["User-Agent"], "Charm-Fantasy/"))
}
