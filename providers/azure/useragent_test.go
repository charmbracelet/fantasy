package azure

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestUserAgent(t *testing.T) {
	t.Parallel()

	newUAServer := func() (*httptest.Server, *[]map[string]string) {
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
			_ = json.NewEncoder(w).Encode(mockOpenAIResponse())
		}))
		return server, &captured
	}

	prompt := fantasy.Prompt{
		{
			Role:    fantasy.MessageRoleUser,
			Content: []fantasy.MessagePart{fantasy.TextPart{Text: "Hi"}},
		},
	}

	t.Run("default UA applied", func(t *testing.T) {
		t.Parallel()
		server, captured := newUAServer()
		defer server.Close()

		p, err := New(WithAPIKey("k"), WithBaseURL(server.URL))
		require.NoError(t, err)
		model, _ := p.LanguageModel(t.Context(), "gpt-4")
		_, _ = model.Generate(t.Context(), fantasy.Call{Prompt: prompt})

		require.Len(t, *captured, 1)
		assert.Equal(t, "Charm Fantasy/"+fantasy.Version, (*captured)[0]["User-Agent"])
	})

	t.Run("model segment format", func(t *testing.T) {
		t.Parallel()
		server, captured := newUAServer()
		defer server.Close()

		p, err := New(WithAPIKey("k"), WithBaseURL(server.URL), WithModelSegment("Claude 4.6 Opus"))
		require.NoError(t, err)
		model, _ := p.LanguageModel(t.Context(), "gpt-4")
		_, _ = model.Generate(t.Context(), fantasy.Call{Prompt: prompt})

		require.Len(t, *captured, 1)
		assert.Equal(t, "Charm Fantasy/"+fantasy.Version+" (Claude 4.6 Opus)", (*captured)[0]["User-Agent"])
	})

	t.Run("WithUserAgent wins over default", func(t *testing.T) {
		t.Parallel()
		server, captured := newUAServer()
		defer server.Close()

		p, err := New(WithAPIKey("k"), WithBaseURL(server.URL), WithUserAgent("explicit-ua"))
		require.NoError(t, err)
		model, _ := p.LanguageModel(t.Context(), "gpt-4")
		_, _ = model.Generate(t.Context(), fantasy.Call{Prompt: prompt})

		require.Len(t, *captured, 1)
		assert.Equal(t, "explicit-ua", (*captured)[0]["User-Agent"])
	})

	t.Run("WithUserAgent wins over WithHeaders", func(t *testing.T) {
		t.Parallel()
		server, captured := newUAServer()
		defer server.Close()

		p, err := New(
			WithAPIKey("k"),
			WithBaseURL(server.URL),
			WithHeaders(map[string]string{"User-Agent": "from-headers"}),
			WithUserAgent("explicit-ua"),
		)
		require.NoError(t, err)
		model, _ := p.LanguageModel(t.Context(), "gpt-4")
		_, _ = model.Generate(t.Context(), fantasy.Call{Prompt: prompt})

		require.Len(t, *captured, 1)
		assert.Equal(t, "explicit-ua", (*captured)[0]["User-Agent"])
	})
}

func mockOpenAIResponse() map[string]any {
	return map[string]any{
		"id":      "chatcmpl-test",
		"object":  "chat.completion",
		"created": 1711115037,
		"model":   "gpt-4",
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
			"total_tokens":      6,
			"completion_tokens": 2,
		},
	}
}
