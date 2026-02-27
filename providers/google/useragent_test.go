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
			_ = json.NewEncoder(w).Encode(map[string]any{
				"candidates": []map[string]any{
					{
						"content": map[string]any{
							"role": "model",
							"parts": []map[string]any{
								{"text": "Hello"},
							},
						},
						"finishReason": "STOP",
					},
				},
				"usageMetadata": map[string]any{
					"promptTokenCount":     5,
					"candidatesTokenCount": 2,
					"totalTokenCount":      7,
				},
			})
		}))
		return server, &captured
	}

	prompt := fantasy.Prompt{
		{
			Role:    fantasy.MessageRoleUser,
			Content: []fantasy.MessagePart{fantasy.TextPart{Text: "Hi"}},
		},
	}

	findUA := func(captured *[]map[string]string, want string) bool {
		for _, h := range *captured {
			if ua, ok := h["User-Agent"]; ok && ua == want {
				return true
			}
		}
		return false
	}

	t.Run("default UA applied", func(t *testing.T) {
		t.Parallel()
		server, captured := newUAServer()
		defer server.Close()

		p, err := New(
			WithVertex("test-project", "us-central1"),
			WithBaseURL(server.URL),
			WithSkipAuth(true),
		)
		require.NoError(t, err)
		model, err := p.LanguageModel(t.Context(), "gemini-2.0-flash")
		require.NoError(t, err)
		_, _ = model.Generate(t.Context(), fantasy.Call{Prompt: prompt})

		require.NotEmpty(t, *captured)
		assert.True(t, findUA(captured, "Charm Fantasy/"+fantasy.Version))
	})

	t.Run("agent segment format", func(t *testing.T) {
		t.Parallel()
		server, captured := newUAServer()
		defer server.Close()

		p, err := New(
			WithVertex("test-project", "us-central1"),
			WithBaseURL(server.URL),
			WithSkipAuth(true),
			WithAgentSegment("Claude 4.6 Opus"),
		)
		require.NoError(t, err)
		model, err := p.LanguageModel(t.Context(), "gemini-2.0-flash")
		require.NoError(t, err)
		_, _ = model.Generate(t.Context(), fantasy.Call{Prompt: prompt})

		require.NotEmpty(t, *captured)
		assert.True(t, findUA(captured, "Charm Fantasy/"+fantasy.Version+" (Claude 4.6 Opus)"))
	})

	t.Run("WithUserAgent wins over default", func(t *testing.T) {
		t.Parallel()
		server, captured := newUAServer()
		defer server.Close()

		p, err := New(
			WithVertex("test-project", "us-central1"),
			WithBaseURL(server.URL),
			WithSkipAuth(true),
			WithUserAgent("explicit-ua"),
		)
		require.NoError(t, err)
		model, err := p.LanguageModel(t.Context(), "gemini-2.0-flash")
		require.NoError(t, err)
		_, _ = model.Generate(t.Context(), fantasy.Call{Prompt: prompt})

		require.NotEmpty(t, *captured)
		assert.True(t, findUA(captured, "explicit-ua"))
	})

	t.Run("WithHeaders User-Agent wins over default", func(t *testing.T) {
		t.Parallel()
		server, captured := newUAServer()
		defer server.Close()

		p, err := New(
			WithVertex("test-project", "us-central1"),
			WithBaseURL(server.URL),
			WithSkipAuth(true),
			WithHeaders(map[string]string{"User-Agent": "custom-from-headers"}),
		)
		require.NoError(t, err)
		model, err := p.LanguageModel(t.Context(), "gemini-2.0-flash")
		require.NoError(t, err)
		_, _ = model.Generate(t.Context(), fantasy.Call{Prompt: prompt})

		require.NotEmpty(t, *captured)
		assert.True(t, findUA(captured, "custom-from-headers"))
	})

	t.Run("WithUserAgent wins over WithHeaders", func(t *testing.T) {
		t.Parallel()
		server, captured := newUAServer()
		defer server.Close()

		p, err := New(
			WithVertex("test-project", "us-central1"),
			WithBaseURL(server.URL),
			WithSkipAuth(true),
			WithHeaders(map[string]string{"User-Agent": "from-headers"}),
			WithUserAgent("explicit-ua"),
		)
		require.NoError(t, err)
		model, err := p.LanguageModel(t.Context(), "gemini-2.0-flash")
		require.NoError(t, err)
		_, _ = model.Generate(t.Context(), fantasy.Call{Prompt: prompt})

		require.NotEmpty(t, *captured)
		assert.True(t, findUA(captured, "explicit-ua"))
	})

	t.Run("WithAgentSegment empty clears segment", func(t *testing.T) {
		t.Parallel()
		server, captured := newUAServer()
		defer server.Close()

		p, err := New(
			WithVertex("test-project", "us-central1"),
			WithBaseURL(server.URL),
			WithSkipAuth(true),
			WithAgentSegment("initial"),
			WithAgentSegment(""),
		)
		require.NoError(t, err)
		model, err := p.LanguageModel(t.Context(), "gemini-2.0-flash")
		require.NoError(t, err)
		_, _ = model.Generate(t.Context(), fantasy.Call{Prompt: prompt})

		require.NotEmpty(t, *captured)
		assert.True(t, findUA(captured, "Charm Fantasy/"+fantasy.Version))
	})
}
