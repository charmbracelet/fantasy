package bedrock

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
			_ = json.NewEncoder(w).Encode(mockAnthropicResponse())
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

		p, err := New(
			WithAPIKey("k"),
			WithSkipAuth(true),
			WithHTTPClient(&http.Client{Transport: redirectTransport(server.URL)}),
		)
		require.NoError(t, err)
		model, _ := p.LanguageModel(t.Context(), "us.anthropic.claude-sonnet-4-20250514-v1:0")
		_, _ = model.Generate(t.Context(), fantasy.Call{Prompt: prompt})

		require.Len(t, *captured, 1)
		assert.Equal(t, "Charm Fantasy/"+fantasy.Version, (*captured)[0]["User-Agent"])
	})

	t.Run("WithUserAgent wins over default", func(t *testing.T) {
		t.Parallel()
		server, captured := newUAServer()
		defer server.Close()

		p, err := New(
			WithAPIKey("k"),
			WithSkipAuth(true),
			WithHTTPClient(&http.Client{Transport: redirectTransport(server.URL)}),
			WithUserAgent("explicit-ua"),
		)
		require.NoError(t, err)
		model, _ := p.LanguageModel(t.Context(), "us.anthropic.claude-sonnet-4-20250514-v1:0")
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
			WithSkipAuth(true),
			WithHTTPClient(&http.Client{Transport: redirectTransport(server.URL)}),
			WithHeaders(map[string]string{"User-Agent": "from-headers"}),
			WithUserAgent("explicit-ua"),
		)
		require.NoError(t, err)
		model, _ := p.LanguageModel(t.Context(), "us.anthropic.claude-sonnet-4-20250514-v1:0")
		_, _ = model.Generate(t.Context(), fantasy.Call{Prompt: prompt})

		require.Len(t, *captured, 1)
		assert.Equal(t, "explicit-ua", (*captured)[0]["User-Agent"])
	})
}

type redirectRoundTripper struct {
	target string
}

func redirectTransport(target string) *redirectRoundTripper {
	return &redirectRoundTripper{target: target}
}

func (rt *redirectRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	req = req.Clone(req.Context())
	req.URL.Scheme = "http"
	req.URL.Host = (&redirectRoundTripper{target: rt.target}).host()
	return http.DefaultTransport.RoundTrip(req)
}

func (rt *redirectRoundTripper) host() string {
	u := rt.target
	if len(u) > 7 && u[:7] == "http://" {
		return u[7:]
	}
	if len(u) > 8 && u[:8] == "https://" {
		return u[8:]
	}
	return u
}

func mockAnthropicResponse() map[string]any {
	return map[string]any{
		"id":    "msg_01Test",
		"type":  "message",
		"role":  "assistant",
		"model": "claude-sonnet-4-20250514",
		"content": []any{
			map[string]any{
				"type": "text",
				"text": "Hi there",
			},
		},
		"stop_reason":   "end_turn",
		"stop_sequence": "",
		"usage": map[string]any{
			"cache_creation": map[string]any{
				"ephemeral_1h_input_tokens": 0,
				"ephemeral_5m_input_tokens": 0,
			},
			"cache_creation_input_tokens": 0,
			"cache_read_input_tokens":     0,
			"input_tokens":                5,
			"output_tokens":               2,
			"server_tool_use": map[string]any{
				"web_search_requests": 0,
			},
			"service_tier": "standard",
		},
	}
}
