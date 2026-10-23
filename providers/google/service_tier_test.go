package google

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestServiceTier(t *testing.T) {
	t.Parallel()

	prompt := fantasy.Prompt{
		{
			Role:    fantasy.MessageRoleUser,
			Content: []fantasy.MessagePart{fantasy.TextPart{Text: "Hi"}},
		},
	}

	// call runs a single Generate against a stub server and returns the
	// decoded request body plus any warnings surfaced on the response.
	call := func(t *testing.T, model string, opts *ProviderOptions) (map[string]any, []fantasy.CallWarning) {
		t.Helper()

		var body map[string]any
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			raw, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(raw, &body)
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"candidates": []map[string]any{
					{
						"content": map[string]any{
							"role":  "model",
							"parts": []map[string]any{{"text": "Hello"}},
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
		t.Cleanup(server.Close)

		p, err := New(
			WithVertex("test-project", "us-central1"),
			WithBaseURL(server.URL),
			WithSkipAuth(true),
		)
		require.NoError(t, err)
		lm, err := p.LanguageModel(t.Context(), model)
		require.NoError(t, err)

		c := fantasy.Call{Prompt: prompt}
		if opts != nil {
			c.ProviderOptions = fantasy.ProviderOptions{Name: opts}
		}
		resp, err := lm.Generate(t.Context(), c)
		require.NoError(t, err)
		return body, resp.Warnings
	}

	t.Run("flex sent for supported model", func(t *testing.T) {
		t.Parallel()
		body, warnings := call(t, "gemini-2.5-flash", &ProviderOptions{ServiceTier: ServiceTierFlex})
		assert.Equal(t, "flex", body["serviceTier"])
		assert.Empty(t, warnings)
	})

	t.Run("omitted by default", func(t *testing.T) {
		t.Parallel()
		body, _ := call(t, "gemini-2.5-flash", nil)
		_, ok := body["serviceTier"]
		assert.False(t, ok)
	})

	t.Run("flex warns and is dropped for unsupported model", func(t *testing.T) {
		t.Parallel()
		body, warnings := call(t, "gemini-1.5-flash", &ProviderOptions{ServiceTier: ServiceTierFlex})
		_, ok := body["serviceTier"]
		assert.False(t, ok)
		require.NotEmpty(t, warnings)
		assert.Equal(t, fantasy.CallWarningTypeUnsupportedSetting, warnings[0].Type)
		assert.Equal(t, "ServiceTier", warnings[0].Setting)
	})
}
