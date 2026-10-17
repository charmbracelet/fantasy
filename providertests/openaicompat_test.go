package providertests

import (
	"net/http"
	"os"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openai"
	"charm.land/fantasy/providers/openaicompat"
	"charm.land/x/vcr"
	"github.com/stretchr/testify/require"
)

// openaicompatModel builds a builderFunc for an OpenAI-compatible endpoint.
// An empty apiKeyEnv skips the API-key option (for endpoints that need none).
func openaicompatModel(baseURL, apiKeyEnv, model string) builderFunc {
	return func(t *testing.T, r *vcr.Recorder) (fantasy.LanguageModel, error) {
		opts := []openaicompat.Option{
			openaicompat.WithBaseURL(baseURL),
			openaicompat.WithHTTPClient(&http.Client{Transport: r}),
		}
		if apiKeyEnv != "" {
			opts = append(opts, openaicompat.WithAPIKey(os.Getenv(apiKeyEnv)))
		}
		provider, err := openaicompat.New(opts...)
		if err != nil {
			return nil, err
		}
		return provider.LanguageModel(t.Context(), model)
	}
}

func TestOpenAICompatibleCommon(t *testing.T) {
	testCommon(t, []builderPair{
		{"xai-grok-4-fast", openaicompatModel("https://api.x.ai/v1", "FANTASY_XAI_API_KEY", "grok-4-fast"), nil, nil},
		{"xai-grok-code-fast", openaicompatModel("https://api.x.ai/v1", "FANTASY_XAI_API_KEY", "grok-code-fast-1"), nil, nil},
		{"groq-kimi-k2", openaicompatModel("https://api.groq.com/openai/v1", "FANTASY_GROQ_API_KEY", "moonshotai/kimi-k2-instruct-0905"), nil, nil},
		{"zai-glm-4-7-flash", openaicompatModel("https://api.z.ai/api/coding/paas/v4", "FANTASY_ZAI_API_KEY", "glm-4.7-flash"), nil, nil},
		{"huggingface-kimi-k2", openaicompatModel("https://router.huggingface.co/v1", "FANTASY_HUGGINGFACE_API_KEY", "moonshotai/Kimi-K2-Instruct-0905:groq"), nil, nil},
		{"llama-cpp-gpt-oss", openaicompatModel("http://localhost:8080/v1", "", "openai/gpt-oss-20b"), nil, nil},
		{"avian-kimi", openaicompatModel("https://api.avian.io/v1", "FANTASY_AVIAN_API_KEY", "moonshotai/kimi-k2.5"), nil, nil},
	})
}

func TestOpenAICompatObjectGeneration(t *testing.T) {
	testObjectGeneration(t, []builderPair{
		{"xai-grok-4-fast", openaicompatModel("https://api.x.ai/v1", "FANTASY_XAI_API_KEY", "grok-4-fast"), nil, nil},
		{"xai-grok-code-fast", openaicompatModel("https://api.x.ai/v1", "FANTASY_XAI_API_KEY", "grok-code-fast-1"), nil, nil},
		{"zai-glm-4-7-flash", openaicompatModel("https://api.z.ai/api/coding/paas/v4", "FANTASY_ZAI_API_KEY", "glm-4.7-flash"), nil, nil},
	})
}

func TestOpenAICompatibleThinking(t *testing.T) {
	opts := fantasy.ProviderOptions{
		openaicompat.Name: &openaicompat.ProviderOptions{
			ReasoningEffort: openai.ReasoningEffortOption(openai.ReasoningEffortHigh),
		},
	}
	testThinking(t, []builderPair{
		{"xai-grok-3-mini", openaicompatModel("https://api.x.ai/v1", "FANTASY_XAI_API_KEY", "grok-3-mini"), opts, nil},
		{"zai-glm-4-7-flash", openaicompatModel("https://api.z.ai/api/coding/paas/v4", "FANTASY_ZAI_API_KEY", "glm-4.7-flash"), opts, nil},
		{"llama-cpp-gpt-oss", openaicompatModel("http://localhost:8080/v1", "", "openai/gpt-oss-20b"), opts, nil},
		{"avian-kimi", openaicompatModel("https://api.avian.io/v1", "FANTASY_AVIAN_API_KEY", "moonshotai/kimi-k2.5"), nil, nil},
	}, testOpenAICompatThinking)
}

func testOpenAICompatThinking(t *testing.T, result *fantasy.AgentResult) {
	reasoningContentCount := 0
	for _, step := range result.Steps {
		for _, msg := range step.Messages {
			for _, content := range msg.Content {
				if content.GetType() == fantasy.ContentTypeReasoning {
					reasoningContentCount += 1
				}
			}
		}
	}
	require.Greater(t, reasoningContentCount, 0, "expected reasoning content, got none")
}
