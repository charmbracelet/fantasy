package providertests

import (
	"net/http"
	"os"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/openrouter"
	"github.com/stretchr/testify/require"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

var openrouterTestModels = []testModel{
	{"kimi-k2", "moonshotai/kimi-k2-0905", false},
	{"grok-code-fast-1", "x-ai/grok-code-fast-1", false},
	{"claude-sonnet-4", "anthropic/claude-sonnet-4", true},
	{"gemini-2.5-flash", "google/gemini-2.5-flash", false},
	{"deepseek-chat-v3.1-free", "deepseek/deepseek-chat-v3.1:free", false},
	{"qwen3-235b-a22b-2507", "qwen/qwen3-235b-a22b-2507", false},
	{"gpt-5", "openai/gpt-5", true},
	{"glm-4.5", "z-ai/glm-4.5", false},
}

func TestOpenRouterCommon(t *testing.T) {
	var pairs []builderPair
	for _, m := range openrouterTestModels {
		pairs = append(pairs, builderPair{m.name, openrouterBuilder(m.model), nil})
	}
	testCommon(t, pairs)
}

func TestOpenRouterThinking(t *testing.T) {
	opts := fantasy.ProviderOptions{
		openrouter.Name: &openrouter.ProviderOptions{
			Reasoning: &openrouter.ReasoningOptions{
				Effort: openrouter.ReasoningEffortOption(openrouter.ReasoningEffortMedium),
			},
		},
	}

	var pairs []builderPair
	for _, m := range openrouterTestModels {
		if !m.reasoning {
			continue
		}
		pairs = append(pairs, builderPair{m.name, openrouterBuilder(m.model), opts})
	}
	testThinking(t, pairs, testOpenrouterThinking)

	// test anthropic signature
	testThinking(t, []builderPair{
		{"claude-sonnet-4-sig", openrouterBuilder("anthropic/claude-sonnet-4"), opts},
	}, testOpenrouterThinkingWithSignature)
}

func testOpenrouterThinkingWithSignature(t *testing.T, result *fantasy.AgentResult) {
	reasoningContentCount := 0
	signaturesCount := 0
	// Test if we got the signature
	for _, step := range result.Steps {
		for _, msg := range step.Messages {
			for _, content := range msg.Content {
				if content.GetType() == fantasy.ContentTypeReasoning {
					reasoningContentCount += 1
					reasoningContent, ok := fantasy.AsContentType[fantasy.ReasoningPart](content)
					if !ok {
						continue
					}
					if len(reasoningContent.ProviderOptions) == 0 {
						continue
					}

					anthropicReasoningMetadata, ok := reasoningContent.ProviderOptions[openrouter.Name]
					if !ok {
						continue
					}
					if reasoningContent.Text != "" {
						if typed, ok := anthropicReasoningMetadata.(*openrouter.ReasoningMetadata); ok {
							require.NotEmpty(t, typed.Signature)
							signaturesCount += 1
						}
					}
				}
			}
		}
	}
	require.Greater(t, reasoningContentCount, 0)
	require.Greater(t, signaturesCount, 0)
	require.Equal(t, reasoningContentCount, signaturesCount)
	// we also add the anthropic metadata so test that
	testAnthropicThinking(t, result)
}

func testOpenrouterThinking(t *testing.T, result *fantasy.AgentResult) {
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
	require.Greater(t, reasoningContentCount, 0)
}

func openrouterBuilder(model string) builderFunc {
	return func(r *recorder.Recorder) (fantasy.LanguageModel, error) {
		provider := openrouter.New(
			openrouter.WithAPIKey(os.Getenv("FANTASY_OPENROUTER_API_KEY")),
			openrouter.WithHTTPClient(&http.Client{Transport: r}),
		)
		return provider.LanguageModel(model)
	}
}
