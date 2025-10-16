package providertests

import (
	"net/http"
	"os"
	"testing"

	"charm.land/fantasy/ai"
	"charm.land/fantasy/anthropic"
	"github.com/stretchr/testify/require"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

var anthropicTestModels = []testModel{
	{"claude-sonnet-4", "claude-sonnet-4-20250514", true},
}

func TestAnthropicCommon(t *testing.T) {
	var pairs []builderPair
	for _, m := range anthropicTestModels {
		pairs = append(pairs, builderPair{m.name, anthropicBuilder(m.model), nil})
	}
	testCommon(t, pairs)
}

func TestAnthropicThinking(t *testing.T) {
	opts := ai.ProviderOptions{
		anthropic.Name: &anthropic.ProviderOptions{
			Thinking: &anthropic.ThinkingProviderOption{
				BudgetTokens: 4000,
			},
		},
	}
	var pairs []builderPair
	for _, m := range anthropicTestModels {
		if !m.reasoning {
			continue
		}
		pairs = append(pairs, builderPair{m.name, anthropicBuilder(m.model), opts})
	}
	testThinking(t, pairs, testAnthropicThinking)
}

func testAnthropicThinking(t *testing.T, result *ai.AgentResult) {
	reasoningContentCount := 0
	signaturesCount := 0
	// Test if we got the signature
	for _, step := range result.Steps {
		for _, msg := range step.Messages {
			for _, content := range msg.Content {
				if content.GetType() == ai.ContentTypeReasoning {
					reasoningContentCount += 1
					reasoningContent, ok := ai.AsContentType[ai.ReasoningPart](content)
					if !ok {
						continue
					}
					if len(reasoningContent.ProviderOptions) == 0 {
						continue
					}

					anthropicReasoningMetadata, ok := reasoningContent.ProviderOptions[anthropic.Name]
					if !ok {
						continue
					}
					if reasoningContent.Text != "" {
						if typed, ok := anthropicReasoningMetadata.(*anthropic.ReasoningOptionMetadata); ok {
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
}

func anthropicBuilder(model string) builderFunc {
	return func(r *recorder.Recorder) (ai.LanguageModel, error) {
		provider := anthropic.New(
			anthropic.WithAPIKey(os.Getenv("FANTASY_ANTHROPIC_API_KEY")),
			anthropic.WithHTTPClient(&http.Client{Transport: r}),
		)
		return provider.LanguageModel(model)
	}
}
