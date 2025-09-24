package providertests

import (
	"net/http"
	"os"
	"testing"

	"github.com/charmbracelet/fantasy/ai"
	"github.com/charmbracelet/fantasy/anthropic"
	"github.com/stretchr/testify/require"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func TestAnthropicCommon(t *testing.T) {
	testCommon(t, []builderPair{
		{"claude-sonnet-4", builderAnthropicClaudeSonnet4, nil},
	})

	opts := ai.ProviderOptions{
		anthropic.Name: &anthropic.ProviderOptions{
			Thinking: &anthropic.ThinkingProviderOption{
				BudgetTokens: 4000,
			},
		},
	}
	testThinking(t, []builderPair{
		{"claude-sonnet-4", builderAnthropicClaudeSonnet4, opts},
	}, testGoogleThinking)
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

func builderAnthropicClaudeSonnet4(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := anthropic.New(
		anthropic.WithAPIKey(os.Getenv("ANTHROPIC_API_KEY")),
		anthropic.WithHTTPClient(&http.Client{Transport: r}),
	)
	return provider.LanguageModel("claude-sonnet-4-20250514")
}
