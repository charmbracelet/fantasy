package providertests

import (
	"testing"

	"github.com/charmbracelet/fantasy/ai"
	"github.com/charmbracelet/fantasy/anthropic"
	"github.com/charmbracelet/fantasy/google"
	"github.com/stretchr/testify/require"
)

func testThinkingSteps(t *testing.T, providerName string, steps []ai.StepResult) {
	switch providerName {
	case anthropic.Name:
		testAnthropicThinking(t, steps)
	case google.Name:
		testGoogleThinking(t, steps)
	}
}

func testGoogleThinking(t *testing.T, steps []ai.StepResult) {
	reasoningContentCount := 0
	// Test if we got the signature
	for _, step := range steps {
		for _, msg := range step.Messages {
			for _, content := range msg.Content {
				if content.GetType() == ai.ContentTypeReasoning {
					reasoningContentCount += 1
				}
			}
		}
	}
	require.Greater(t, reasoningContentCount, 0)
}

func testAnthropicThinking(t *testing.T, steps []ai.StepResult) {
	reasoningContentCount := 0
	signaturesCount := 0
	// Test if we got the signature
	for _, step := range steps {
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
