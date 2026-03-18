package providertests

import (
	"net/http"
	"os"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/anthropic"
	"charm.land/fantasy/providers/vercel"
	"charm.land/x/vcr"
	"github.com/stretchr/testify/require"
)

var vercelTestModels = []testModel{
	{"claude-sonnet-4", "anthropic/claude-sonnet-4", true},
	{"gemini-2.5-flash", "google/gemini-2.5-flash", false},
	{"gpt-5", "openai/gpt-5", true},
	{"gemini-3-pro-preview", "google/gemini-3-pro-preview", true},
}

func TestVercelCommon(t *testing.T) {
	var pairs []builderPair
	for _, m := range vercelTestModels {
		pairs = append(pairs, builderPair{m.name, vercelBuilder(m.model), nil, nil})
	}
	testCommon(t, pairs)
}

func TestVercelCommonWithAnthropicCache(t *testing.T) {
	testCommon(t, []builderPair{
		{"claude-sonnet-4", vercelBuilder("anthropic/claude-sonnet-4"), nil, addAnthropicCaching},
	})
}

func TestVercelThinking(t *testing.T) {
	enabled := true
	opts := fantasy.ProviderOptions{
		vercel.Name: &vercel.ProviderOptions{
			Reasoning: &vercel.ReasoningOptions{
				Enabled: &enabled,
			},
		},
	}

	var pairs []builderPair
	for _, m := range vercelTestModels {
		if !m.reasoning {
			continue
		}
		pairs = append(pairs, builderPair{m.name, vercelBuilder(m.model), opts, nil})
	}
	testThinking(t, pairs, testVercelThinking)

	// test anthropic signature
	testThinking(t, []builderPair{
		{"claude-sonnet-4-sig", vercelBuilder("anthropic/claude-sonnet-4"), opts, nil},
	}, testVercelThinkingWithSignature)
}

func testVercelThinkingWithSignature(t *testing.T, result *fantasy.AgentResult) {
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
	// we also add the anthropic metadata so test that
	testAnthropicThinking(t, result)
}

func testVercelThinking(t *testing.T, result *fantasy.AgentResult) {
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

func vercelBuilder(model string) builderFunc {
	return func(t *testing.T, r *vcr.Recorder) (fantasy.LanguageModel, error) {
		provider, err := vercel.New(
			vercel.WithAPIKey(os.Getenv("FANTASY_VERCEL_API_KEY")),
			vercel.WithHTTPClient(&http.Client{Transport: r}),
		)
		if err != nil {
			return nil, err
		}
		return provider.LanguageModel(t.Context(), model)
	}
}
