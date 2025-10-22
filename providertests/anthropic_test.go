package providertests

import (
	"context"
	"net/http"
	"os"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/anthropic"
	"github.com/stretchr/testify/require"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

var anthropicTestModels = []testModel{
	{"claude-sonnet-4", "claude-sonnet-4-20250514", true},
}

func TestAnthropicCommon(t *testing.T) {
	var pairs []builderPair
	for _, m := range anthropicTestModels {
		pairs = append(pairs, builderPair{m.name, anthropicBuilder(m.model), nil, nil})
	}
	testCommon(t, pairs)
}

func addAnthropicCaching(ctx context.Context, options fantasy.PrepareStepFunctionOptions) (context.Context, fantasy.PrepareStepResult, error) {
	prepared := fantasy.PrepareStepResult{}
	prepared.Messages = options.Messages

	for i := range prepared.Messages {
		prepared.Messages[i].ProviderOptions = nil
	}
	providerOption := fantasy.ProviderOptions{
		anthropic.Name: &anthropic.ProviderCacheControlOptions{
			CacheControl: anthropic.CacheControl{Type: "ephemeral"},
		},
	}

	lastSystemRoleInx := 0
	systemMessageUpdated := false
	for i, msg := range prepared.Messages {
		// only add cache control to the last message
		if msg.Role == fantasy.MessageRoleSystem {
			lastSystemRoleInx = i
		} else if !systemMessageUpdated {
			prepared.Messages[lastSystemRoleInx].ProviderOptions = providerOption
			systemMessageUpdated = true
		}
		// than add cache control to the last 2 messages
		if i > len(prepared.Messages)-3 {
			prepared.Messages[i].ProviderOptions = providerOption
		}
	}
	return ctx, prepared, nil
}

func TestAnthropicCommonWithCacheControl(t *testing.T) {
	var pairs []builderPair
	for _, m := range anthropicTestModels {
		pairs = append(pairs, builderPair{m.name, anthropicBuilder(m.model), nil, addAnthropicCaching})
	}
	testCommon(t, pairs)
}

func TestAnthropicThinking(t *testing.T) {
	opts := fantasy.ProviderOptions{
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
		pairs = append(pairs, builderPair{m.name, anthropicBuilder(m.model), opts, nil})
	}
	testThinking(t, pairs, testAnthropicThinking)
}

func TestAnthropicThinkingWithCacheControl(t *testing.T) {
	opts := fantasy.ProviderOptions{
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
		pairs = append(pairs, builderPair{m.name, anthropicBuilder(m.model), opts, addAnthropicCaching})
	}
	testThinking(t, pairs, testAnthropicThinking)
}

func testAnthropicThinking(t *testing.T, result *fantasy.AgentResult) {
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
}

func anthropicBuilder(model string) builderFunc {
	return func(r *recorder.Recorder) (fantasy.LanguageModel, error) {
		provider, err := anthropic.New(
			anthropic.WithAPIKey(os.Getenv("FANTASY_ANTHROPIC_API_KEY")),
			anthropic.WithHTTPClient(&http.Client{Transport: r}),
		)
		if err != nil {
			return nil, err
		}
		return provider.LanguageModel(model)
	}
}
