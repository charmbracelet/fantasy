package bedrock

import (
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/anthropic"
	"github.com/stretchr/testify/require"
)

func TestExtractProviderOptions_BedrockOptions(t *testing.T) {
	t.Parallel()

	call := fantasy.Call{
		ProviderOptions: fantasy.ProviderOptions{
			Name: &ProviderOptions{
				Thinking: &ThinkingProviderOption{
					ReasoningEffort: ReasoningEffortHigh,
				},
			},
		},
	}

	opts := extractProviderOptions(call)
	require.NotNil(t, opts.Thinking)
	require.Equal(t, ReasoningEffortHigh, opts.Thinking.ReasoningEffort)
}

func TestExtractProviderOptions_AnthropicFallback(t *testing.T) {
	t.Parallel()

	call := fantasy.Call{
		ProviderOptions: fantasy.ProviderOptions{
			anthropic.Name: &anthropic.ProviderOptions{
				Thinking: &anthropic.ThinkingProviderOption{
					BudgetTokens: 2000,
				},
			},
		},
	}

	opts := extractProviderOptions(call)
	require.True(t, thinkingEnabled(opts))
	require.Equal(t, ReasoningEffortLow, resolveReasoningEffort(opts))
}
