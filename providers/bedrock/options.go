package bedrock

import (
	"charm.land/fantasy"
	"charm.land/fantasy/providers/anthropic"
)

// extractProviderOptions reads Bedrock provider options from a call.
// Crush routes all Bedrock models through anthropic.ParseOptions today, so we
// fall back to Anthropic options when Bedrock-specific options are absent.
func extractProviderOptions(call fantasy.Call) *ProviderOptions {
	if v, ok := call.ProviderOptions[Name]; ok {
		if opts, ok := v.(*ProviderOptions); ok && opts != nil {
			return opts
		}
	}

	if v, ok := call.ProviderOptions[anthropic.Name]; ok {
		if opts, ok := v.(*anthropic.ProviderOptions); ok {
			return convertAnthropicProviderOptions(opts)
		}
	}

	return &ProviderOptions{}
}

func convertAnthropicProviderOptions(opts *anthropic.ProviderOptions) *ProviderOptions {
	if opts == nil || opts.Thinking == nil {
		return &ProviderOptions{}
	}

	thinking := &ThinkingProviderOption{
		BudgetTokens: opts.Thinking.BudgetTokens,
	}

	if opts.Thinking.BudgetTokens > 0 {
		switch {
		case opts.Thinking.BudgetTokens < 5000:
			thinking.ReasoningEffort = ReasoningEffortLow
		case opts.Thinking.BudgetTokens < 15000:
			thinking.ReasoningEffort = ReasoningEffortMedium
		default:
			thinking.ReasoningEffort = ReasoningEffortHigh
		}
	}

	return &ProviderOptions{Thinking: thinking}
}

func thinkingEnabled(opts *ProviderOptions) bool {
	return opts != nil && opts.Thinking != nil &&
		(opts.Thinking.ReasoningEffort != "" || opts.Thinking.BudgetTokens > 0)
}

func resolveReasoningEffort(opts *ProviderOptions) ReasoningEffort {
	if opts == nil || opts.Thinking == nil {
		return ReasoningEffortMedium
	}

	effort := opts.Thinking.ReasoningEffort
	if effort == "" && opts.Thinking.BudgetTokens > 0 {
		switch {
		case opts.Thinking.BudgetTokens < 5000:
			effort = ReasoningEffortLow
		case opts.Thinking.BudgetTokens < 15000:
			effort = ReasoningEffortMedium
		default:
			effort = ReasoningEffortHigh
		}
	}
	if effort == "" {
		effort = ReasoningEffortMedium
	}
	return effort
}
