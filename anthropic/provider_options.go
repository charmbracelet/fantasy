package anthropic

import "github.com/charmbracelet/fantasy/ai"

const OptionsKey = "anthropic"

type ProviderOptions struct {
	SendReasoning          *bool
	Thinking               *ThinkingProviderOption
	DisableParallelToolUse *bool
}

func (o *ProviderOptions) Options() {}

type ThinkingProviderOption struct {
	BudgetTokens int64
}

type ReasoningOptionMetadata struct {
	Signature    string
	RedactedData string
}

func (*ReasoningOptionMetadata) Options() {}

type ProviderCacheControlOptions struct {
	CacheControl CacheControl
}

func (*ProviderCacheControlOptions) Options() {}

type CacheControl struct {
	Type string
}

func NewProviderOptions(opts *ProviderOptions) ai.ProviderOptions {
	return ai.ProviderOptions{
		OptionsKey: opts,
	}
}

func NewProviderCacheControlOptions(opts *ProviderCacheControlOptions) ai.ProviderOptions {
	return ai.ProviderOptions{
		OptionsKey: opts,
	}
}
