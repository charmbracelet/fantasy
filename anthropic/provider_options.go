package anthropic

import "github.com/charmbracelet/ai/ai"

type ProviderOptions struct {
	SendReasoning          *bool
	Thinking               *ThinkingProviderOption
	DisableParallelToolUse *bool
}

type ThinkingProviderOption struct {
	BudgetTokens int64
}

type ReasoningMetadata struct {
	Signature    string
	RedactedData string
}

type ProviderCacheControlOptions struct {
	CacheControl CacheControl
}

type CacheControl struct {
	Type string
}

func NewProviderOptions(opts *ProviderOptions) ai.ProviderOptions {
	return ai.ProviderOptions{
		"anthropic": opts,
	}
}

func NewProviderCacheControlOptions(opts *ProviderCacheControlOptions) ai.ProviderOptions {
	return ai.ProviderOptions{
		"anthropic": opts,
	}
}
