// Package anthropic provides an implementation of the fantasy AI SDK for Anthropic's language models.
package anthropic

import "charm.land/fantasy"

// ProviderOptions represents additional options for the Anthropic provider.
type ProviderOptions struct {
	SendReasoning          *bool                   `json:"send_reasoning"`
	Thinking               *ThinkingProviderOption `json:"thinking"`
	DisableParallelToolUse *bool                   `json:"disable_parallel_tool_use"`
}

// Options implements the ProviderOptions interface.
func (o *ProviderOptions) Options() {}

// ThinkingProviderOption represents thinking options for the Anthropic provider.
type ThinkingProviderOption struct {
	BudgetTokens int64 `json:"budget_tokens"`
}

// ReasoningOptionMetadata represents reasoning metadata for the Anthropic provider.
type ReasoningOptionMetadata struct {
	Signature    string `json:"signature"`
	RedactedData string `json:"redacted_data"`
}

// Options implements the ProviderOptions interface.
func (*ReasoningOptionMetadata) Options() {}

// ProviderCacheControlOptions represents cache control options for the Anthropic provider.
type ProviderCacheControlOptions struct {
	CacheControl CacheControl `json:"cache_control"`
}

// Options implements the ProviderOptions interface.
func (*ProviderCacheControlOptions) Options() {}

// CacheControl represents cache control settings for the Anthropic provider.
type CacheControl struct {
	Type string `json:"type"`
}

// NewProviderOptions creates new provider options for the Anthropic provider.
func NewProviderOptions(opts *ProviderOptions) fantasy.ProviderOptions {
	return fantasy.ProviderOptions{
		Name: opts,
	}
}

// NewProviderCacheControlOptions creates new cache control options for the Anthropic provider.
func NewProviderCacheControlOptions(opts *ProviderCacheControlOptions) fantasy.ProviderOptions {
	return fantasy.ProviderOptions{
		Name: opts,
	}
}

// ParseOptions parses provider options from a map for the Anthropic provider.
func ParseOptions(data map[string]any) (*ProviderOptions, error) {
	var options ProviderOptions
	if err := fantasy.ParseOptions(data, &options); err != nil {
		return nil, err
	}
	return &options, nil
}
