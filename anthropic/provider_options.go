package anthropic

import "charm.land/fantasy/ai"

type ProviderOptions struct {
	SendReasoning          *bool                   `json:"send_reasoning"`
	Thinking               *ThinkingProviderOption `json:"thinking"`
	DisableParallelToolUse *bool                   `json:"disable_parallel_tool_use"`
}

func (o *ProviderOptions) Options() {}

type ThinkingProviderOption struct {
	BudgetTokens int64 `json:"budget_tokens"`
}

type ReasoningOptionMetadata struct {
	Signature    string `json:"signature"`
	RedactedData string `json:"redacted_data"`
}

func (*ReasoningOptionMetadata) Options() {}

type ProviderCacheControlOptions struct {
	CacheControl CacheControl `json:"cache_control"`
}

func (*ProviderCacheControlOptions) Options() {}

type CacheControl struct {
	Type string `json:"type"`
}

func NewProviderOptions(opts *ProviderOptions) ai.ProviderOptions {
	return ai.ProviderOptions{
		Name: opts,
	}
}

func NewProviderCacheControlOptions(opts *ProviderCacheControlOptions) ai.ProviderOptions {
	return ai.ProviderOptions{
		Name: opts,
	}
}

func ParseOptions(data map[string]any) (*ProviderOptions, error) {
	var options ProviderOptions
	if err := ai.ParseOptions(data, &options); err != nil {
		return nil, err
	}
	return &options, nil
}
