package anthropic

type ProviderOptions struct {
	SendReasoning          *bool                   `mapstructure:"send_reasoning,omitempty"`
	Thinking               *ThinkingProviderOption `mapstructure:"thinking,omitempty"`
	DisableParallelToolUse *bool                   `mapstructure:"disable_parallel_tool_use,omitempty"`
}

type ThinkingProviderOption struct {
	BudgetTokens int64 `mapstructure:"budget_tokens"`
}

type ReasoningMetadata struct {
	Signature    string `mapstructure:"signature"`
	RedactedData string `mapstructure:"redacted_data"`
}

type CacheControlProviderOptions struct {
	Type string `mapstructure:"type"`
}
