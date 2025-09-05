package anthropic

type ProviderOptions struct {
	SendReasoning          *bool                   `json:"send_reasoning,omitempty"`
	Thinking               *ThinkingProviderOption `json:"thinking,omitempty"`
	DisableParallelToolUse *bool                   `json:"disable_parallel_tool_use,omitempty"`
}

type ThinkingProviderOption struct {
	BudgetTokens int64 `json:"budget_tokens"`
}

type ReasoningMetadata struct {
	Signature    string `json:"signature"`
	RedactedData string `json:"redacted_data"`
}

type CacheControlProviderOptions struct {
	Type string `json:"type"`
}
