package anthropic

type providerOptions struct {
	SendReasoning          *bool                   `json:"send_reasoning,omitempty"`
	Thinking               *thinkingProviderOption `json:"thinking,omitempty"`
	DisableParallelToolUse *bool                   `json:"disable_parallel_tool_use,omitempty"`
}

type thinkingProviderOption struct {
	BudgetTokens int64 `json:"budget_tokens"`
}

type reasoningMetadata struct {
	Signature    string `json:"signature"`
	RedactedData string `json:"redacted_data"`
}

type cacheControlProviderOptions struct {
	Type string `json:"type"`
}
