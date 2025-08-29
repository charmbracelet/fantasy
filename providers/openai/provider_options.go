package openai

type reasoningEffort string

const (
	reasoningEffortMinimal reasoningEffort = "minimal"
	reasoningEffortLow     reasoningEffort = "low"
	reasoningEffortMedium  reasoningEffort = "medium"
	reasoningEffortHigh    reasoningEffort = "high"
)

type providerOptions struct {
	LogitBias           map[string]int64 `json:"logit_bias"`
	LogProbs            *bool            `json:"log_probes"`
	TopLogProbs         *int64           `json:"top_log_probs"`
	ParallelToolCalls   *bool            `json:"parallel_tool_calls"`
	User                *string          `json:"user"`
	ReasoningEffort     *reasoningEffort `json:"reasoning_effort"`
	MaxCompletionTokens *int64           `json:"max_completion_tokens"`
	TextVerbosity       *string          `json:"text_verbosity"`
	Prediction          map[string]any   `json:"prediction"`
	Store               *bool            `json:"store"`
	Metadata            map[string]any   `json:"metadata"`
	PromptCacheKey      *string          `json:"prompt_cache_key"`
	SafetyIdentifier    *string          `json:"safety_identifier"`
	ServiceTier         *string          `json:"service_tier"`
	StructuredOutputs   *bool            `json:"structured_outputs"`
}
