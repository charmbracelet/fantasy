package openai

type ReasoningEffort string

const (
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	ReasoningEffortLow     ReasoningEffort = "low"
	ReasoningEffortMedium  ReasoningEffort = "medium"
	ReasoningEffortHigh    ReasoningEffort = "high"
)

type ProviderOptions struct {
	LogitBias           map[string]int64 `mapstructure:"logit_bias"`
	LogProbs            *bool            `mapstructure:"log_probes"`
	TopLogProbs         *int64           `mapstructure:"top_log_probs"`
	ParallelToolCalls   *bool            `mapstructure:"parallel_tool_calls"`
	User                *string          `mapstructure:"user"`
	ReasoningEffort     *ReasoningEffort `mapstructure:"reasoning_effort"`
	MaxCompletionTokens *int64           `mapstructure:"max_completion_tokens"`
	TextVerbosity       *string          `mapstructure:"text_verbosity"`
	Prediction          map[string]any   `mapstructure:"prediction"`
	Store               *bool            `mapstructure:"store"`
	Metadata            map[string]any   `mapstructure:"metadata"`
	PromptCacheKey      *string          `mapstructure:"prompt_cache_key"`
	SafetyIdentifier    *string          `mapstructure:"safety_identifier"`
	ServiceTier         *string          `mapstructure:"service_tier"`
	StructuredOutputs   *bool            `mapstructure:"structured_outputs"`
}
