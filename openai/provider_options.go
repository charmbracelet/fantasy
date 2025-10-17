package openai

import (
	"charm.land/fantasy/ai"
	"github.com/openai/openai-go/v2"
)

type ReasoningEffort string

const (
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	ReasoningEffortLow     ReasoningEffort = "low"
	ReasoningEffortMedium  ReasoningEffort = "medium"
	ReasoningEffortHigh    ReasoningEffort = "high"
)

type ProviderMetadata struct {
	Logprobs                 []openai.ChatCompletionTokenLogprob `json:"logprobs"`
	AcceptedPredictionTokens int64                               `json:"accepted_prediction_tokens"`
	RejectedPredictionTokens int64                               `json:"rejected_prediction_tokens"`
}

func (*ProviderMetadata) Options() {}

type ProviderOptions struct {
	LogitBias           map[string]int64 `json:"logit_bias"`
	LogProbs            *bool            `json:"log_probs"`
	TopLogProbs         *int64           `json:"top_log_probs"`
	ParallelToolCalls   *bool            `json:"parallel_tool_calls"`
	User                *string          `json:"user"`
	ReasoningEffort     *ReasoningEffort `json:"reasoning_effort"`
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

func (*ProviderOptions) Options() {}

type ProviderFileOptions struct {
	ImageDetail string `json:"image_detail"`
}

func (*ProviderFileOptions) Options() {}

func ReasoningEffortOption(e ReasoningEffort) *ReasoningEffort {
	return &e
}

func NewProviderOptions(opts *ProviderOptions) ai.ProviderOptions {
	return ai.ProviderOptions{
		Name: opts,
	}
}

func NewProviderFileOptions(opts *ProviderFileOptions) ai.ProviderOptions {
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
