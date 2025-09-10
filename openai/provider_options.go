package openai

import (
	"github.com/charmbracelet/ai/ai"
	"github.com/openai/openai-go/v2"
)

type ReasoningEffort string

const (
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	ReasoningEffortLow     ReasoningEffort = "low"
	ReasoningEffortMedium  ReasoningEffort = "medium"
	ReasoningEffortHigh    ReasoningEffort = "high"
)

type ProviderFileOptions struct {
	ImageDetail string
}

type ProviderMetadata struct {
	Logprobs                 []openai.ChatCompletionTokenLogprob
	AcceptedPredictionTokens int64
	RejectedPredictionTokens int64
}

type ProviderOptions struct {
	LogitBias           map[string]int64
	LogProbs            *bool
	TopLogProbs         *int64
	ParallelToolCalls   *bool
	User                *string
	ReasoningEffort     *ReasoningEffort
	MaxCompletionTokens *int64
	TextVerbosity       *string
	Prediction          map[string]any
	Store               *bool
	Metadata            map[string]any
	PromptCacheKey      *string
	SafetyIdentifier    *string
	ServiceTier         *string
	StructuredOutputs   *bool
}

func ReasoningEffortOption(e ReasoningEffort) *ReasoningEffort {
	return &e
}

func NewProviderOptions(opts *ProviderOptions) ai.ProviderOptions {
	return ai.ProviderOptions{
		"openai": opts,
	}
}

func NewProviderFileOptions(opts *ProviderFileOptions) ai.ProviderOptions {
	return ai.ProviderOptions{
		"openai": opts,
	}
}
