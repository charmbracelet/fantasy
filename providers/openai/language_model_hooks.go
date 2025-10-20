package openai

import (
	"fmt"

	"charm.land/fantasy"
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/packages/param"
	"github.com/openai/openai-go/v2/shared"
)

// LanguageModelPrepareCallFunc is a function that prepares the call for the language model.
type LanguageModelPrepareCallFunc = func(model fantasy.LanguageModel, params *openai.ChatCompletionNewParams, call fantasy.Call) ([]fantasy.CallWarning, error)

// LanguageModelMapFinishReasonFunc is a function that maps the finish reason for the language model.
type LanguageModelMapFinishReasonFunc = func(finishReason string) fantasy.FinishReason

// LanguageModelUsageFunc is a function that calculates usage for the language model.
type LanguageModelUsageFunc = func(choice openai.ChatCompletion) (fantasy.Usage, fantasy.ProviderOptionsData)

// LanguageModelExtraContentFunc is a function that adds extra content for the language model.
type LanguageModelExtraContentFunc = func(choice openai.ChatCompletionChoice) []fantasy.Content

// LanguageModelStreamExtraFunc is a function that handles stream extra functionality for the language model.
type LanguageModelStreamExtraFunc = func(chunk openai.ChatCompletionChunk, yield func(fantasy.StreamPart) bool, ctx map[string]any) (map[string]any, bool)

// LanguageModelStreamUsageFunc is a function that calculates stream usage for the language model.
type LanguageModelStreamUsageFunc = func(chunk openai.ChatCompletionChunk, ctx map[string]any, metadata fantasy.ProviderMetadata) (fantasy.Usage, fantasy.ProviderMetadata)

// LanguageModelStreamProviderMetadataFunc is a function that handles stream provider metadata for the language model.
type LanguageModelStreamProviderMetadataFunc = func(choice openai.ChatCompletionChoice, metadata fantasy.ProviderMetadata) fantasy.ProviderMetadata

// DefaultPrepareCallFunc is the default implementation for preparing a call to the language model.
func DefaultPrepareCallFunc(model fantasy.LanguageModel, params *openai.ChatCompletionNewParams, call fantasy.Call) ([]fantasy.CallWarning, error) {
	if call.ProviderOptions == nil {
		return nil, nil
	}
	var warnings []fantasy.CallWarning
	providerOptions := &ProviderOptions{}
	if v, ok := call.ProviderOptions[Name]; ok {
		providerOptions, ok = v.(*ProviderOptions)
		if !ok {
			return nil, fantasy.NewInvalidArgumentError("providerOptions", "openai provider options should be *openai.ProviderOptions", nil)
		}
	}

	if providerOptions.LogitBias != nil {
		params.LogitBias = providerOptions.LogitBias
	}
	if providerOptions.LogProbs != nil && providerOptions.TopLogProbs != nil {
		providerOptions.LogProbs = nil
	}
	if providerOptions.LogProbs != nil {
		params.Logprobs = param.NewOpt(*providerOptions.LogProbs)
	}
	if providerOptions.TopLogProbs != nil {
		params.TopLogprobs = param.NewOpt(*providerOptions.TopLogProbs)
	}
	if providerOptions.User != nil {
		params.User = param.NewOpt(*providerOptions.User)
	}
	if providerOptions.ParallelToolCalls != nil {
		params.ParallelToolCalls = param.NewOpt(*providerOptions.ParallelToolCalls)
	}
	if providerOptions.MaxCompletionTokens != nil {
		params.MaxCompletionTokens = param.NewOpt(*providerOptions.MaxCompletionTokens)
	}

	if providerOptions.TextVerbosity != nil {
		params.Verbosity = openai.ChatCompletionNewParamsVerbosity(*providerOptions.TextVerbosity)
	}
	if providerOptions.Prediction != nil {
		// Convert map[string]any to ChatCompletionPredictionContentParam
		if content, ok := providerOptions.Prediction["content"]; ok {
			if contentStr, ok := content.(string); ok {
				params.Prediction = openai.ChatCompletionPredictionContentParam{
					Content: openai.ChatCompletionPredictionContentContentUnionParam{
						OfString: param.NewOpt(contentStr),
					},
				}
			}
		}
	}
	if providerOptions.Store != nil {
		params.Store = param.NewOpt(*providerOptions.Store)
	}
	if providerOptions.Metadata != nil {
		// Convert map[string]any to map[string]string
		metadata := make(map[string]string)
		for k, v := range providerOptions.Metadata {
			if str, ok := v.(string); ok {
				metadata[k] = str
			}
		}
		params.Metadata = metadata
	}
	if providerOptions.PromptCacheKey != nil {
		params.PromptCacheKey = param.NewOpt(*providerOptions.PromptCacheKey)
	}
	if providerOptions.SafetyIdentifier != nil {
		params.SafetyIdentifier = param.NewOpt(*providerOptions.SafetyIdentifier)
	}
	if providerOptions.ServiceTier != nil {
		params.ServiceTier = openai.ChatCompletionNewParamsServiceTier(*providerOptions.ServiceTier)
	}

	if providerOptions.ReasoningEffort != nil {
		switch *providerOptions.ReasoningEffort {
		case ReasoningEffortMinimal:
			params.ReasoningEffort = shared.ReasoningEffortMinimal
		case ReasoningEffortLow:
			params.ReasoningEffort = shared.ReasoningEffortLow
		case ReasoningEffortMedium:
			params.ReasoningEffort = shared.ReasoningEffortMedium
		case ReasoningEffortHigh:
			params.ReasoningEffort = shared.ReasoningEffortHigh
		default:
			return nil, fmt.Errorf("reasoning model `%s` not supported", *providerOptions.ReasoningEffort)
		}
	}

	if isReasoningModel(model.Model()) {
		if providerOptions.LogitBias != nil {
			params.LogitBias = nil
			warnings = append(warnings, fantasy.CallWarning{
				Type:    fantasy.CallWarningTypeUnsupportedSetting,
				Setting: "LogitBias",
				Message: "LogitBias is not supported for reasoning models",
			})
		}
		if providerOptions.LogProbs != nil {
			params.Logprobs = param.Opt[bool]{}
			warnings = append(warnings, fantasy.CallWarning{
				Type:    fantasy.CallWarningTypeUnsupportedSetting,
				Setting: "Logprobs",
				Message: "Logprobs is not supported for reasoning models",
			})
		}
		if providerOptions.TopLogProbs != nil {
			params.TopLogprobs = param.Opt[int64]{}
			warnings = append(warnings, fantasy.CallWarning{
				Type:    fantasy.CallWarningTypeUnsupportedSetting,
				Setting: "TopLogprobs",
				Message: "TopLogprobs is not supported for reasoning models",
			})
		}
	}

	// Handle service tier validation
	if providerOptions.ServiceTier != nil {
		serviceTier := *providerOptions.ServiceTier
		if serviceTier == "flex" && !supportsFlexProcessing(model.Model()) {
			params.ServiceTier = ""
			warnings = append(warnings, fantasy.CallWarning{
				Type:    fantasy.CallWarningTypeUnsupportedSetting,
				Setting: "ServiceTier",
				Details: "flex processing is only available for o3, o4-mini, and gpt-5 models",
			})
		} else if serviceTier == "priority" && !supportsPriorityProcessing(model.Model()) {
			params.ServiceTier = ""
			warnings = append(warnings, fantasy.CallWarning{
				Type:    fantasy.CallWarningTypeUnsupportedSetting,
				Setting: "ServiceTier",
				Details: "priority processing is only available for supported models (gpt-4, gpt-5, gpt-5-mini, o3, o4-mini) and requires Enterprise access. gpt-5-nano is not supported",
			})
		}
	}
	return warnings, nil
}

// DefaultMapFinishReasonFunc is the default implementation for mapping finish reasons.
func DefaultMapFinishReasonFunc(finishReason string) fantasy.FinishReason {
	switch finishReason {
	case "stop":
		return fantasy.FinishReasonStop
	case "length":
		return fantasy.FinishReasonLength
	case "content_filter":
		return fantasy.FinishReasonContentFilter
	case "function_call", "tool_calls":
		return fantasy.FinishReasonToolCalls
	default:
		return fantasy.FinishReasonUnknown
	}
}

// DefaultUsageFunc is the default implementation for calculating usage.
func DefaultUsageFunc(response openai.ChatCompletion) (fantasy.Usage, fantasy.ProviderOptionsData) {
	completionTokenDetails := response.Usage.CompletionTokensDetails
	promptTokenDetails := response.Usage.PromptTokensDetails

	// Build provider metadata
	providerMetadata := &ProviderMetadata{}

	// Add logprobs if available
	if len(response.Choices) > 0 && len(response.Choices[0].Logprobs.Content) > 0 {
		providerMetadata.Logprobs = response.Choices[0].Logprobs.Content
	}

	// Add prediction tokens if available
	if completionTokenDetails.AcceptedPredictionTokens > 0 || completionTokenDetails.RejectedPredictionTokens > 0 {
		if completionTokenDetails.AcceptedPredictionTokens > 0 {
			providerMetadata.AcceptedPredictionTokens = completionTokenDetails.AcceptedPredictionTokens
		}
		if completionTokenDetails.RejectedPredictionTokens > 0 {
			providerMetadata.RejectedPredictionTokens = completionTokenDetails.RejectedPredictionTokens
		}
	}
	return fantasy.Usage{
		InputTokens:     response.Usage.PromptTokens,
		OutputTokens:    response.Usage.CompletionTokens,
		TotalTokens:     response.Usage.TotalTokens,
		ReasoningTokens: completionTokenDetails.ReasoningTokens,
		CacheReadTokens: promptTokenDetails.CachedTokens,
	}, providerMetadata
}

// DefaultStreamUsageFunc is the default implementation for calculating stream usage.
func DefaultStreamUsageFunc(chunk openai.ChatCompletionChunk, _ map[string]any, metadata fantasy.ProviderMetadata) (fantasy.Usage, fantasy.ProviderMetadata) {
	if chunk.Usage.TotalTokens == 0 {
		return fantasy.Usage{}, nil
	}
	streamProviderMetadata := &ProviderMetadata{}
	if metadata != nil {
		if providerMetadata, ok := metadata[Name]; ok {
			converted, ok := providerMetadata.(*ProviderMetadata)
			if ok {
				streamProviderMetadata = converted
			}
		}
	}
	// we do this here because the acc does not add prompt details
	completionTokenDetails := chunk.Usage.CompletionTokensDetails
	promptTokenDetails := chunk.Usage.PromptTokensDetails
	usage := fantasy.Usage{
		InputTokens:     chunk.Usage.PromptTokens,
		OutputTokens:    chunk.Usage.CompletionTokens,
		TotalTokens:     chunk.Usage.TotalTokens,
		ReasoningTokens: completionTokenDetails.ReasoningTokens,
		CacheReadTokens: promptTokenDetails.CachedTokens,
	}

	// Add prediction tokens if available
	if completionTokenDetails.AcceptedPredictionTokens > 0 || completionTokenDetails.RejectedPredictionTokens > 0 {
		if completionTokenDetails.AcceptedPredictionTokens > 0 {
			streamProviderMetadata.AcceptedPredictionTokens = completionTokenDetails.AcceptedPredictionTokens
		}
		if completionTokenDetails.RejectedPredictionTokens > 0 {
			streamProviderMetadata.RejectedPredictionTokens = completionTokenDetails.RejectedPredictionTokens
		}
	}

	return usage, fantasy.ProviderMetadata{
		Name: streamProviderMetadata,
	}
}

// DefaultStreamProviderMetadataFunc is the default implementation for handling stream provider metadata.
func DefaultStreamProviderMetadataFunc(choice openai.ChatCompletionChoice, metadata fantasy.ProviderMetadata) fantasy.ProviderMetadata {
	streamProviderMetadata, ok := metadata[Name]
	if !ok {
		streamProviderMetadata = &ProviderMetadata{}
	}
	if converted, ok := streamProviderMetadata.(*ProviderMetadata); ok {
		converted.Logprobs = choice.Logprobs.Content
		metadata[Name] = converted
	}
	return metadata
}
