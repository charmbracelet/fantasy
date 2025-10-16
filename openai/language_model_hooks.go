package openai

import (
	"fmt"

	"charm.land/fantasy/ai"
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/packages/param"
	"github.com/openai/openai-go/v2/shared"
)

type (
	LanguageModelPrepareCallFunc            = func(model ai.LanguageModel, params *openai.ChatCompletionNewParams, call ai.Call) ([]ai.CallWarning, error)
	LanguageModelMapFinishReasonFunc        = func(finishReason string) ai.FinishReason
	LanguageModelUsageFunc                  = func(choice openai.ChatCompletion) (ai.Usage, ai.ProviderOptionsData)
	LanguageModelExtraContentFunc           = func(choice openai.ChatCompletionChoice) []ai.Content
	LanguageModelStreamExtraFunc            = func(chunk openai.ChatCompletionChunk, yield func(ai.StreamPart) bool, ctx map[string]any) (map[string]any, bool)
	LanguageModelStreamUsageFunc            = func(chunk openai.ChatCompletionChunk, ctx map[string]any, metadata ai.ProviderMetadata) (ai.Usage, ai.ProviderMetadata)
	LanguageModelStreamProviderMetadataFunc = func(choice openai.ChatCompletionChoice, metadata ai.ProviderMetadata) ai.ProviderMetadata
)

func DefaultPrepareCallFunc(model ai.LanguageModel, params *openai.ChatCompletionNewParams, call ai.Call) ([]ai.CallWarning, error) {
	if call.ProviderOptions == nil {
		return nil, nil
	}
	var warnings []ai.CallWarning
	providerOptions := &ProviderOptions{}
	if v, ok := call.ProviderOptions[Name]; ok {
		providerOptions, ok = v.(*ProviderOptions)
		if !ok {
			return nil, ai.NewInvalidArgumentError("providerOptions", "openai provider options should be *openai.ProviderOptions", nil)
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
			warnings = append(warnings, ai.CallWarning{
				Type:    ai.CallWarningTypeUnsupportedSetting,
				Setting: "LogitBias",
				Message: "LogitBias is not supported for reasoning models",
			})
		}
		if providerOptions.LogProbs != nil {
			params.Logprobs = param.Opt[bool]{}
			warnings = append(warnings, ai.CallWarning{
				Type:    ai.CallWarningTypeUnsupportedSetting,
				Setting: "Logprobs",
				Message: "Logprobs is not supported for reasoning models",
			})
		}
		if providerOptions.TopLogProbs != nil {
			params.TopLogprobs = param.Opt[int64]{}
			warnings = append(warnings, ai.CallWarning{
				Type:    ai.CallWarningTypeUnsupportedSetting,
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
			warnings = append(warnings, ai.CallWarning{
				Type:    ai.CallWarningTypeUnsupportedSetting,
				Setting: "ServiceTier",
				Details: "flex processing is only available for o3, o4-mini, and gpt-5 models",
			})
		} else if serviceTier == "priority" && !supportsPriorityProcessing(model.Model()) {
			params.ServiceTier = ""
			warnings = append(warnings, ai.CallWarning{
				Type:    ai.CallWarningTypeUnsupportedSetting,
				Setting: "ServiceTier",
				Details: "priority processing is only available for supported models (gpt-4, gpt-5, gpt-5-mini, o3, o4-mini) and requires Enterprise access. gpt-5-nano is not supported",
			})
		}
	}
	return warnings, nil
}

func DefaultMapFinishReasonFunc(finishReason string) ai.FinishReason {
	switch finishReason {
	case "stop":
		return ai.FinishReasonStop
	case "length":
		return ai.FinishReasonLength
	case "content_filter":
		return ai.FinishReasonContentFilter
	case "function_call", "tool_calls":
		return ai.FinishReasonToolCalls
	default:
		return ai.FinishReasonUnknown
	}
}

func DefaultUsageFunc(response openai.ChatCompletion) (ai.Usage, ai.ProviderOptionsData) {
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
	return ai.Usage{
		InputTokens:     response.Usage.PromptTokens,
		OutputTokens:    response.Usage.CompletionTokens,
		TotalTokens:     response.Usage.TotalTokens,
		ReasoningTokens: completionTokenDetails.ReasoningTokens,
		CacheReadTokens: promptTokenDetails.CachedTokens,
	}, providerMetadata
}

func DefaultStreamUsageFunc(chunk openai.ChatCompletionChunk, ctx map[string]any, metadata ai.ProviderMetadata) (ai.Usage, ai.ProviderMetadata) {
	if chunk.Usage.TotalTokens == 0 {
		return ai.Usage{}, nil
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
	usage := ai.Usage{
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

	return usage, ai.ProviderMetadata{
		Name: streamProviderMetadata,
	}
}

func DefaultStreamProviderMetadataFunc(choice openai.ChatCompletionChoice, metadata ai.ProviderMetadata) ai.ProviderMetadata {
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
