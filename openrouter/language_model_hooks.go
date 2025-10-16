package openrouter

import (
	"encoding/json"
	"fmt"
	"maps"

	"charm.land/fantasy/ai"
	"charm.land/fantasy/anthropic"
	openaisdk "github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/packages/param"
)

const reasoningStartedCtx = "reasoning_started"

func languagePrepareModelCall(model ai.LanguageModel, params *openaisdk.ChatCompletionNewParams, call ai.Call) ([]ai.CallWarning, error) {
	providerOptions := &ProviderOptions{}
	if v, ok := call.ProviderOptions[Name]; ok {
		providerOptions, ok = v.(*ProviderOptions)
		if !ok {
			return nil, ai.NewInvalidArgumentError("providerOptions", "openrouter provider options should be *openrouter.ProviderOptions", nil)
		}
	}

	extraFields := make(map[string]any)

	if providerOptions.Provider != nil {
		data, err := structToMapJSON(providerOptions.Provider)
		if err != nil {
			return nil, err
		}
		extraFields["provider"] = data
	}

	if providerOptions.Reasoning != nil {
		data, err := structToMapJSON(providerOptions.Reasoning)
		if err != nil {
			return nil, err
		}
		extraFields["reasoning"] = data
	}

	if providerOptions.IncludeUsage != nil {
		extraFields["usage"] = map[string]any{
			"include": *providerOptions.IncludeUsage,
		}
	} else { // default include usage
		extraFields["usage"] = map[string]any{
			"include": true,
		}
	}
	if providerOptions.LogitBias != nil {
		params.LogitBias = providerOptions.LogitBias
	}
	if providerOptions.LogProbs != nil {
		params.Logprobs = param.NewOpt(*providerOptions.LogProbs)
	}
	if providerOptions.User != nil {
		params.User = param.NewOpt(*providerOptions.User)
	}
	if providerOptions.ParallelToolCalls != nil {
		params.ParallelToolCalls = param.NewOpt(*providerOptions.ParallelToolCalls)
	}

	maps.Copy(extraFields, providerOptions.ExtraBody)
	params.SetExtraFields(extraFields)
	return nil, nil
}

func languageModelExtraContent(choice openaisdk.ChatCompletionChoice) []ai.Content {
	var content []ai.Content
	reasoningData := ReasoningData{}
	err := json.Unmarshal([]byte(choice.Message.RawJSON()), &reasoningData)
	if err != nil {
		return content
	}
	for _, detail := range reasoningData.ReasoningDetails {
		var metadata ai.ProviderMetadata

		if detail.Signature != "" {
			metadata = ai.ProviderMetadata{
				Name: &ReasoningMetadata{
					Signature: detail.Signature,
				},
				anthropic.Name: &anthropic.ReasoningOptionMetadata{
					Signature: detail.Signature,
				},
			}
		}
		switch detail.Type {
		case "reasoning.text":
			content = append(content, ai.ReasoningContent{
				Text:             detail.Text,
				ProviderMetadata: metadata,
			})
		case "reasoning.summary":
			content = append(content, ai.ReasoningContent{
				Text:             detail.Summary,
				ProviderMetadata: metadata,
			})
		case "reasoning.encrypted":
			content = append(content, ai.ReasoningContent{
				Text:             "[REDACTED]",
				ProviderMetadata: metadata,
			})
		}
	}
	return content
}

func extractReasoningContext(ctx map[string]any) bool {
	reasoningStarted, ok := ctx[reasoningStartedCtx]
	if !ok {
		return false
	}
	b, ok := reasoningStarted.(bool)
	if !ok {
		return false
	}
	return b
}

func languageModelStreamExtra(chunk openaisdk.ChatCompletionChunk, yield func(ai.StreamPart) bool, ctx map[string]any) (map[string]any, bool) {
	if len(chunk.Choices) == 0 {
		return ctx, true
	}

	reasoningStarted := extractReasoningContext(ctx)

	for inx, choice := range chunk.Choices {
		reasoningData := ReasoningData{}
		err := json.Unmarshal([]byte(choice.Delta.RawJSON()), &reasoningData)
		if err != nil {
			yield(ai.StreamPart{
				Type:  ai.StreamPartTypeError,
				Error: ai.NewAIError("Unexpected", "error unmarshalling delta", err),
			})
			return ctx, false
		}

		emitEvent := func(reasoningContent string, signature string) bool {
			if !reasoningStarted {
				shouldContinue := yield(ai.StreamPart{
					Type: ai.StreamPartTypeReasoningStart,
					ID:   fmt.Sprintf("%d", inx),
				})
				if !shouldContinue {
					return false
				}
			}

			var metadata ai.ProviderMetadata

			if signature != "" {
				metadata = ai.ProviderMetadata{
					Name: &ReasoningMetadata{
						Signature: signature,
					},
					anthropic.Name: &anthropic.ReasoningOptionMetadata{
						Signature: signature,
					},
				}
			}

			return yield(ai.StreamPart{
				Type:             ai.StreamPartTypeReasoningDelta,
				ID:               fmt.Sprintf("%d", inx),
				Delta:            reasoningContent,
				ProviderMetadata: metadata,
			})
		}
		if len(reasoningData.ReasoningDetails) > 0 {
			for _, detail := range reasoningData.ReasoningDetails {
				if !reasoningStarted {
					ctx[reasoningStartedCtx] = true
				}
				switch detail.Type {
				case "reasoning.text":
					return ctx, emitEvent(detail.Text, detail.Signature)
				case "reasoning.summary":
					return ctx, emitEvent(detail.Summary, detail.Signature)
				case "reasoning.encrypted":
					return ctx, emitEvent("[REDACTED]", detail.Signature)
				}
			}
		} else if reasoningData.Reasoning != "" {
			return ctx, emitEvent(reasoningData.Reasoning, "")
		}
		if reasoningStarted && (choice.Delta.Content != "" || len(choice.Delta.ToolCalls) > 0) {
			ctx[reasoningStartedCtx] = false
			return ctx, yield(ai.StreamPart{
				Type: ai.StreamPartTypeReasoningEnd,
				ID:   fmt.Sprintf("%d", inx),
			})
		}
	}
	return ctx, true
}

func languageModelUsage(response openaisdk.ChatCompletion) (ai.Usage, ai.ProviderOptionsData) {
	if len(response.Choices) == 0 {
		return ai.Usage{}, nil
	}
	openrouterUsage := UsageAccounting{}
	usage := response.Usage

	_ = json.Unmarshal([]byte(usage.RawJSON()), &openrouterUsage)

	completionTokenDetails := usage.CompletionTokensDetails
	promptTokenDetails := usage.PromptTokensDetails

	var provider string
	if p, ok := response.JSON.ExtraFields["provider"]; ok {
		provider = p.Raw()
	}

	// Build provider metadata
	providerMetadata := &ProviderMetadata{
		Provider: provider,
		Usage:    openrouterUsage,
	}

	return ai.Usage{
		InputTokens:     usage.PromptTokens,
		OutputTokens:    usage.CompletionTokens,
		TotalTokens:     usage.TotalTokens,
		ReasoningTokens: completionTokenDetails.ReasoningTokens,
		CacheReadTokens: promptTokenDetails.CachedTokens,
	}, providerMetadata
}

func languageModelStreamUsage(chunk openaisdk.ChatCompletionChunk, _ map[string]any, metadata ai.ProviderMetadata) (ai.Usage, ai.ProviderMetadata) {
	usage := chunk.Usage
	if usage.TotalTokens == 0 {
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
	openrouterUsage := UsageAccounting{}
	_ = json.Unmarshal([]byte(usage.RawJSON()), &openrouterUsage)
	streamProviderMetadata.Usage = openrouterUsage

	if p, ok := chunk.JSON.ExtraFields["provider"]; ok {
		streamProviderMetadata.Provider = p.Raw()
	}

	// we do this here because the acc does not add prompt details
	completionTokenDetails := usage.CompletionTokensDetails
	promptTokenDetails := usage.PromptTokensDetails
	aiUsage := ai.Usage{
		InputTokens:     usage.PromptTokens,
		OutputTokens:    usage.CompletionTokens,
		TotalTokens:     usage.TotalTokens,
		ReasoningTokens: completionTokenDetails.ReasoningTokens,
		CacheReadTokens: promptTokenDetails.CachedTokens,
	}

	return aiUsage, ai.ProviderMetadata{
		Name: streamProviderMetadata,
	}
}
