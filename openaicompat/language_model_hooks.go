package openaicompat

import (
	"encoding/json"
	"fmt"

	"github.com/charmbracelet/fantasy/ai"
	"github.com/charmbracelet/fantasy/openai"
	openaisdk "github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/packages/param"
	"github.com/openai/openai-go/v2/shared"
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

	if providerOptions.ReasoningEffort != nil {
		switch *providerOptions.ReasoningEffort {
		case openai.ReasoningEffortMinimal:
			params.ReasoningEffort = shared.ReasoningEffortMinimal
		case openai.ReasoningEffortLow:
			params.ReasoningEffort = shared.ReasoningEffortLow
		case openai.ReasoningEffortMedium:
			params.ReasoningEffort = shared.ReasoningEffortMedium
		case openai.ReasoningEffortHigh:
			params.ReasoningEffort = shared.ReasoningEffortHigh
		default:
			return nil, fmt.Errorf("reasoning model `%s` not supported", *providerOptions.ReasoningEffort)
		}
	}

	if providerOptions.User != nil {
		params.User = param.NewOpt(*providerOptions.User)
	}
	return nil, nil
}

func languageModelExtraContent(choice openaisdk.ChatCompletionChoice) []ai.Content {
	var content []ai.Content
	reasoningData := ReasoningData{}
	err := json.Unmarshal([]byte(choice.Message.RawJSON()), &reasoningData)
	if err != nil {
		return content
	}
	if reasoningData.ReasoningContent != "" {
		content = append(content, ai.ReasoningContent{
			Text: reasoningData.ReasoningContent,
		})
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

		emitEvent := func(reasoningContent string) bool {
			if !reasoningStarted {
				shouldContinue := yield(ai.StreamPart{
					Type: ai.StreamPartTypeReasoningStart,
					ID:   fmt.Sprintf("%d", inx),
				})
				if !shouldContinue {
					return false
				}
			}

			return yield(ai.StreamPart{
				Type:  ai.StreamPartTypeReasoningDelta,
				ID:    fmt.Sprintf("%d", inx),
				Delta: reasoningContent,
			})
		}
		if reasoningData.ReasoningContent != "" {
			if !reasoningStarted {
				ctx[reasoningStartedCtx] = true
			}
			return ctx, emitEvent(reasoningData.ReasoningContent)
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
