package openaicompat

import (
	"encoding/json"
	"fmt"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openai"
	openaisdk "github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/packages/param"
	"github.com/openai/openai-go/v2/shared"
)

const reasoningStartedCtx = "reasoning_started"

// PrepareCallFunc prepares the call for the language model.
func PrepareCallFunc(_ fantasy.LanguageModel, params *openaisdk.ChatCompletionNewParams, call fantasy.Call) ([]fantasy.CallWarning, error) {
	providerOptions := &ProviderOptions{}
	if v, ok := call.ProviderOptions[Name]; ok {
		providerOptions, ok = v.(*ProviderOptions)
		if !ok {
			return nil, fantasy.NewInvalidArgumentError("providerOptions", "openrouter provider options should be *openrouter.ProviderOptions", nil)
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

// ExtraContentFunc adds extra content to the response.
func ExtraContentFunc(choice openaisdk.ChatCompletionChoice) []fantasy.Content {
	var content []fantasy.Content
	reasoningData := ReasoningData{}
	err := json.Unmarshal([]byte(choice.Message.RawJSON()), &reasoningData)
	if err != nil {
		return content
	}
	if reasoningData.ReasoningContent != "" {
		content = append(content, fantasy.ReasoningContent{
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

// StreamExtraFunc handles extra functionality for streaming responses.
func StreamExtraFunc(chunk openaisdk.ChatCompletionChunk, yield func(fantasy.StreamPart) bool, ctx map[string]any) (map[string]any, bool) {
	if len(chunk.Choices) == 0 {
		return ctx, true
	}

	reasoningStarted := extractReasoningContext(ctx)

	for inx, choice := range chunk.Choices {
		reasoningData := ReasoningData{}
		err := json.Unmarshal([]byte(choice.Delta.RawJSON()), &reasoningData)
		if err != nil {
			yield(fantasy.StreamPart{
				Type:  fantasy.StreamPartTypeError,
				Error: fantasy.NewAIError("error unmarshalling delta", err),
			})
			return ctx, false
		}

		emitEvent := func(reasoningContent string) bool {
			if !reasoningStarted {
				shouldContinue := yield(fantasy.StreamPart{
					Type: fantasy.StreamPartTypeReasoningStart,
					ID:   fmt.Sprintf("%d", inx),
				})
				if !shouldContinue {
					return false
				}
			}

			return yield(fantasy.StreamPart{
				Type:  fantasy.StreamPartTypeReasoningDelta,
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
			return ctx, yield(fantasy.StreamPart{
				Type: fantasy.StreamPartTypeReasoningEnd,
				ID:   fmt.Sprintf("%d", inx),
			})
		}
	}
	return ctx, true
}
