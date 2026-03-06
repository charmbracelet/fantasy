package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"charm.land/fantasy"
	"github.com/google/uuid"
	"github.com/openai/openai-go/v2/responses"
)

// generateViaWebSocket sends a response.create event over WebSocket and collects
// the full response from streaming events.
func (o responsesLanguageModel) generateViaWebSocket(ctx context.Context, params *responses.ResponseNewParams, warnings []fantasy.CallWarning, call fantasy.Call) (*fantasy.Response, error) {
	o.wsTransport.mu.Lock()
	defer o.wsTransport.mu.Unlock()

	body, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("marshal params: %w", err)
	}

	var fullInputLen int
	body, fullInputLen = o.wsTransport.applyWSOptions(body, call)

	events, err := o.wsTransport.sendResponseCreate(ctx, body)
	if err != nil {
		return nil, err
	}

	var content []fantasy.Content
	hasFunctionCall := false
	var usage fantasy.Usage
	var responseErr error

	for evt := range events {
		var streamEvent responses.ResponseStreamEventUnion
		if err := json.Unmarshal(evt.Raw, &streamEvent); err != nil {
			continue
		}

		switch evt.Type {
		case "response.completed", "response.incomplete":
			completed := streamEvent.AsResponseCompleted()
			o.wsTransport.lastResponseID = completed.Response.ID
			o.wsTransport.lastInputLen = fullInputLen

			// Build content from the completed response output
			content = nil // Reset — use the final response
			for _, outputItem := range completed.Response.Output {
				switch outputItem.Type {
				case "message":
					for _, contentPart := range outputItem.Content {
						if contentPart.Type == "output_text" {
							content = append(content, fantasy.TextContent{
								Text: contentPart.Text,
							})
							for _, annotation := range contentPart.Annotations {
								switch annotation.Type {
								case "url_citation":
									content = append(content, fantasy.SourceContent{
										SourceType: fantasy.SourceTypeURL,
										ID:         uuid.NewString(),
										URL:        annotation.URL,
										Title:      annotation.Title,
									})
								case "file_citation":
									title := "Document"
									if annotation.Filename != "" {
										title = annotation.Filename
									}
									filename := annotation.Filename
									if filename == "" {
										filename = annotation.FileID
									}
									content = append(content, fantasy.SourceContent{
										SourceType: fantasy.SourceTypeDocument,
										ID:         uuid.NewString(),
										MediaType:  "text/plain",
										Title:      title,
										Filename:   filename,
									})
								}
							}
						}
					}
				case "function_call":
					hasFunctionCall = true
					content = append(content, fantasy.ToolCallContent{
						ProviderExecuted: false,
						ToolCallID:       outputItem.CallID,
						ToolName:         outputItem.Name,
						Input:            outputItem.Arguments,
					})
				case "reasoning":
					metadata := &ResponsesReasoningMetadata{
						ItemID: outputItem.ID,
					}
					if outputItem.EncryptedContent != "" {
						metadata.EncryptedContent = &outputItem.EncryptedContent
					}
					if len(outputItem.Summary) == 0 && metadata.EncryptedContent == nil {
						continue
					}
					summaries := outputItem.Summary
					if len(summaries) == 0 {
						summaries = []responses.ResponseReasoningItemSummary{{Type: "summary_text", Text: ""}}
					}
					for _, s := range summaries {
						metadata.Summary = append(metadata.Summary, s.Text)
					}
					content = append(content, fantasy.ReasoningContent{
						Text: strings.Join(metadata.Summary, "\n"),
						ProviderMetadata: fantasy.ProviderMetadata{
							Name: metadata,
						},
					})
				}
			}

			usage = fantasy.Usage{
				InputTokens:  completed.Response.Usage.InputTokens,
				OutputTokens: completed.Response.Usage.OutputTokens,
				TotalTokens:  completed.Response.Usage.InputTokens + completed.Response.Usage.OutputTokens,
			}
			if completed.Response.Usage.OutputTokensDetails.ReasoningTokens != 0 {
				usage.ReasoningTokens = completed.Response.Usage.OutputTokensDetails.ReasoningTokens
			}
			if completed.Response.Usage.InputTokensDetails.CachedTokens != 0 {
				usage.CacheReadTokens = completed.Response.Usage.InputTokensDetails.CachedTokens
			}

		case "response.failed":
			completed := streamEvent.AsResponseCompleted()
			responseErr = fmt.Errorf("response failed: %s (code: %s)",
				completed.Response.Error.Message, completed.Response.Error.Code)

		case "error":
			errorEvent := streamEvent.AsError()
			if errorEvent.Code == "previous_response_not_found" {
				o.wsTransport.lastResponseID = ""
				o.wsTransport.lastInputLen = 0
				return nil, fmt.Errorf("previous_response_not_found")
			}
			responseErr = fmt.Errorf("%s (code: %s)", errorEvent.Message, errorEvent.Code)
		}
	}

	if responseErr != nil {
		return nil, responseErr
	}

	finishReason := mapResponsesFinishReason("", hasFunctionCall)

	return &fantasy.Response{
		Content:          content,
		Usage:            usage,
		FinishReason:     finishReason,
		ProviderMetadata: fantasy.ProviderMetadata{},
		Warnings:         warnings,
	}, nil
}

// streamViaWebSocket sends a response.create event over WebSocket and yields
// StreamParts from the server events.
func (o responsesLanguageModel) streamViaWebSocket(ctx context.Context, params *responses.ResponseNewParams, warnings []fantasy.CallWarning, call fantasy.Call) (fantasy.StreamResponse, error) {
	o.wsTransport.mu.Lock()

	body, err := json.Marshal(params)
	if err != nil {
		o.wsTransport.mu.Unlock()
		return nil, fmt.Errorf("marshal params: %w", err)
	}

	var fullInputLen int
	body, fullInputLen = o.wsTransport.applyWSOptions(body, call)

	events, err := o.wsTransport.sendResponseCreate(ctx, body)
	if err != nil {
		o.wsTransport.mu.Unlock()
		return nil, err
	}

	return func(yield func(fantasy.StreamPart) bool) {
		defer o.wsTransport.mu.Unlock()

		if len(warnings) > 0 {
			if !yield(fantasy.StreamPart{
				Type:     fantasy.StreamPartTypeWarnings,
				Warnings: warnings,
			}) {
				return
			}
		}

		finishReason := fantasy.FinishReasonUnknown
		var usage fantasy.Usage
		ongoingToolCalls := make(map[int64]*ongoingToolCall)
		hasFunctionCall := false
		activeReasoning := make(map[string]*reasoningState)

		for evt := range events {
			var event responses.ResponseStreamEventUnion
			if err := json.Unmarshal(evt.Raw, &event); err != nil {
				continue
			}

			switch evt.Type {
			case "response.created":
				_ = event.AsResponseCreated()

			case "response.output_item.added":
				added := event.AsResponseOutputItemAdded()
				switch added.Item.Type {
				case "function_call":
					ongoingToolCalls[added.OutputIndex] = &ongoingToolCall{
						toolName:   added.Item.Name,
						toolCallID: added.Item.CallID,
					}
					if !yield(fantasy.StreamPart{
						Type:         fantasy.StreamPartTypeToolInputStart,
						ID:           added.Item.CallID,
						ToolCallName: added.Item.Name,
					}) {
						return
					}
				case "message":
					if !yield(fantasy.StreamPart{
						Type: fantasy.StreamPartTypeTextStart,
						ID:   added.Item.ID,
					}) {
						return
					}
				case "reasoning":
					metadata := &ResponsesReasoningMetadata{
						ItemID:  added.Item.ID,
						Summary: []string{},
					}
					if added.Item.EncryptedContent != "" {
						metadata.EncryptedContent = &added.Item.EncryptedContent
					}
					activeReasoning[added.Item.ID] = &reasoningState{metadata: metadata}
					if !yield(fantasy.StreamPart{
						Type: fantasy.StreamPartTypeReasoningStart,
						ID:   added.Item.ID,
						ProviderMetadata: fantasy.ProviderMetadata{
							Name: metadata,
						},
					}) {
						return
					}
				}

			case "response.output_item.done":
				done := event.AsResponseOutputItemDone()
				switch done.Item.Type {
				case "function_call":
					tc := ongoingToolCalls[done.OutputIndex]
					if tc != nil {
						delete(ongoingToolCalls, done.OutputIndex)
						hasFunctionCall = true
						if !yield(fantasy.StreamPart{
							Type: fantasy.StreamPartTypeToolInputEnd,
							ID:   done.Item.CallID,
						}) {
							return
						}
						if !yield(fantasy.StreamPart{
							Type:          fantasy.StreamPartTypeToolCall,
							ID:            done.Item.CallID,
							ToolCallName:  done.Item.Name,
							ToolCallInput: done.Item.Arguments,
						}) {
							return
						}
					}
				case "message":
					if !yield(fantasy.StreamPart{
						Type: fantasy.StreamPartTypeTextEnd,
						ID:   done.Item.ID,
					}) {
						return
					}
				case "reasoning":
					state := activeReasoning[done.Item.ID]
					if state != nil {
						if !yield(fantasy.StreamPart{
							Type: fantasy.StreamPartTypeReasoningEnd,
							ID:   done.Item.ID,
							ProviderMetadata: fantasy.ProviderMetadata{
								Name: state.metadata,
							},
						}) {
							return
						}
						delete(activeReasoning, done.Item.ID)
					}
				}

			case "response.function_call_arguments.delta":
				delta := event.AsResponseFunctionCallArgumentsDelta()
				tc := ongoingToolCalls[delta.OutputIndex]
				if tc != nil {
					if !yield(fantasy.StreamPart{
						Type:  fantasy.StreamPartTypeToolInputDelta,
						ID:    tc.toolCallID,
						Delta: delta.Delta,
					}) {
						return
					}
				}

			case "response.output_text.delta":
				textDelta := event.AsResponseOutputTextDelta()
				if !yield(fantasy.StreamPart{
					Type:  fantasy.StreamPartTypeTextDelta,
					ID:    textDelta.ItemID,
					Delta: textDelta.Delta,
				}) {
					return
				}

			case "response.reasoning_summary_part.added":
				added := event.AsResponseReasoningSummaryPartAdded()
				state := activeReasoning[added.ItemID]
				if state != nil {
					state.metadata.Summary = append(state.metadata.Summary, "")
					if !yield(fantasy.StreamPart{
						Type:  fantasy.StreamPartTypeReasoningDelta,
						ID:    added.ItemID,
						Delta: "\n",
						ProviderMetadata: fantasy.ProviderMetadata{
							Name: state.metadata,
						},
					}) {
						return
					}
				}

			case "response.reasoning_summary_text.delta":
				textDelta := event.AsResponseReasoningSummaryTextDelta()
				state := activeReasoning[textDelta.ItemID]
				if state != nil {
					if len(state.metadata.Summary)-1 >= int(textDelta.SummaryIndex) {
						state.metadata.Summary[textDelta.SummaryIndex] += textDelta.Delta
					}
					if !yield(fantasy.StreamPart{
						Type:  fantasy.StreamPartTypeReasoningDelta,
						ID:    textDelta.ItemID,
						Delta: textDelta.Delta,
						ProviderMetadata: fantasy.ProviderMetadata{
							Name: state.metadata,
						},
					}) {
						return
					}
				}

			case "response.completed", "response.incomplete":
				completed := event.AsResponseCompleted()
				o.wsTransport.lastResponseID = completed.Response.ID
				o.wsTransport.lastInputLen = fullInputLen
				finishReason = mapResponsesFinishReason(completed.Response.IncompleteDetails.Reason, hasFunctionCall)
				usage = fantasy.Usage{
					InputTokens:  completed.Response.Usage.InputTokens,
					OutputTokens: completed.Response.Usage.OutputTokens,
					TotalTokens:  completed.Response.Usage.InputTokens + completed.Response.Usage.OutputTokens,
				}
				if completed.Response.Usage.OutputTokensDetails.ReasoningTokens != 0 {
					usage.ReasoningTokens = completed.Response.Usage.OutputTokensDetails.ReasoningTokens
				}
				if completed.Response.Usage.InputTokensDetails.CachedTokens != 0 {
					usage.CacheReadTokens = completed.Response.Usage.InputTokensDetails.CachedTokens
				}

			case "response.failed":
				completed := event.AsResponseCompleted()
				if !yield(fantasy.StreamPart{
					Type:  fantasy.StreamPartTypeError,
					Error: fmt.Errorf("response failed: %s (code: %s)", completed.Response.Error.Message, completed.Response.Error.Code),
				}) {
					return
				}
				return

			case "error":
				errorEvent := event.AsError()
				if errorEvent.Code == "previous_response_not_found" {
					o.wsTransport.lastResponseID = ""
					o.wsTransport.lastInputLen = 0
				}
				if !yield(fantasy.StreamPart{
					Type:  fantasy.StreamPartTypeError,
					Error: fmt.Errorf("response error: %s (code: %s)", errorEvent.Message, errorEvent.Code),
				}) {
					return
				}
				return
			}
		}

		yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeFinish,
			Usage:        usage,
			FinishReason: finishReason,
		})
	}, nil
}
