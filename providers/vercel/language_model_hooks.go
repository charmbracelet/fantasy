package vercel

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"maps"
	"strings"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/anthropic"
	"charm.land/fantasy/providers/google"
	openaipkg "charm.land/fantasy/providers/openai"
	openaisdk "github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/packages/param"
)

const reasoningStartedCtx = "reasoning_started"

type currentReasoningState struct {
	metadata       *openaipkg.ResponsesReasoningMetadata
	googleMetadata *google.ReasoningMetadata
	googleText     string
	anthropicSig   string
}

func languagePrepareModelCall(_ fantasy.LanguageModel, params *openaisdk.ChatCompletionNewParams, call fantasy.Call) ([]fantasy.CallWarning, error) {
	providerOptions := &ProviderOptions{}
	if v, ok := call.ProviderOptions[Name]; ok {
		providerOptions, ok = v.(*ProviderOptions)
		if !ok {
			return nil, &fantasy.Error{Title: "invalid argument", Message: "vercel provider options should be *vercel.ProviderOptions"}
		}
	}

	extraFields := make(map[string]any)

	// Handle reasoning options
	if providerOptions.Reasoning != nil {
		data, err := structToMapJSON(providerOptions.Reasoning)
		if err != nil {
			return nil, err
		}
		extraFields["reasoning"] = data
	}

	// Handle provider options for gateway routing
	if providerOptions.ProviderOptions != nil {
		data, err := structToMapJSON(providerOptions.ProviderOptions)
		if err != nil {
			return nil, err
		}
		extraFields["providerOptions"] = map[string]any{
			"gateway": data,
		}
	}

	// Handle BYOK (Bring Your Own Key)
	if providerOptions.BYOK != nil {
		data, err := structToMapJSON(providerOptions.BYOK)
		if err != nil {
			return nil, err
		}
		if gatewayOpts, ok := extraFields["providerOptions"].(map[string]any); ok {
			gatewayOpts["byok"] = data
		} else {
			extraFields["providerOptions"] = map[string]any{
				"gateway": map[string]any{
					"byok": data,
				},
			}
		}
	}

	// Handle standard OpenAI options
	if providerOptions.LogitBias != nil {
		params.LogitBias = providerOptions.LogitBias
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

	// Handle model fallbacks - direct models field
	if providerOptions.ProviderOptions != nil && len(providerOptions.ProviderOptions.Models) > 0 {
		extraFields["models"] = providerOptions.ProviderOptions.Models
	}

	maps.Copy(extraFields, providerOptions.ExtraBody)
	params.SetExtraFields(extraFields)
	return nil, nil
}

func languageModelExtraContent(choice openaisdk.ChatCompletionChoice) []fantasy.Content {
	content := make([]fantasy.Content, 0)
	reasoningData := ReasoningData{}
	err := json.Unmarshal([]byte(choice.Message.RawJSON()), &reasoningData)
	if err != nil {
		return content
	}

	responsesReasoningBlocks := make([]openaipkg.ResponsesReasoningMetadata, 0)
	anthropicReasoningBlocks := make([]struct {
		text     string
		metadata *anthropic.ReasoningOptionMetadata
	}, 0)
	googleReasoningBlocks := make([]struct {
		text     string
		metadata *google.ReasoningMetadata
	}, 0)
	otherReasoning := make([]string, 0)

	for _, detail := range reasoningData.ReasoningDetails {
		if strings.HasPrefix(detail.Format, "openai-responses") || strings.HasPrefix(detail.Format, "xai-responses") {
			var thinkingBlock openaipkg.ResponsesReasoningMetadata
			if len(responsesReasoningBlocks)-1 >= detail.Index {
				thinkingBlock = responsesReasoningBlocks[detail.Index]
			} else {
				thinkingBlock = openaipkg.ResponsesReasoningMetadata{}
				responsesReasoningBlocks = append(responsesReasoningBlocks, thinkingBlock)
			}

			switch detail.Type {
			case "reasoning.summary":
				thinkingBlock.Summary = append(thinkingBlock.Summary, detail.Summary)
			case "reasoning.encrypted":
				thinkingBlock.EncryptedContent = &detail.Data
			}
			if detail.ID != "" {
				thinkingBlock.ItemID = detail.ID
			}
			responsesReasoningBlocks[detail.Index] = thinkingBlock
			continue
		}

		if strings.HasPrefix(detail.Format, "google-gemini") {
			var thinkingBlock struct {
				text     string
				metadata *google.ReasoningMetadata
			}
			if len(googleReasoningBlocks)-1 >= detail.Index {
				thinkingBlock = googleReasoningBlocks[detail.Index]
			} else {
				thinkingBlock = struct {
					text     string
					metadata *google.ReasoningMetadata
				}{metadata: &google.ReasoningMetadata{}}
				googleReasoningBlocks = append(googleReasoningBlocks, thinkingBlock)
			}

			switch detail.Type {
			case "reasoning.text":
				thinkingBlock.text = detail.Text
			case "reasoning.encrypted":
				thinkingBlock.metadata.Signature = detail.Data
				thinkingBlock.metadata.ToolID = detail.ID
			}
			googleReasoningBlocks[detail.Index] = thinkingBlock
			continue
		}

		if strings.HasPrefix(detail.Format, "anthropic-claude") {
			anthropicReasoningBlocks = append(anthropicReasoningBlocks, struct {
				text     string
				metadata *anthropic.ReasoningOptionMetadata
			}{
				text: detail.Text,
				metadata: &anthropic.ReasoningOptionMetadata{
					Signature: detail.Signature,
				},
			})
			continue
		}

		otherReasoning = append(otherReasoning, detail.Text)
	}

	// Fallback to simple reasoning field if no details
	if reasoningData.Reasoning != "" && len(reasoningData.ReasoningDetails) == 0 {
		otherReasoning = append(otherReasoning, reasoningData.Reasoning)
	}

	for _, block := range responsesReasoningBlocks {
		if len(block.Summary) == 0 {
			block.Summary = []string{""}
		}
		content = append(content, fantasy.ReasoningContent{
			Text: strings.Join(block.Summary, "\n"),
			ProviderMetadata: fantasy.ProviderMetadata{
				openaipkg.Name: &block,
			},
		})
	}

	for _, block := range anthropicReasoningBlocks {
		content = append(content, fantasy.ReasoningContent{
			Text: block.text,
			ProviderMetadata: fantasy.ProviderMetadata{
				anthropic.Name: block.metadata,
			},
		})
	}

	for _, block := range googleReasoningBlocks {
		content = append(content, fantasy.ReasoningContent{
			Text: block.text,
			ProviderMetadata: fantasy.ProviderMetadata{
				google.Name: block.metadata,
			},
		})
	}

	for _, reasoning := range otherReasoning {
		if reasoning != "" {
			content = append(content, fantasy.ReasoningContent{
				Text: reasoning,
			})
		}
	}

	return content
}

func extractReasoningContext(ctx map[string]any) *currentReasoningState {
	reasoningStarted, ok := ctx[reasoningStartedCtx]
	if !ok {
		return nil
	}
	state, ok := reasoningStarted.(*currentReasoningState)
	if !ok {
		return nil
	}
	return state
}

func languageModelStreamExtra(chunk openaisdk.ChatCompletionChunk, yield func(fantasy.StreamPart) bool, ctx map[string]any) (map[string]any, bool) {
	if len(chunk.Choices) == 0 {
		return ctx, true
	}

	currentState := extractReasoningContext(ctx)

	inx := 0
	choice := chunk.Choices[inx]
	reasoningData := ReasoningData{}
	err := json.Unmarshal([]byte(choice.Delta.RawJSON()), &reasoningData)
	if err != nil {
		yield(fantasy.StreamPart{
			Type:  fantasy.StreamPartTypeError,
			Error: &fantasy.Error{Title: "stream error", Message: "error unmarshalling delta", Cause: err},
		})
		return ctx, false
	}

	// Reasoning Start
	if currentState == nil {
		if len(reasoningData.ReasoningDetails) == 0 && reasoningData.Reasoning == "" {
			return ctx, true
		}

		var metadata fantasy.ProviderMetadata
		currentState = &currentReasoningState{}

		if len(reasoningData.ReasoningDetails) > 0 {
			detail := reasoningData.ReasoningDetails[0]

			if strings.HasPrefix(detail.Format, "openai-responses") || strings.HasPrefix(detail.Format, "xai-responses") {
				currentState.metadata = &openaipkg.ResponsesReasoningMetadata{
					Summary: []string{detail.Summary},
				}
				metadata = fantasy.ProviderMetadata{
					openaipkg.Name: currentState.metadata,
				}
				if detail.Data != "" {
					shouldContinue := yield(fantasy.StreamPart{
						Type:             fantasy.StreamPartTypeReasoningStart,
						ID:               fmt.Sprintf("%d", inx),
						Delta:            detail.Summary,
						ProviderMetadata: metadata,
					})
					if !shouldContinue {
						return ctx, false
					}
					return ctx, yield(fantasy.StreamPart{
						Type: fantasy.StreamPartTypeReasoningEnd,
						ID:   fmt.Sprintf("%d", inx),
						ProviderMetadata: fantasy.ProviderMetadata{
							openaipkg.Name: &openaipkg.ResponsesReasoningMetadata{
								Summary:          []string{detail.Summary},
								EncryptedContent: &detail.Data,
								ItemID:           detail.ID,
							},
						},
					})
				}
			}

			if strings.HasPrefix(detail.Format, "google-gemini") {
				if detail.Type == "reasoning.encrypted" {
					ctx[reasoningStartedCtx] = nil
					if !yield(fantasy.StreamPart{
						Type: fantasy.StreamPartTypeReasoningStart,
						ID:   fmt.Sprintf("%d", inx),
					}) {
						return ctx, false
					}
					return ctx, yield(fantasy.StreamPart{
						Type: fantasy.StreamPartTypeReasoningEnd,
						ID:   fmt.Sprintf("%d", inx),
						ProviderMetadata: fantasy.ProviderMetadata{
							google.Name: &google.ReasoningMetadata{
								Signature: detail.Data,
								ToolID:    detail.ID,
							},
						},
					})
				}
				currentState.googleMetadata = &google.ReasoningMetadata{}
				currentState.googleText = detail.Text
				metadata = fantasy.ProviderMetadata{
					google.Name: currentState.googleMetadata,
				}
			}

			if strings.HasPrefix(detail.Format, "anthropic-claude") {
				currentState.anthropicSig = detail.Signature
			}
		}

		ctx[reasoningStartedCtx] = currentState
		delta := reasoningData.Reasoning
		if len(reasoningData.ReasoningDetails) > 0 {
			delta = reasoningData.ReasoningDetails[0].Summary
			if strings.HasPrefix(reasoningData.ReasoningDetails[0].Format, "google-gemini") {
				delta = reasoningData.ReasoningDetails[0].Text
			}
			if strings.HasPrefix(reasoningData.ReasoningDetails[0].Format, "anthropic-claude") {
				delta = reasoningData.ReasoningDetails[0].Text
			}
		}
		return ctx, yield(fantasy.StreamPart{
			Type:             fantasy.StreamPartTypeReasoningStart,
			ID:               fmt.Sprintf("%d", inx),
			Delta:            delta,
			ProviderMetadata: metadata,
		})
	}

	if len(reasoningData.ReasoningDetails) == 0 && reasoningData.Reasoning == "" {
		if choice.Delta.Content != "" || len(choice.Delta.ToolCalls) > 0 {
			ctx[reasoningStartedCtx] = nil
			return ctx, yield(fantasy.StreamPart{
				Type: fantasy.StreamPartTypeReasoningEnd,
				ID:   fmt.Sprintf("%d", inx),
			})
		}
		return ctx, true
	}

	if len(reasoningData.ReasoningDetails) > 0 {
		detail := reasoningData.ReasoningDetails[0]

		if strings.HasPrefix(detail.Format, "openai-responses") || strings.HasPrefix(detail.Format, "xai-responses") {
			if detail.Data != "" {
				currentState.metadata.EncryptedContent = &detail.Data
				currentState.metadata.ItemID = detail.ID
				ctx[reasoningStartedCtx] = nil
				return ctx, yield(fantasy.StreamPart{
					Type: fantasy.StreamPartTypeReasoningEnd,
					ID:   fmt.Sprintf("%d", inx),
					ProviderMetadata: fantasy.ProviderMetadata{
						openaipkg.Name: currentState.metadata,
					},
				})
			}
			var textDelta string
			if len(currentState.metadata.Summary)-1 >= detail.Index {
				currentState.metadata.Summary[detail.Index] += detail.Summary
				textDelta = detail.Summary
			} else {
				currentState.metadata.Summary = append(currentState.metadata.Summary, detail.Summary)
				textDelta = "\n" + detail.Summary
			}
			ctx[reasoningStartedCtx] = currentState
			return ctx, yield(fantasy.StreamPart{
				Type:  fantasy.StreamPartTypeReasoningDelta,
				ID:    fmt.Sprintf("%d", inx),
				Delta: textDelta,
				ProviderMetadata: fantasy.ProviderMetadata{
					openaipkg.Name: currentState.metadata,
				},
			})
		}

		if strings.HasPrefix(detail.Format, "anthropic-claude") {
			if detail.Signature != "" {
				metadata := fantasy.ProviderMetadata{
					anthropic.Name: &anthropic.ReasoningOptionMetadata{
						Signature: detail.Signature,
					},
				}
				shouldContinue := yield(fantasy.StreamPart{
					Type:             fantasy.StreamPartTypeReasoningDelta,
					ID:               fmt.Sprintf("%d", inx),
					Delta:            detail.Text,
					ProviderMetadata: metadata,
				})
				if !shouldContinue {
					return ctx, false
				}
				ctx[reasoningStartedCtx] = nil
				return ctx, yield(fantasy.StreamPart{
					Type: fantasy.StreamPartTypeReasoningEnd,
					ID:   fmt.Sprintf("%d", inx),
				})
			}
			return ctx, yield(fantasy.StreamPart{
				Type:  fantasy.StreamPartTypeReasoningDelta,
				ID:    fmt.Sprintf("%d", inx),
				Delta: detail.Text,
			})
		}

		if strings.HasPrefix(detail.Format, "google-gemini") {
			if detail.Type == "reasoning.text" {
				currentState.googleText += detail.Text
				ctx[reasoningStartedCtx] = currentState
				return ctx, yield(fantasy.StreamPart{
					Type:  fantasy.StreamPartTypeReasoningDelta,
					ID:    fmt.Sprintf("%d", inx),
					Delta: detail.Text,
				})
			}
			if detail.Type == "reasoning.encrypted" {
				currentState.googleMetadata.Signature = detail.Data
				currentState.googleMetadata.ToolID = detail.ID
				metadata := fantasy.ProviderMetadata{
					google.Name: currentState.googleMetadata,
				}
				ctx[reasoningStartedCtx] = nil
				return ctx, yield(fantasy.StreamPart{
					Type:             fantasy.StreamPartTypeReasoningEnd,
					ID:               fmt.Sprintf("%d", inx),
					ProviderMetadata: metadata,
				})
			}
		}

		return ctx, yield(fantasy.StreamPart{
			Type:  fantasy.StreamPartTypeReasoningDelta,
			ID:    fmt.Sprintf("%d", inx),
			Delta: detail.Text,
		})
	}

	if reasoningData.Reasoning != "" {
		return ctx, yield(fantasy.StreamPart{
			Type:  fantasy.StreamPartTypeReasoningDelta,
			ID:    fmt.Sprintf("%d", inx),
			Delta: reasoningData.Reasoning,
		})
	}

	return ctx, true
}

func languageModelUsage(response openaisdk.ChatCompletion) (fantasy.Usage, fantasy.ProviderOptionsData) {
	if len(response.Choices) == 0 {
		return fantasy.Usage{}, nil
	}

	usage := response.Usage
	completionTokenDetails := usage.CompletionTokensDetails
	promptTokenDetails := usage.PromptTokensDetails

	var provider string
	if p, ok := response.JSON.ExtraFields["provider"]; ok {
		provider = p.Raw()
	}

	providerMetadata := &ProviderMetadata{
		Provider: provider,
	}

	return fantasy.Usage{
		InputTokens:     usage.PromptTokens,
		OutputTokens:    usage.CompletionTokens,
		TotalTokens:     usage.TotalTokens,
		ReasoningTokens: completionTokenDetails.ReasoningTokens,
		CacheReadTokens: promptTokenDetails.CachedTokens,
	}, providerMetadata
}

func languageModelStreamUsage(chunk openaisdk.ChatCompletionChunk, _ map[string]any, metadata fantasy.ProviderMetadata) (fantasy.Usage, fantasy.ProviderMetadata) {
	usage := chunk.Usage
	if usage.TotalTokens == 0 {
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

	if p, ok := chunk.JSON.ExtraFields["provider"]; ok {
		streamProviderMetadata.Provider = p.Raw()
	}

	completionTokenDetails := usage.CompletionTokensDetails
	promptTokenDetails := usage.PromptTokensDetails
	aiUsage := fantasy.Usage{
		InputTokens:     usage.PromptTokens,
		OutputTokens:    usage.CompletionTokens,
		TotalTokens:     usage.TotalTokens,
		ReasoningTokens: completionTokenDetails.ReasoningTokens,
		CacheReadTokens: promptTokenDetails.CachedTokens,
	}

	return aiUsage, fantasy.ProviderMetadata{
		Name: streamProviderMetadata,
	}
}

func languageModelToPrompt(prompt fantasy.Prompt, _, model string) ([]openaisdk.ChatCompletionMessageParamUnion, []fantasy.CallWarning) {
	var messages []openaisdk.ChatCompletionMessageParamUnion
	var warnings []fantasy.CallWarning

	for _, msg := range prompt {
		switch msg.Role {
		case fantasy.MessageRoleSystem:
			var systemPromptParts []string
			for _, c := range msg.Content {
				if c.GetType() != fantasy.ContentTypeText {
					warnings = append(warnings, fantasy.CallWarning{
						Type:    fantasy.CallWarningTypeOther,
						Message: "system prompt can only have text content",
					})
					continue
				}
				textPart, ok := fantasy.AsContentType[fantasy.TextPart](c)
				if !ok {
					warnings = append(warnings, fantasy.CallWarning{
						Type:    fantasy.CallWarningTypeOther,
						Message: "system prompt text part does not have the right type",
					})
					continue
				}
				text := textPart.Text
				if strings.TrimSpace(text) != "" {
					systemPromptParts = append(systemPromptParts, textPart.Text)
				}
			}
			if len(systemPromptParts) == 0 {
				warnings = append(warnings, fantasy.CallWarning{
					Type:    fantasy.CallWarningTypeOther,
					Message: "system prompt has no text parts",
				})
				continue
			}
			systemMsg := openaisdk.SystemMessage(strings.Join(systemPromptParts, "\n"))
			anthropicCache := anthropic.GetCacheControl(msg.ProviderOptions)
			if anthropicCache != nil {
				systemMsg.OfSystem.SetExtraFields(map[string]any{
					"cache_control": map[string]string{
						"type": anthropicCache.Type,
					},
				})
			}
			messages = append(messages, systemMsg)

		case fantasy.MessageRoleUser:
			if len(msg.Content) == 1 && msg.Content[0].GetType() == fantasy.ContentTypeText {
				textPart, ok := fantasy.AsContentType[fantasy.TextPart](msg.Content[0])
				if !ok {
					warnings = append(warnings, fantasy.CallWarning{
						Type:    fantasy.CallWarningTypeOther,
						Message: "user message text part does not have the right type",
					})
					continue
				}
				userMsg := openaisdk.UserMessage(textPart.Text)
				anthropicCache := anthropic.GetCacheControl(msg.ProviderOptions)
				if anthropicCache != nil {
					userMsg.OfUser.SetExtraFields(map[string]any{
						"cache_control": map[string]string{
							"type": anthropicCache.Type,
						},
					})
				}
				messages = append(messages, userMsg)
				continue
			}

			var content []openaisdk.ChatCompletionContentPartUnionParam
			for i, c := range msg.Content {
				isLastPart := i == len(msg.Content)-1
				cacheControl := anthropic.GetCacheControl(c.Options())
				if cacheControl == nil && isLastPart {
					cacheControl = anthropic.GetCacheControl(msg.ProviderOptions)
				}
				switch c.GetType() {
				case fantasy.ContentTypeText:
					textPart, ok := fantasy.AsContentType[fantasy.TextPart](c)
					if !ok {
						warnings = append(warnings, fantasy.CallWarning{
							Type:    fantasy.CallWarningTypeOther,
							Message: "user message text part does not have the right type",
						})
						continue
					}
					part := openaisdk.ChatCompletionContentPartUnionParam{
						OfText: &openaisdk.ChatCompletionContentPartTextParam{
							Text: textPart.Text,
						},
					}
					if cacheControl != nil {
						part.OfText.SetExtraFields(map[string]any{
							"cache_control": map[string]string{
								"type": cacheControl.Type,
							},
						})
					}
					content = append(content, part)
				case fantasy.ContentTypeFile:
					filePart, ok := fantasy.AsContentType[fantasy.FilePart](c)
					if !ok {
						warnings = append(warnings, fantasy.CallWarning{
							Type:    fantasy.CallWarningTypeOther,
							Message: "user message file part does not have the right type",
						})
						continue
					}
					switch {
					case strings.HasPrefix(filePart.MediaType, "image/"):
						base64Encoded := base64.StdEncoding.EncodeToString(filePart.Data)
						data := "data:" + filePart.MediaType + ";base64," + base64Encoded
						imageURL := openaisdk.ChatCompletionContentPartImageImageURLParam{URL: data}
						if providerOptions, ok := filePart.ProviderOptions[openaipkg.Name]; ok {
							if detail, ok := providerOptions.(*openaipkg.ProviderFileOptions); ok {
								imageURL.Detail = detail.ImageDetail
							}
						}
						imageBlock := openaisdk.ChatCompletionContentPartImageParam{ImageURL: imageURL}
						if cacheControl != nil {
							imageBlock.SetExtraFields(map[string]any{
								"cache_control": map[string]string{
									"type": cacheControl.Type,
								},
							})
						}
						content = append(content, openaisdk.ChatCompletionContentPartUnionParam{OfImageURL: &imageBlock})

					case filePart.MediaType == "audio/wav":
						base64Encoded := base64.StdEncoding.EncodeToString(filePart.Data)
						audioBlock := openaisdk.ChatCompletionContentPartInputAudioParam{
							InputAudio: openaisdk.ChatCompletionContentPartInputAudioInputAudioParam{
								Data:   base64Encoded,
								Format: "wav",
							},
						}
						if cacheControl != nil {
							audioBlock.SetExtraFields(map[string]any{
								"cache_control": map[string]string{
									"type": cacheControl.Type,
								},
							})
						}
						content = append(content, openaisdk.ChatCompletionContentPartUnionParam{OfInputAudio: &audioBlock})

					case filePart.MediaType == "audio/mpeg" || filePart.MediaType == "audio/mp3":
						base64Encoded := base64.StdEncoding.EncodeToString(filePart.Data)
						audioBlock := openaisdk.ChatCompletionContentPartInputAudioParam{
							InputAudio: openaisdk.ChatCompletionContentPartInputAudioInputAudioParam{
								Data:   base64Encoded,
								Format: "mp3",
							},
						}
						if cacheControl != nil {
							audioBlock.SetExtraFields(map[string]any{
								"cache_control": map[string]string{
									"type": cacheControl.Type,
								},
							})
						}
						content = append(content, openaisdk.ChatCompletionContentPartUnionParam{OfInputAudio: &audioBlock})

					case filePart.MediaType == "application/pdf":
						dataStr := string(filePart.Data)
						if strings.HasPrefix(dataStr, "file-") {
							fileBlock := openaisdk.ChatCompletionContentPartFileParam{
								File: openaisdk.ChatCompletionContentPartFileFileParam{
									FileID: param.NewOpt(dataStr),
								},
							}
							if cacheControl != nil {
								fileBlock.SetExtraFields(map[string]any{
									"cache_control": map[string]string{
										"type": cacheControl.Type,
									},
								})
							}
							content = append(content, openaisdk.ChatCompletionContentPartUnionParam{OfFile: &fileBlock})
						} else {
							base64Encoded := base64.StdEncoding.EncodeToString(filePart.Data)
							data := "data:application/pdf;base64," + base64Encoded
							filename := filePart.Filename
							if filename == "" {
								filename = fmt.Sprintf("part-%d.pdf", len(content))
							}
							fileBlock := openaisdk.ChatCompletionContentPartFileParam{
								File: openaisdk.ChatCompletionContentPartFileFileParam{
									Filename: param.NewOpt(filename),
									FileData: param.NewOpt(data),
								},
							}
							if cacheControl != nil {
								fileBlock.SetExtraFields(map[string]any{
									"cache_control": map[string]string{
										"type": cacheControl.Type,
									},
								})
							}
							content = append(content, openaisdk.ChatCompletionContentPartUnionParam{OfFile: &fileBlock})
						}

					default:
						warnings = append(warnings, fantasy.CallWarning{
							Type:    fantasy.CallWarningTypeOther,
							Message: fmt.Sprintf("file part media type %s not supported", filePart.MediaType),
						})
					}
				}
			}
			if !hasVisibleUserContent(content) {
				warnings = append(warnings, fantasy.CallWarning{
					Type:    fantasy.CallWarningTypeOther,
					Message: "dropping empty user message (contains neither user-facing content nor tool results)",
				})
				continue
			}
			messages = append(messages, openaisdk.UserMessage(content))

		case fantasy.MessageRoleAssistant:
			if len(msg.Content) == 1 && msg.Content[0].GetType() == fantasy.ContentTypeText {
				textPart, ok := fantasy.AsContentType[fantasy.TextPart](msg.Content[0])
				if !ok {
					warnings = append(warnings, fantasy.CallWarning{
						Type:    fantasy.CallWarningTypeOther,
						Message: "assistant message text part does not have the right type",
					})
					continue
				}
				assistantMsg := openaisdk.AssistantMessage(textPart.Text)
				anthropicCache := anthropic.GetCacheControl(msg.ProviderOptions)
				if anthropicCache != nil {
					assistantMsg.OfAssistant.SetExtraFields(map[string]any{
						"cache_control": map[string]string{
							"type": anthropicCache.Type,
						},
					})
				}
				messages = append(messages, assistantMsg)
				continue
			}

			assistantMsg := openaisdk.ChatCompletionAssistantMessageParam{
				Role: "assistant",
			}
			for i, c := range msg.Content {
				isLastPart := i == len(msg.Content)-1
				cacheControl := anthropic.GetCacheControl(c.Options())
				if cacheControl == nil && isLastPart {
					cacheControl = anthropic.GetCacheControl(msg.ProviderOptions)
				}
				switch c.GetType() {
				case fantasy.ContentTypeText:
					textPart, ok := fantasy.AsContentType[fantasy.TextPart](c)
					if !ok {
						warnings = append(warnings, fantasy.CallWarning{
							Type:    fantasy.CallWarningTypeOther,
							Message: "assistant message text part does not have the right type",
						})
						continue
					}
					if assistantMsg.Content.OfString.Valid() {
						textPart.Text = assistantMsg.Content.OfString.Value + "\n" + textPart.Text
					}
					assistantMsg.Content = openaisdk.ChatCompletionAssistantMessageParamContentUnion{
						OfString: param.NewOpt(textPart.Text),
					}
					if cacheControl != nil {
						assistantMsg.Content.SetExtraFields(map[string]any{
							"cache_control": map[string]string{
								"type": cacheControl.Type,
							},
						})
					}
				case fantasy.ContentTypeReasoning:
					reasoningPart, ok := fantasy.AsContentType[fantasy.ReasoningPart](c)
					if !ok {
						warnings = append(warnings, fantasy.CallWarning{
							Type:    fantasy.CallWarningTypeOther,
							Message: "assistant message reasoning part does not have the right type",
						})
						continue
					}
					var reasoningDetails []ReasoningDetail
					switch {
					case strings.HasPrefix(model, "anthropic/") && reasoningPart.Text != "":
						metadata := anthropic.GetReasoningMetadata(reasoningPart.Options())
						if metadata == nil {
							text := fmt.Sprintf("<thoughts>%s</thoughts>", reasoningPart.Text)
							if assistantMsg.Content.OfString.Valid() {
								text = assistantMsg.Content.OfString.Value + "\n" + text
							}
							assistantMsg.Content = openaisdk.ChatCompletionAssistantMessageParamContentUnion{
								OfString: param.NewOpt(text),
							}
							if cacheControl != nil {
								assistantMsg.Content.SetExtraFields(map[string]any{
									"cache_control": map[string]string{
										"type": cacheControl.Type,
									},
								})
							}
							continue
						}
						reasoningDetails = append(reasoningDetails, ReasoningDetail{
							Format:    "anthropic-claude-v1",
							Type:      "reasoning.text",
							Text:      reasoningPart.Text,
							Signature: metadata.Signature,
						})
						data, _ := json.Marshal(reasoningDetails)
						reasoningDetailsMap := []map[string]any{}
						_ = json.Unmarshal(data, &reasoningDetailsMap)
						assistantMsg.SetExtraFields(map[string]any{
							"reasoning_details": reasoningDetailsMap,
							"reasoning":         reasoningPart.Text,
						})
					case strings.HasPrefix(model, "openai/"):
						metadata := openaipkg.GetReasoningMetadata(reasoningPart.Options())
						if metadata == nil {
							text := fmt.Sprintf("<thoughts>%s</thoughts>", reasoningPart.Text)
							if assistantMsg.Content.OfString.Valid() {
								text = assistantMsg.Content.OfString.Value + "\n" + text
							}
							assistantMsg.Content = openaisdk.ChatCompletionAssistantMessageParamContentUnion{
								OfString: param.NewOpt(text),
							}
							continue
						}
						for inx, summary := range metadata.Summary {
							if summary == "" {
								continue
							}
							reasoningDetails = append(reasoningDetails, ReasoningDetail{
								Type:    "reasoning.summary",
								Format:  "openai-responses-v1",
								Summary: summary,
								Index:   inx,
							})
						}
						reasoningDetails = append(reasoningDetails, ReasoningDetail{
							Type:   "reasoning.encrypted",
							Format: "openai-responses-v1",
							Data:   *metadata.EncryptedContent,
							ID:     metadata.ItemID,
						})
						data, _ := json.Marshal(reasoningDetails)
						reasoningDetailsMap := []map[string]any{}
						_ = json.Unmarshal(data, &reasoningDetailsMap)
						assistantMsg.SetExtraFields(map[string]any{
							"reasoning_details": reasoningDetailsMap,
						})
					case strings.HasPrefix(model, "google/"):
						metadata := google.GetReasoningMetadata(reasoningPart.Options())
						if metadata == nil {
							text := fmt.Sprintf("<thoughts>%s</thoughts>", reasoningPart.Text)
							if assistantMsg.Content.OfString.Valid() {
								text = assistantMsg.Content.OfString.Value + "\n" + text
							}
							assistantMsg.Content = openaisdk.ChatCompletionAssistantMessageParamContentUnion{
								OfString: param.NewOpt(text),
							}
							continue
						}
						if reasoningPart.Text != "" {
							reasoningDetails = append(reasoningDetails, ReasoningDetail{
								Type:   "reasoning.text",
								Format: "google-gemini-v1",
								Text:   reasoningPart.Text,
							})
						}
						reasoningDetails = append(reasoningDetails, ReasoningDetail{
							Type:   "reasoning.encrypted",
							Format: "google-gemini-v1",
							Data:   metadata.Signature,
							ID:     metadata.ToolID,
						})
						data, _ := json.Marshal(reasoningDetails)
						reasoningDetailsMap := []map[string]any{}
						_ = json.Unmarshal(data, &reasoningDetailsMap)
						assistantMsg.SetExtraFields(map[string]any{
							"reasoning_details": reasoningDetailsMap,
						})
					default:
						reasoningDetails = append(reasoningDetails, ReasoningDetail{
							Type:   "reasoning.text",
							Text:   reasoningPart.Text,
							Format: "unknown",
						})
						data, _ := json.Marshal(reasoningDetails)
						reasoningDetailsMap := []map[string]any{}
						_ = json.Unmarshal(data, &reasoningDetailsMap)
						assistantMsg.SetExtraFields(map[string]any{
							"reasoning_details": reasoningDetailsMap,
						})
					}
				case fantasy.ContentTypeToolCall:
					toolCallPart, ok := fantasy.AsContentType[fantasy.ToolCallPart](c)
					if !ok {
						warnings = append(warnings, fantasy.CallWarning{
							Type:    fantasy.CallWarningTypeOther,
							Message: "assistant message tool part does not have the right type",
						})
						continue
					}
					tc := openaisdk.ChatCompletionMessageToolCallUnionParam{
						OfFunction: &openaisdk.ChatCompletionMessageFunctionToolCallParam{
							ID:   toolCallPart.ToolCallID,
							Type: "function",
							Function: openaisdk.ChatCompletionMessageFunctionToolCallFunctionParam{
								Name:      toolCallPart.ToolName,
								Arguments: toolCallPart.Input,
							},
						},
					}
					if cacheControl != nil {
						tc.OfFunction.SetExtraFields(map[string]any{
							"cache_control": map[string]string{
								"type": cacheControl.Type,
							},
						})
					}
					assistantMsg.ToolCalls = append(assistantMsg.ToolCalls, tc)
				}
			}
			messages = append(messages, openaisdk.ChatCompletionMessageParamUnion{
				OfAssistant: &assistantMsg,
			})

		case fantasy.MessageRoleTool:
			for i, c := range msg.Content {
				isLastPart := i == len(msg.Content)-1
				cacheControl := anthropic.GetCacheControl(c.Options())
				if cacheControl == nil && isLastPart {
					cacheControl = anthropic.GetCacheControl(msg.ProviderOptions)
				}
				if c.GetType() != fantasy.ContentTypeToolResult {
					warnings = append(warnings, fantasy.CallWarning{
						Type:    fantasy.CallWarningTypeOther,
						Message: "tool message can only have tool result content",
					})
					continue
				}
				toolResultPart, ok := fantasy.AsContentType[fantasy.ToolResultPart](c)
				if !ok {
					warnings = append(warnings, fantasy.CallWarning{
						Type:    fantasy.CallWarningTypeOther,
						Message: "tool message result part does not have the right type",
					})
					continue
				}
				switch toolResultPart.Output.GetType() {
				case fantasy.ToolResultContentTypeText:
					output, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentText](toolResultPart.Output)
					if !ok {
						warnings = append(warnings, fantasy.CallWarning{
							Type:    fantasy.CallWarningTypeOther,
							Message: "tool result output does not have the right type",
						})
						continue
					}
					tr := openaisdk.ToolMessage(output.Text, toolResultPart.ToolCallID)
					if cacheControl != nil {
						tr.SetExtraFields(map[string]any{
							"cache_control": map[string]string{
								"type": cacheControl.Type,
							},
						})
					}
					messages = append(messages, tr)
				case fantasy.ToolResultContentTypeError:
					output, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentError](toolResultPart.Output)
					if !ok {
						warnings = append(warnings, fantasy.CallWarning{
							Type:    fantasy.CallWarningTypeOther,
							Message: "tool result output does not have the right type",
						})
						continue
					}
					tr := openaisdk.ToolMessage(output.Error.Error(), toolResultPart.ToolCallID)
					if cacheControl != nil {
						tr.SetExtraFields(map[string]any{
							"cache_control": map[string]string{
								"type": cacheControl.Type,
							},
						})
					}
					messages = append(messages, tr)
				}
			}
		}
	}
	return messages, warnings
}

func hasVisibleUserContent(content []openaisdk.ChatCompletionContentPartUnionParam) bool {
	for _, part := range content {
		if part.OfText != nil || part.OfImageURL != nil || part.OfInputAudio != nil || part.OfFile != nil {
			return true
		}
	}
	return false
}

func structToMapJSON(s any) (map[string]any, error) {
	var result map[string]any
	jsonBytes, err := json.Marshal(s)
	if err != nil {
		return nil, err
	}
	err = json.Unmarshal(jsonBytes, &result)
	if err != nil {
		return nil, err
	}
	return result, nil
}
