package openrouter

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"maps"
	"strings"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/anthropic"
	"charm.land/fantasy/providers/openai"
	openaisdk "github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/packages/param"
)

const reasoningStartedCtx = "reasoning_started"

func languagePrepareModelCall(_ fantasy.LanguageModel, params *openaisdk.ChatCompletionNewParams, call fantasy.Call) ([]fantasy.CallWarning, error) {
	providerOptions := &ProviderOptions{}
	if v, ok := call.ProviderOptions[Name]; ok {
		providerOptions, ok = v.(*ProviderOptions)
		if !ok {
			return nil, fantasy.NewInvalidArgumentError("providerOptions", "openrouter provider options should be *openrouter.ProviderOptions", nil)
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

func languageModelExtraContent(choice openaisdk.ChatCompletionChoice) []fantasy.Content {
	var content []fantasy.Content
	reasoningData := ReasoningData{}
	err := json.Unmarshal([]byte(choice.Message.RawJSON()), &reasoningData)
	if err != nil {
		return content
	}
	for _, detail := range reasoningData.ReasoningDetails {
		var metadata fantasy.ProviderMetadata

		if detail.Signature != "" {
			metadata = fantasy.ProviderMetadata{
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
			content = append(content, fantasy.ReasoningContent{
				Text:             detail.Text,
				ProviderMetadata: metadata,
			})
		case "reasoning.summary":
			content = append(content, fantasy.ReasoningContent{
				Text:             detail.Summary,
				ProviderMetadata: metadata,
			})
		case "reasoning.encrypted":
			content = append(content, fantasy.ReasoningContent{
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

func languageModelStreamExtra(chunk openaisdk.ChatCompletionChunk, yield func(fantasy.StreamPart) bool, ctx map[string]any) (map[string]any, bool) {
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
				Error: fantasy.NewAIError("Unexpected", "error unmarshalling delta", err),
			})
			return ctx, false
		}

		emitEvent := func(reasoningContent string, signature string) bool {
			if !reasoningStarted {
				shouldContinue := yield(fantasy.StreamPart{
					Type: fantasy.StreamPartTypeReasoningStart,
					ID:   fmt.Sprintf("%d", inx),
				})
				if !shouldContinue {
					return false
				}
			}

			var metadata fantasy.ProviderMetadata

			if signature != "" {
				metadata = fantasy.ProviderMetadata{
					Name: &ReasoningMetadata{
						Signature: signature,
					},
					anthropic.Name: &anthropic.ReasoningOptionMetadata{
						Signature: signature,
					},
				}
			}

			return yield(fantasy.StreamPart{
				Type:             fantasy.StreamPartTypeReasoningDelta,
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
			return ctx, yield(fantasy.StreamPart{
				Type: fantasy.StreamPartTypeReasoningEnd,
				ID:   fmt.Sprintf("%d", inx),
			})
		}
	}
	return ctx, true
}

func languageModelUsage(response openaisdk.ChatCompletion) (fantasy.Usage, fantasy.ProviderOptionsData) {
	if len(response.Choices) == 0 {
		return fantasy.Usage{}, nil
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
	openrouterUsage := UsageAccounting{}
	_ = json.Unmarshal([]byte(usage.RawJSON()), &openrouterUsage)
	streamProviderMetadata.Usage = openrouterUsage

	if p, ok := chunk.JSON.ExtraFields["provider"]; ok {
		streamProviderMetadata.Provider = p.Raw()
	}

	// we do this here because the acc does not add prompt details
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

func languageModelToPrompt(prompt fantasy.Prompt) ([]openaisdk.ChatCompletionMessageParamUnion, []fantasy.CallWarning) {
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
			// simple user message just text content
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
			// text content and attachments
			// for now we only support image content later we need to check
			// TODO: add the supported media types to the language model so we
			//  can use that to validate the data here.
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
						// Handle image files
						base64Encoded := base64.StdEncoding.EncodeToString(filePart.Data)
						data := "data:" + filePart.MediaType + ";base64," + base64Encoded
						imageURL := openaisdk.ChatCompletionContentPartImageImageURLParam{URL: data}

						// Check for provider-specific options like image detail
						if providerOptions, ok := filePart.ProviderOptions[Name]; ok {
							if detail, ok := providerOptions.(*openai.ProviderFileOptions); ok {
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
						// Handle WAV audio files
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
						// Handle MP3 audio files
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
						// Handle PDF files
						dataStr := string(filePart.Data)

						// Check if data looks like a file ID (starts with "file-")
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
							// Handle as base64 data
							base64Encoded := base64.StdEncoding.EncodeToString(filePart.Data)
							data := "data:application/pdf;base64," + base64Encoded

							filename := filePart.Filename
							if filename == "" {
								// Generate default filename based on content index
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
			messages = append(messages, openaisdk.UserMessage(content))
		case fantasy.MessageRoleAssistant:
			// simple assistant message just text content
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
					// TODO: check if better handling is needed
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
