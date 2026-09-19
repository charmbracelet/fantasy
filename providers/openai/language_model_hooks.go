package openai

import (
	"encoding/base64"
	"fmt"
	"net/http"
	"strings"

	"charm.land/fantasy"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/shared"
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

// LanguageModelHeaderFunc is a function that receives the HTTP response
// headers of a completed call. It may copy headers of interest into the
// provider metadata, so they are surfaced to callers through
// [fantasy.Response] and the terminal [fantasy.StreamPart].
type LanguageModelHeaderFunc = func(header http.Header, metadata *ProviderMetadata)

// LanguageModelToPromptFunc is a function that handles converting fantasy prompts to openai sdk messages.
type LanguageModelToPromptFunc = func(prompt fantasy.Prompt, provider, model string) ([]openai.ChatCompletionMessageParamUnion, []fantasy.CallWarning)

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
			return nil, &fantasy.Error{Title: "invalid argument", Message: "openai provider options should be *openai.ProviderOptions"}
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
		case ReasoningEffortNone:
			params.ReasoningEffort = shared.ReasoningEffortNone
		case ReasoningEffortMinimal:
			params.ReasoningEffort = shared.ReasoningEffortMinimal
		case ReasoningEffortLow:
			params.ReasoningEffort = shared.ReasoningEffortLow
		case ReasoningEffortMedium:
			params.ReasoningEffort = shared.ReasoningEffortMedium
		case ReasoningEffortHigh:
			params.ReasoningEffort = shared.ReasoningEffortHigh
		case ReasoningEffortXHigh:
			params.ReasoningEffort = shared.ReasoningEffortXhigh
		case ReasoningEffortMax:
			params.ReasoningEffort = shared.ReasoningEffortMax
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
	case "insufficient_system_resource":
		// DeepSeek-family upstreams report this when the request could not
		// be served; the partial output (including open tool calls) must be
		// treated like a truncation, never a completed turn.
		return fantasy.FinishReasonError
	default:
		return fantasy.FinishReasonUnknown
	}
}

// FoldDisjointReasoning keeps output tokens all-inclusive. OpenAI reports
// reasoning as a subset of completion_tokens, but some OpenAI-shaped gateways
// report it disjointly (Vercel AI Gateway streaming Gemini, for one).
// Reasoning exceeding output can only happen in the disjoint shape — a subset
// can never exceed its parent — so folding it in is always safe, and total
// follows output.
func FoldDisjointReasoning(output, reasoning, total int64) (int64, int64) {
	if reasoning > output {
		return output + reasoning, total + reasoning
	}
	return output, total
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
	// OpenAI reports prompt_tokens INCLUDING cached tokens. Subtract to avoid double-counting.
	inputTokens := max(response.Usage.PromptTokens-promptTokenDetails.CachedTokens, 0)
	outputTokens, totalTokens := FoldDisjointReasoning(
		response.Usage.CompletionTokens,
		completionTokenDetails.ReasoningTokens,
		response.Usage.TotalTokens,
	)
	providerMetadata.ExtraFields = ExtractExtraFields(response.Usage.JSON.ExtraFields)
	return fantasy.Usage{
		InputTokens:     inputTokens,
		OutputTokens:    outputTokens,
		TotalTokens:     totalTokens,
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
	// OpenAI reports prompt_tokens INCLUDING cached tokens. Subtract to avoid double-counting.
	inputTokens := max(chunk.Usage.PromptTokens-promptTokenDetails.CachedTokens, 0)
	outputTokens, totalTokens := FoldDisjointReasoning(
		chunk.Usage.CompletionTokens,
		completionTokenDetails.ReasoningTokens,
		chunk.Usage.TotalTokens,
	)
	usage := fantasy.Usage{
		InputTokens:     inputTokens,
		OutputTokens:    outputTokens,
		TotalTokens:     totalTokens,
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

	streamProviderMetadata.ExtraFields = ExtractExtraFields(chunk.Usage.JSON.ExtraFields)

	return usage, fantasy.ProviderMetadata{
		Name: streamProviderMetadata,
	}
}

// DefaultStreamProviderMetadataFunc is the default implementation for handling stream provider metadata.
func DefaultStreamProviderMetadataFunc(choice openai.ChatCompletionChoice, metadata fantasy.ProviderMetadata) fantasy.ProviderMetadata {
	if metadata == nil {
		metadata = fantasy.ProviderMetadata{}
	}
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

// DefaultToPrompt converts a fantasy prompt to OpenAI format with default handling.
func DefaultToPrompt(prompt fantasy.Prompt, _, _ string) ([]openai.ChatCompletionMessageParamUnion, []fantasy.CallWarning) {
	var messages []openai.ChatCompletionMessageParamUnion
	var warnings []fantasy.CallWarning
	var media ToolRunBuffer
	for _, msg := range prompt {
		messages = media.Role(msg.Role, messages)
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
			messages = append(messages, openai.SystemMessage(strings.Join(systemPromptParts, "\n")))
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
				messages = append(messages, openai.UserMessage(textPart.Text))
				continue
			}
			// text content and attachments
			// for now we only support image content later we need to check
			// TODO: add the supported media types to the language model so we
			//  can use that to validate the data here.
			var content []openai.ChatCompletionContentPartUnionParam
			for _, c := range msg.Content {
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
					content = append(content, openai.ChatCompletionContentPartUnionParam{
						OfText: &openai.ChatCompletionContentPartTextParam{
							Text: textPart.Text,
						},
					})
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
					case strings.HasPrefix(filePart.MediaType, "text/"):
						base64Encoded := base64.StdEncoding.EncodeToString(filePart.Data)
						documentBlock := openai.ChatCompletionContentPartFileFileParam{
							FileData: param.NewOpt(base64Encoded),
						}
						content = append(content, openai.FileContentPart(documentBlock))

					case strings.HasPrefix(filePart.MediaType, "image/"):
						// Handle image files
						base64Encoded := base64.StdEncoding.EncodeToString(filePart.Data)
						data := "data:" + filePart.MediaType + ";base64," + base64Encoded
						imageURL := openai.ChatCompletionContentPartImageImageURLParam{URL: data}

						// Check for provider-specific options like image detail
						if providerOptions, ok := filePart.ProviderOptions[Name]; ok {
							if detail, ok := providerOptions.(*ProviderFileOptions); ok {
								imageURL.Detail = detail.ImageDetail
							}
						}

						imageBlock := openai.ChatCompletionContentPartImageParam{ImageURL: imageURL}
						content = append(content, openai.ChatCompletionContentPartUnionParam{OfImageURL: &imageBlock})

					case filePart.MediaType == "audio/wav":
						// Handle WAV audio files
						base64Encoded := base64.StdEncoding.EncodeToString(filePart.Data)
						audioBlock := openai.ChatCompletionContentPartInputAudioParam{
							InputAudio: openai.ChatCompletionContentPartInputAudioInputAudioParam{
								Data:   base64Encoded,
								Format: "wav",
							},
						}
						content = append(content, openai.ChatCompletionContentPartUnionParam{OfInputAudio: &audioBlock})

					case filePart.MediaType == "audio/mpeg" || filePart.MediaType == "audio/mp3":
						// Handle MP3 audio files
						base64Encoded := base64.StdEncoding.EncodeToString(filePart.Data)
						audioBlock := openai.ChatCompletionContentPartInputAudioParam{
							InputAudio: openai.ChatCompletionContentPartInputAudioInputAudioParam{
								Data:   base64Encoded,
								Format: "mp3",
							},
						}
						content = append(content, openai.ChatCompletionContentPartUnionParam{OfInputAudio: &audioBlock})

					case filePart.MediaType == "application/pdf":
						// Handle PDF files
						dataStr := string(filePart.Data)

						// Check if data looks like a file ID (starts with "file-")
						if strings.HasPrefix(dataStr, "file-") {
							fileBlock := openai.ChatCompletionContentPartFileParam{
								File: openai.ChatCompletionContentPartFileFileParam{
									FileID: param.NewOpt(dataStr),
								},
							}
							content = append(content, openai.ChatCompletionContentPartUnionParam{OfFile: &fileBlock})
						} else {
							// Handle as base64 data
							base64Encoded := base64.StdEncoding.EncodeToString(filePart.Data)
							data := "data:application/pdf;base64," + base64Encoded

							filename := filePart.Filename
							if filename == "" {
								// Generate default filename based on content index
								filename = fmt.Sprintf("part-%d.pdf", len(content))
							}

							fileBlock := openai.ChatCompletionContentPartFileParam{
								File: openai.ChatCompletionContentPartFileFileParam{
									Filename: param.NewOpt(filename),
									FileData: param.NewOpt(data),
								},
							}
							content = append(content, openai.ChatCompletionContentPartUnionParam{OfFile: &fileBlock})
						}

					default:
						warnings = append(warnings, fantasy.CallWarning{
							Type:    fantasy.CallWarningTypeOther,
							Message: fmt.Sprintf("file part media type %s not supported", filePart.MediaType),
						})
					}
				}
			}
			if !HasVisibleUserContent(content) {
				warnings = append(warnings, fantasy.CallWarning{
					Type:    fantasy.CallWarningTypeOther,
					Message: "dropping empty user message (contains neither user-facing content nor tool results)",
				})
				continue
			}
			messages = append(messages, openai.UserMessage(content))
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
				messages = append(messages, openai.AssistantMessage(textPart.Text))
				continue
			}
			assistantMsg := openai.ChatCompletionAssistantMessageParam{
				Role: "assistant",
			}
			for _, c := range msg.Content {
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
					assistantMsg.Content = openai.ChatCompletionAssistantMessageParamContentUnion{
						OfString: param.NewOpt(textPart.Text),
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
					assistantMsg.ToolCalls = append(assistantMsg.ToolCalls,
						openai.ChatCompletionMessageToolCallUnionParam{
							OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
								ID:   toolCallPart.ToolCallID,
								Type: "function",
								Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
									Name:      toolCallPart.ToolName,
									Arguments: toolCallPart.Input,
								},
							},
						})
				}
			}
			if !HasVisibleAssistantContent(&assistantMsg) {
				warnings = append(warnings, fantasy.CallWarning{
					Type:    fantasy.CallWarningTypeOther,
					Message: "dropping empty assistant message (contains neither user-facing content nor tool calls)",
				})
				continue
			}
			messages = append(messages, openai.ChatCompletionMessageParamUnion{
				OfAssistant: &assistantMsg,
			})
		case fantasy.MessageRoleTool:
			for _, c := range msg.Content {
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
					messages = append(messages, openai.ToolMessage(output.Text, toolResultPart.ToolCallID))
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
					messages = append(messages, openai.ToolMessage(output.Error.Error(), toolResultPart.ToolCallID))
				case fantasy.ToolResultContentTypeMedia:
					output, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentMedia](toolResultPart.Output)
					if !ok {
						warnings = append(warnings, fantasy.CallWarning{
							Type:    fantasy.CallWarningTypeOther,
							Message: "tool result output does not have the right type",
						})
						continue
					}
					// OpenAI Chat Completions tool messages cannot carry image
					// or audio content directly; see ToolResultMediaMessages.
					toolMessage, mediaMessages, mediaWarnings := ToolResultMediaMessages(output, toolResultPart.ToolCallID)
					messages = append(messages, toolMessage)
					media.Defer(mediaMessages...)
					warnings = append(warnings, mediaWarnings...)
				default:
					warnings = append(warnings, fantasy.CallWarning{
						Type:    fantasy.CallWarningTypeOther,
						Message: fmt.Sprintf("tool result output type %q not supported", toolResultPart.Output.GetType()),
					})
				}
			}
		}
	}
	return media.Close(messages), warnings
}

// ToolResultMediaMessages maps a tool-result media output to the chat
// completions messages that convey it. OpenAI tool messages can only carry
// text, so this returns a text tool message (using any accompanying text, or a
// placeholder describing the media) to keep the tool_call/tool_result pairing
// valid, plus synthetic user messages holding the actual image or audio
// content part so vision- and audio-capable models can see it.
//
// The two are returned separately because they belong in different places: the
// tool message must stay inside the contiguous run of tool messages answering
// an assistant's tool_calls, while the media messages must land after that run
// ends. Strict validators reject a tool message that does not immediately
// follow either its assistant message or another tool message.
//
// Unsupported media types produce only the text tool message plus a warning.
// This is shared with OpenAI-compatible providers, which face the same
// constraint.
func ToolResultMediaMessages(output fantasy.ToolResultOutputContentMedia, toolCallID string) (openai.ChatCompletionMessageParamUnion, []openai.ChatCompletionMessageParamUnion, []fantasy.CallWarning) {
	placeholder := output.Text
	if placeholder == "" {
		placeholder = fmt.Sprintf("The tool returned %s content; see the following user message.", output.MediaType)
	}
	toolMessage := openai.ToolMessage(placeholder, toolCallID)

	mediaPart, warning, emit := toolResultMediaUserPart(output)
	if warning != nil {
		return toolMessage, nil, []fantasy.CallWarning{*warning}
	}
	if !emit {
		return toolMessage, nil, nil
	}
	return toolMessage, []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage([]openai.ChatCompletionContentPartUnionParam{mediaPart}),
	}, nil
}

// toolResultMediaUserPart maps a tool-result media output to an OpenAI chat
// completions user content part. It returns the content part, an optional
// warning, and whether the caller should emit the returned part.
func toolResultMediaUserPart(output fantasy.ToolResultOutputContentMedia) (openai.ChatCompletionContentPartUnionParam, *fantasy.CallWarning, bool) {
	switch {
	case strings.HasPrefix(output.MediaType, "image/"):
		data := "data:" + output.MediaType + ";base64," + output.Data
		imageBlock := openai.ChatCompletionContentPartImageParam{
			ImageURL: openai.ChatCompletionContentPartImageImageURLParam{URL: data},
		}
		return openai.ChatCompletionContentPartUnionParam{OfImageURL: &imageBlock}, nil, true
	case output.MediaType == "audio/wav", output.MediaType == "audio/mpeg", output.MediaType == "audio/mp3":
		format := "wav"
		if output.MediaType != "audio/wav" {
			format = "mp3"
		}
		audioBlock := openai.ChatCompletionContentPartInputAudioParam{
			InputAudio: openai.ChatCompletionContentPartInputAudioInputAudioParam{
				Data:   output.Data,
				Format: format,
			},
		}
		return openai.ChatCompletionContentPartUnionParam{OfInputAudio: &audioBlock}, nil, true
	default:
		return openai.ChatCompletionContentPartUnionParam{}, &fantasy.CallWarning{
			Type:    fantasy.CallWarningTypeOther,
			Message: fmt.Sprintf("tool result media type %s not supported, sending text placeholder only", output.MediaType),
		}, false
	}
}

// HasVisibleUserContent reports whether a user message carries anything the
// model can actually see. A message that converts to no visible parts must be
// dropped rather than sent: the API rejects a content-less user message, and
// the error points at the request rather than at the content that vanished.
//
// Exported because every chat-completions provider in this module needs the
// same check against the same SDK type.
func HasVisibleUserContent(content []openai.ChatCompletionContentPartUnionParam) bool {
	for _, part := range content {
		if part.OfText != nil || part.OfImageURL != nil || part.OfInputAudio != nil || part.OfFile != nil {
			return true
		}
	}
	return false
}

// HasVisibleAssistantContent reports whether an assistant message carries text
// or tool calls. Reasoning alone is not enough: the reasoning travels in
// provider-specific extra fields, and a message holding only those is empty as
// far as the API is concerned.
//
// Dropping a reasoning-only turn is safe as well as necessary. Strict
// OpenAI-compatible upstreams reject such a message outright ("content or
// tool_calls must be set"), and because the message stays in history the
// rejection repeats on every later request (charmbracelet/crush#3794). The
// DeepSeek and Kimi replay contract only needs reasoning carried on turns that
// also have content or tool calls, which pass the checks here; a bare
// reasoning turn is a truncated or canceled turn with no completion to resume
// from.
func HasVisibleAssistantContent(msg *openai.ChatCompletionAssistantMessageParam) bool {
	// Check if there's text content
	if !param.IsOmitted(msg.Content.OfString) || len(msg.Content.OfArrayOfContentParts) > 0 {
		return true
	}
	// Check if there are tool calls
	if len(msg.ToolCalls) > 0 {
		return true
	}
	return false
}

// TagToolCacheControl marks a tool message for Anthropic-style prompt caching.
//
// It takes the tool param rather than the message union that wraps it because
// the union is not the value that gets marshalled: setting the extra field
// there compiles, reads correctly, and silently drops the hint, so every
// request goes out uncached with nothing to show for it. Taking the inner
// value makes that mistake unspellable.
//
// Exported because openrouter and vercel both attach the same hint to the same
// SDK type.
func TagToolCacheControl(msg *openai.ChatCompletionToolMessageParam, cacheType string) {
	if msg == nil || cacheType == "" {
		return
	}
	msg.SetExtraFields(map[string]any{
		"cache_control": map[string]string{"type": cacheType},
	})
}

// ToolRunBuffer holds synthetic user messages carrying tool-result media back
// until the current run of tool messages ends.
//
// Chat-completions validators require every tool message answering an
// assistant's tool_calls to follow that assistant message with only other tool
// messages in between, so a media result cannot emit its user message where it
// is produced: doing so splits the run and every tool_call_id after the split
// is reported unanswered. See ToolResultMediaMessages.
//
// The rule lives here rather than in each converter because all four
// chat-completions providers in this module hit it identically, and a rule
// written four times is a rule that drifts.
type ToolRunBuffer struct {
	deferred []openai.ChatCompletionMessageParamUnion
}

// Role reports that a prompt message of the given role is about to be
// converted, flushing any deferred media when that role ends a tool run.
// It returns messages with the flush applied.
func (b *ToolRunBuffer) Role(role fantasy.MessageRole, messages []openai.ChatCompletionMessageParamUnion) []openai.ChatCompletionMessageParamUnion {
	if role == fantasy.MessageRoleTool || len(b.deferred) == 0 {
		return messages
	}
	messages = append(messages, b.deferred...)
	b.deferred = nil
	return messages
}

// Defer holds media messages until the current tool run ends.
func (b *ToolRunBuffer) Defer(messages ...openai.ChatCompletionMessageParamUnion) {
	b.deferred = append(b.deferred, messages...)
}

// Close flushes anything still deferred when the prompt ends on a tool run.
func (b *ToolRunBuffer) Close(messages []openai.ChatCompletionMessageParamUnion) []openai.ChatCompletionMessageParamUnion {
	return append(messages, b.deferred...)
}
