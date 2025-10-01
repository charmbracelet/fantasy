package openai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/charmbracelet/fantasy/ai"
	xjson "github.com/charmbracelet/x/json"
	"github.com/google/uuid"
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/packages/param"
	"github.com/openai/openai-go/v2/shared"
)

type languageModel struct {
	provider                   string
	modelID                    string
	client                     openai.Client
	uniqueToolCallIds          bool
	generateIDFunc             LanguageModelGenerateIDFunc
	prepareCallFunc            LanguageModelPrepareCallFunc
	mapFinishReasonFunc        LanguageModelMapFinishReasonFunc
	extraContentFunc           LanguageModelExtraContentFunc
	usageFunc                  LanguageModelUsageFunc
	streamUsageFunc            LanguageModelStreamUsageFunc
	streamExtraFunc            LanguageModelStreamExtraFunc
	streamProviderMetadataFunc LanguageModelStreamProviderMetadataFunc
}

type LanguageModelOption = func(*languageModel)

func WithLanguageModelPrepareCallFunc(fn LanguageModelPrepareCallFunc) LanguageModelOption {
	return func(l *languageModel) {
		l.prepareCallFunc = fn
	}
}

func WithLanguageModelMapFinishReasonFunc(fn LanguageModelMapFinishReasonFunc) LanguageModelOption {
	return func(l *languageModel) {
		l.mapFinishReasonFunc = fn
	}
}

func WithLanguageModelExtraContentFunc(fn LanguageModelExtraContentFunc) LanguageModelOption {
	return func(l *languageModel) {
		l.extraContentFunc = fn
	}
}

func WithLanguageModelStreamExtraFunc(fn LanguageModelStreamExtraFunc) LanguageModelOption {
	return func(l *languageModel) {
		l.streamExtraFunc = fn
	}
}

func WithLanguageModelUsageFunc(fn LanguageModelUsageFunc) LanguageModelOption {
	return func(l *languageModel) {
		l.usageFunc = fn
	}
}

func WithLanguageModelStreamUsageFunc(fn LanguageModelStreamUsageFunc) LanguageModelOption {
	return func(l *languageModel) {
		l.streamUsageFunc = fn
	}
}

func WithLanguageUniqueToolCallIds() LanguageModelOption {
	return func(l *languageModel) {
		l.uniqueToolCallIds = true
	}
}

func WithLanguageModelGenerateIDFunc(fn LanguageModelGenerateIDFunc) LanguageModelOption {
	return func(l *languageModel) {
		l.generateIDFunc = fn
	}
}

func newLanguageModel(modelID string, provider string, client openai.Client, opts ...LanguageModelOption) languageModel {
	model := languageModel{
		modelID:                    modelID,
		provider:                   provider,
		client:                     client,
		generateIDFunc:             DefaultGenerateID,
		prepareCallFunc:            DefaultPrepareCallFunc,
		mapFinishReasonFunc:        DefaultMapFinishReasonFunc,
		usageFunc:                  DefaultUsageFunc,
		streamUsageFunc:            DefaultStreamUsageFunc,
		streamProviderMetadataFunc: DefaultStreamProviderMetadataFunc,
	}

	for _, o := range opts {
		o(&model)
	}
	return model
}

type streamToolCall struct {
	id          string
	name        string
	arguments   string
	hasFinished bool
}

// Model implements ai.LanguageModel.
func (o languageModel) Model() string {
	return o.modelID
}

// Provider implements ai.LanguageModel.
func (o languageModel) Provider() string {
	return o.provider
}

func (o languageModel) prepareParams(call ai.Call) (*openai.ChatCompletionNewParams, []ai.CallWarning, error) {
	params := &openai.ChatCompletionNewParams{}
	messages, warnings := toPrompt(call.Prompt)
	if call.TopK != nil {
		warnings = append(warnings, ai.CallWarning{
			Type:    ai.CallWarningTypeUnsupportedSetting,
			Setting: "top_k",
		})
	}

	if call.MaxOutputTokens != nil {
		params.MaxTokens = param.NewOpt(*call.MaxOutputTokens)
	}
	if call.Temperature != nil {
		params.Temperature = param.NewOpt(*call.Temperature)
	}
	if call.TopP != nil {
		params.TopP = param.NewOpt(*call.TopP)
	}
	if call.FrequencyPenalty != nil {
		params.FrequencyPenalty = param.NewOpt(*call.FrequencyPenalty)
	}
	if call.PresencePenalty != nil {
		params.PresencePenalty = param.NewOpt(*call.PresencePenalty)
	}

	if isReasoningModel(o.modelID) {
		// remove unsupported settings for reasoning models
		// see https://platform.openai.com/docs/guides/reasoning#limitations
		if call.Temperature != nil {
			params.Temperature = param.Opt[float64]{}
			warnings = append(warnings, ai.CallWarning{
				Type:    ai.CallWarningTypeUnsupportedSetting,
				Setting: "temperature",
				Details: "temperature is not supported for reasoning models",
			})
		}
		if call.TopP != nil {
			params.TopP = param.Opt[float64]{}
			warnings = append(warnings, ai.CallWarning{
				Type:    ai.CallWarningTypeUnsupportedSetting,
				Setting: "TopP",
				Details: "TopP is not supported for reasoning models",
			})
		}
		if call.FrequencyPenalty != nil {
			params.FrequencyPenalty = param.Opt[float64]{}
			warnings = append(warnings, ai.CallWarning{
				Type:    ai.CallWarningTypeUnsupportedSetting,
				Setting: "FrequencyPenalty",
				Details: "FrequencyPenalty is not supported for reasoning models",
			})
		}
		if call.PresencePenalty != nil {
			params.PresencePenalty = param.Opt[float64]{}
			warnings = append(warnings, ai.CallWarning{
				Type:    ai.CallWarningTypeUnsupportedSetting,
				Setting: "PresencePenalty",
				Details: "PresencePenalty is not supported for reasoning models",
			})
		}

		// reasoning models use max_completion_tokens instead of max_tokens
		if call.MaxOutputTokens != nil {
			if !params.MaxCompletionTokens.Valid() {
				params.MaxCompletionTokens = param.NewOpt(*call.MaxOutputTokens)
			}
			params.MaxTokens = param.Opt[int64]{}
		}
	}

	// Handle search preview models
	if isSearchPreviewModel(o.modelID) {
		if call.Temperature != nil {
			params.Temperature = param.Opt[float64]{}
			warnings = append(warnings, ai.CallWarning{
				Type:    ai.CallWarningTypeUnsupportedSetting,
				Setting: "temperature",
				Details: "temperature is not supported for the search preview models and has been removed.",
			})
		}
	}

	optionsWarnings, err := o.prepareCallFunc(o, params, call)
	if err != nil {
		return nil, nil, err
	}

	if len(optionsWarnings) > 0 {
		warnings = append(warnings, optionsWarnings...)
	}

	params.Messages = messages
	params.Model = o.modelID

	if len(call.Tools) > 0 {
		tools, toolChoice, toolWarnings := toOpenAiTools(call.Tools, call.ToolChoice)
		params.Tools = tools
		if toolChoice != nil {
			params.ToolChoice = *toolChoice
		}
		warnings = append(warnings, toolWarnings...)
	}
	return params, warnings, nil
}

func (o languageModel) handleError(err error) error {
	var apiErr *openai.Error
	if errors.As(err, &apiErr) {
		requestDump := apiErr.DumpRequest(true)
		responseDump := apiErr.DumpResponse(true)
		headers := map[string]string{}
		for k, h := range apiErr.Response.Header {
			v := h[len(h)-1]
			headers[strings.ToLower(k)] = v
		}
		return ai.NewAPICallError(
			apiErr.Message,
			apiErr.Request.URL.String(),
			string(requestDump),
			apiErr.StatusCode,
			headers,
			string(responseDump),
			apiErr,
			false,
		)
	}
	return err
}

// Generate implements ai.LanguageModel.
func (o languageModel) Generate(ctx context.Context, call ai.Call) (*ai.Response, error) {
	params, warnings, err := o.prepareParams(call)
	if err != nil {
		return nil, err
	}
	response, err := o.client.Chat.Completions.New(ctx, *params)
	if err != nil {
		return nil, o.handleError(err)
	}

	if len(response.Choices) == 0 {
		return nil, errors.New("no response generated")
	}
	choice := response.Choices[0]
	content := make([]ai.Content, 0, 1+len(choice.Message.ToolCalls)+len(choice.Message.Annotations))
	text := choice.Message.Content
	if text != "" {
		content = append(content, ai.TextContent{
			Text: text,
		})
	}
	if o.extraContentFunc != nil {
		extraContent := o.extraContentFunc(choice)
		content = append(content, extraContent...)
	}
	for _, tc := range choice.Message.ToolCalls {
		toolCallID := tc.ID
		if toolCallID == "" || o.uniqueToolCallIds {
			toolCallID = o.generateIDFunc()
		}
		content = append(content, ai.ToolCallContent{
			ProviderExecuted: false, // TODO: update when handling other tools
			ToolCallID:       toolCallID,
			ToolName:         tc.Function.Name,
			Input:            tc.Function.Arguments,
		})
	}
	// Handle annotations/citations
	for _, annotation := range choice.Message.Annotations {
		if annotation.Type == "url_citation" {
			content = append(content, ai.SourceContent{
				SourceType: ai.SourceTypeURL,
				ID:         uuid.NewString(),
				URL:        annotation.URLCitation.URL,
				Title:      annotation.URLCitation.Title,
			})
		}
	}

	usage, providerMetadata := o.usageFunc(*response)

	return &ai.Response{
		Content:      content,
		Usage:        usage,
		FinishReason: DefaultMapFinishReasonFunc(choice),
		ProviderMetadata: ai.ProviderMetadata{
			Name: providerMetadata,
		},
		Warnings: warnings,
	}, nil
}

// Stream implements ai.LanguageModel.
func (o languageModel) Stream(ctx context.Context, call ai.Call) (ai.StreamResponse, error) {
	params, warnings, err := o.prepareParams(call)
	if err != nil {
		return nil, err
	}

	params.StreamOptions = openai.ChatCompletionStreamOptionsParam{
		IncludeUsage: openai.Bool(true),
	}

	stream := o.client.Chat.Completions.NewStreaming(ctx, *params)
	isActiveText := false
	toolCalls := make(map[int64]streamToolCall)

	// Build provider metadata for streaming
	providerMetadata := ai.ProviderMetadata{
		Name: &ProviderMetadata{},
	}
	acc := openai.ChatCompletionAccumulator{}
	extraContext := make(map[string]any)
	var usage ai.Usage
	return func(yield func(ai.StreamPart) bool) {
		if len(warnings) > 0 {
			if !yield(ai.StreamPart{
				Type:     ai.StreamPartTypeWarnings,
				Warnings: warnings,
			}) {
				return
			}
		}
		for stream.Next() {
			chunk := stream.Current()
			acc.AddChunk(chunk)
			usage, providerMetadata = o.streamUsageFunc(chunk, extraContext, providerMetadata)
			if len(chunk.Choices) == 0 {
				continue
			}
			for _, choice := range chunk.Choices {
				switch {
				case choice.Delta.Content != "":
					if !isActiveText {
						isActiveText = true
						if !yield(ai.StreamPart{
							Type: ai.StreamPartTypeTextStart,
							ID:   "0",
						}) {
							return
						}
					}
					if !yield(ai.StreamPart{
						Type:  ai.StreamPartTypeTextDelta,
						ID:    "0",
						Delta: choice.Delta.Content,
					}) {
						return
					}
				case len(choice.Delta.ToolCalls) > 0:
					if isActiveText {
						isActiveText = false
						if !yield(ai.StreamPart{
							Type: ai.StreamPartTypeTextEnd,
							ID:   "0",
						}) {
							return
						}
					}

					for _, toolCallDelta := range choice.Delta.ToolCalls {
						if existingToolCall, ok := toolCalls[toolCallDelta.Index]; ok {
							if existingToolCall.hasFinished {
								continue
							}
							if toolCallDelta.Function.Arguments != "" {
								existingToolCall.arguments += toolCallDelta.Function.Arguments
							}
							if !yield(ai.StreamPart{
								Type:  ai.StreamPartTypeToolInputDelta,
								ID:    existingToolCall.id,
								Delta: toolCallDelta.Function.Arguments,
							}) {
								return
							}
							toolCalls[toolCallDelta.Index] = existingToolCall
							if xjson.IsValid(existingToolCall.arguments) {
								if !yield(ai.StreamPart{
									Type: ai.StreamPartTypeToolInputEnd,
									ID:   existingToolCall.id,
								}) {
									return
								}

								if !yield(ai.StreamPart{
									Type:          ai.StreamPartTypeToolCall,
									ID:            existingToolCall.id,
									ToolCallName:  existingToolCall.name,
									ToolCallInput: existingToolCall.arguments,
								}) {
									return
								}
								existingToolCall.hasFinished = true
								toolCalls[toolCallDelta.Index] = existingToolCall
							}
						} else {
							// Does not exist
							var err error
							if toolCallDelta.Type != "function" {
								err = ai.NewInvalidResponseDataError(toolCallDelta, "Expected 'function' type.")
							}
							if toolCallDelta.ID == "" {
								err = ai.NewInvalidResponseDataError(toolCallDelta, "Expected 'id' to be a string.")
							}
							if toolCallDelta.Function.Name == "" {
								err = ai.NewInvalidResponseDataError(toolCallDelta, "Expected 'function.name' to be a string.")
							}
							if err != nil {
								yield(ai.StreamPart{
									Type:  ai.StreamPartTypeError,
									Error: o.handleError(stream.Err()),
								})
								return
							}

							// some providers do not send this as a unique id
							// for some usecases in crush we need this ID to be unique.
							// it won't change the behavior on the provider side because the
							// provider only cares about the tool call id matching the result
							// and in our case that will still be the case
							if o.uniqueToolCallIds {
								toolCallDelta.ID = o.generateIDFunc()
							}

							if !yield(ai.StreamPart{
								Type:         ai.StreamPartTypeToolInputStart,
								ID:           toolCallDelta.ID,
								ToolCallName: toolCallDelta.Function.Name,
							}) {
								return
							}
							toolCalls[toolCallDelta.Index] = streamToolCall{
								id:        toolCallDelta.ID,
								name:      toolCallDelta.Function.Name,
								arguments: toolCallDelta.Function.Arguments,
							}

							exTc := toolCalls[toolCallDelta.Index]
							if exTc.arguments != "" {
								if !yield(ai.StreamPart{
									Type:  ai.StreamPartTypeToolInputDelta,
									ID:    exTc.id,
									Delta: exTc.arguments,
								}) {
									return
								}
								if xjson.IsValid(toolCalls[toolCallDelta.Index].arguments) {
									if !yield(ai.StreamPart{
										Type: ai.StreamPartTypeToolInputEnd,
										ID:   toolCallDelta.ID,
									}) {
										return
									}

									if !yield(ai.StreamPart{
										Type:          ai.StreamPartTypeToolCall,
										ID:            exTc.id,
										ToolCallName:  exTc.name,
										ToolCallInput: exTc.arguments,
									}) {
										return
									}
									exTc.hasFinished = true
									toolCalls[toolCallDelta.Index] = exTc
								}
							}
							continue
						}
					}
				}

				if o.streamExtraFunc != nil {
					updatedContext, shouldContinue := o.streamExtraFunc(chunk, yield, extraContext)
					if !shouldContinue {
						return
					}
					extraContext = updatedContext
				}
			}

			// Check for annotations in the delta's raw JSON
			for _, choice := range chunk.Choices {
				if annotations := parseAnnotationsFromDelta(choice.Delta); len(annotations) > 0 {
					for _, annotation := range annotations {
						if annotation.Type == "url_citation" {
							if !yield(ai.StreamPart{
								Type:       ai.StreamPartTypeSource,
								ID:         uuid.NewString(),
								SourceType: ai.SourceTypeURL,
								URL:        annotation.URLCitation.URL,
								Title:      annotation.URLCitation.Title,
							}) {
								return
							}
						}
					}
				}
			}
		}
		err := stream.Err()
		if err == nil || errors.Is(err, io.EOF) {
			// finished
			if isActiveText {
				isActiveText = false
				if !yield(ai.StreamPart{
					Type: ai.StreamPartTypeTextEnd,
					ID:   "0",
				}) {
					return
				}
			}

			if len(acc.Choices) > 0 {
				choice := acc.Choices[0]
				// Add logprobs if available
				providerMetadata = o.streamProviderMetadataFunc(choice, providerMetadata)

				// Handle annotations/citations from accumulated response
				for _, annotation := range choice.Message.Annotations {
					if annotation.Type == "url_citation" {
						if !yield(ai.StreamPart{
							Type:       ai.StreamPartTypeSource,
							ID:         acc.ID,
							SourceType: ai.SourceTypeURL,
							URL:        annotation.URLCitation.URL,
							Title:      annotation.URLCitation.Title,
						}) {
							return
						}
					}
				}
			}
			finishReason := ai.FinishReasonUnknown
			if len(acc.Choices) > 0 {
				finishReason = o.mapFinishReasonFunc(acc.Choices[0])
			}
			yield(ai.StreamPart{
				Type:             ai.StreamPartTypeFinish,
				Usage:            usage,
				FinishReason:     finishReason,
				ProviderMetadata: providerMetadata,
			})
			return
		} else {
			yield(ai.StreamPart{
				Type:  ai.StreamPartTypeError,
				Error: o.handleError(err),
			})
			return
		}
	}, nil
}

func isReasoningModel(modelID string) bool {
	return strings.HasPrefix(modelID, "o") || strings.HasPrefix(modelID, "gpt-5") || strings.HasPrefix(modelID, "gpt-5-chat")
}

func isSearchPreviewModel(modelID string) bool {
	return strings.Contains(modelID, "search-preview")
}

func supportsFlexProcessing(modelID string) bool {
	return strings.HasPrefix(modelID, "o3") || strings.HasPrefix(modelID, "o4-mini") || strings.HasPrefix(modelID, "gpt-5")
}

func supportsPriorityProcessing(modelID string) bool {
	return strings.HasPrefix(modelID, "gpt-4") || strings.HasPrefix(modelID, "gpt-5") ||
		strings.HasPrefix(modelID, "gpt-5-mini") || strings.HasPrefix(modelID, "o3") ||
		strings.HasPrefix(modelID, "o4-mini")
}

func toOpenAiTools(tools []ai.Tool, toolChoice *ai.ToolChoice) (openAiTools []openai.ChatCompletionToolUnionParam, openAiToolChoice *openai.ChatCompletionToolChoiceOptionUnionParam, warnings []ai.CallWarning) {
	for _, tool := range tools {
		if tool.GetType() == ai.ToolTypeFunction {
			ft, ok := tool.(ai.FunctionTool)
			if !ok {
				continue
			}
			openAiTools = append(openAiTools, openai.ChatCompletionToolUnionParam{
				OfFunction: &openai.ChatCompletionFunctionToolParam{
					Function: shared.FunctionDefinitionParam{
						Name:        ft.Name,
						Description: param.NewOpt(ft.Description),
						Parameters:  openai.FunctionParameters(ft.InputSchema),
						Strict:      param.NewOpt(false),
					},
					Type: "function",
				},
			})
			continue
		}

		// TODO: handle provider tool calls
		warnings = append(warnings, ai.CallWarning{
			Type:    ai.CallWarningTypeUnsupportedTool,
			Tool:    tool,
			Message: "tool is not supported",
		})
	}
	if toolChoice == nil {
		return openAiTools, openAiToolChoice, warnings
	}

	switch *toolChoice {
	case ai.ToolChoiceAuto:
		openAiToolChoice = &openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: param.NewOpt("auto"),
		}
	case ai.ToolChoiceNone:
		openAiToolChoice = &openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: param.NewOpt("none"),
		}
	default:
		openAiToolChoice = &openai.ChatCompletionToolChoiceOptionUnionParam{
			OfFunctionToolChoice: &openai.ChatCompletionNamedToolChoiceParam{
				Type: "function",
				Function: openai.ChatCompletionNamedToolChoiceFunctionParam{
					Name: string(*toolChoice),
				},
			},
		}
	}
	return openAiTools, openAiToolChoice, warnings
}

func toPrompt(prompt ai.Prompt) ([]openai.ChatCompletionMessageParamUnion, []ai.CallWarning) {
	var messages []openai.ChatCompletionMessageParamUnion
	var warnings []ai.CallWarning
	for _, msg := range prompt {
		switch msg.Role {
		case ai.MessageRoleSystem:
			var systemPromptParts []string
			for _, c := range msg.Content {
				if c.GetType() != ai.ContentTypeText {
					warnings = append(warnings, ai.CallWarning{
						Type:    ai.CallWarningTypeOther,
						Message: "system prompt can only have text content",
					})
					continue
				}
				textPart, ok := ai.AsContentType[ai.TextPart](c)
				if !ok {
					warnings = append(warnings, ai.CallWarning{
						Type:    ai.CallWarningTypeOther,
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
				warnings = append(warnings, ai.CallWarning{
					Type:    ai.CallWarningTypeOther,
					Message: "system prompt has no text parts",
				})
				continue
			}
			messages = append(messages, openai.SystemMessage(strings.Join(systemPromptParts, "\n")))
		case ai.MessageRoleUser:
			// simple user message just text content
			if len(msg.Content) == 1 && msg.Content[0].GetType() == ai.ContentTypeText {
				textPart, ok := ai.AsContentType[ai.TextPart](msg.Content[0])
				if !ok {
					warnings = append(warnings, ai.CallWarning{
						Type:    ai.CallWarningTypeOther,
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
				case ai.ContentTypeText:
					textPart, ok := ai.AsContentType[ai.TextPart](c)
					if !ok {
						warnings = append(warnings, ai.CallWarning{
							Type:    ai.CallWarningTypeOther,
							Message: "user message text part does not have the right type",
						})
						continue
					}
					content = append(content, openai.ChatCompletionContentPartUnionParam{
						OfText: &openai.ChatCompletionContentPartTextParam{
							Text: textPart.Text,
						},
					})
				case ai.ContentTypeFile:
					filePart, ok := ai.AsContentType[ai.FilePart](c)
					if !ok {
						warnings = append(warnings, ai.CallWarning{
							Type:    ai.CallWarningTypeOther,
							Message: "user message file part does not have the right type",
						})
						continue
					}

					switch {
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
						warnings = append(warnings, ai.CallWarning{
							Type:    ai.CallWarningTypeOther,
							Message: fmt.Sprintf("file part media type %s not supported", filePart.MediaType),
						})
					}
				}
			}
			messages = append(messages, openai.UserMessage(content))
		case ai.MessageRoleAssistant:
			// simple assistant message just text content
			if len(msg.Content) == 1 && msg.Content[0].GetType() == ai.ContentTypeText {
				textPart, ok := ai.AsContentType[ai.TextPart](msg.Content[0])
				if !ok {
					warnings = append(warnings, ai.CallWarning{
						Type:    ai.CallWarningTypeOther,
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
				case ai.ContentTypeText:
					textPart, ok := ai.AsContentType[ai.TextPart](c)
					if !ok {
						warnings = append(warnings, ai.CallWarning{
							Type:    ai.CallWarningTypeOther,
							Message: "assistant message text part does not have the right type",
						})
						continue
					}
					assistantMsg.Content = openai.ChatCompletionAssistantMessageParamContentUnion{
						OfString: param.NewOpt(textPart.Text),
					}
				case ai.ContentTypeToolCall:
					toolCallPart, ok := ai.AsContentType[ai.ToolCallPart](c)
					if !ok {
						warnings = append(warnings, ai.CallWarning{
							Type:    ai.CallWarningTypeOther,
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
			messages = append(messages, openai.ChatCompletionMessageParamUnion{
				OfAssistant: &assistantMsg,
			})
		case ai.MessageRoleTool:
			for _, c := range msg.Content {
				if c.GetType() != ai.ContentTypeToolResult {
					warnings = append(warnings, ai.CallWarning{
						Type:    ai.CallWarningTypeOther,
						Message: "tool message can only have tool result content",
					})
					continue
				}

				toolResultPart, ok := ai.AsContentType[ai.ToolResultPart](c)
				if !ok {
					warnings = append(warnings, ai.CallWarning{
						Type:    ai.CallWarningTypeOther,
						Message: "tool message result part does not have the right type",
					})
					continue
				}

				switch toolResultPart.Output.GetType() {
				case ai.ToolResultContentTypeText:
					output, ok := ai.AsToolResultOutputType[ai.ToolResultOutputContentText](toolResultPart.Output)
					if !ok {
						warnings = append(warnings, ai.CallWarning{
							Type:    ai.CallWarningTypeOther,
							Message: "tool result output does not have the right type",
						})
						continue
					}
					messages = append(messages, openai.ToolMessage(output.Text, toolResultPart.ToolCallID))
				case ai.ToolResultContentTypeError:
					// TODO: check if better handling is needed
					output, ok := ai.AsToolResultOutputType[ai.ToolResultOutputContentError](toolResultPart.Output)
					if !ok {
						warnings = append(warnings, ai.CallWarning{
							Type:    ai.CallWarningTypeOther,
							Message: "tool result output does not have the right type",
						})
						continue
					}
					messages = append(messages, openai.ToolMessage(output.Error.Error(), toolResultPart.ToolCallID))
				}
			}
		}
	}
	return messages, warnings
}

// parseAnnotationsFromDelta parses annotations from the raw JSON of a delta.
func parseAnnotationsFromDelta(delta openai.ChatCompletionChunkChoiceDelta) []openai.ChatCompletionMessageAnnotation {
	var annotations []openai.ChatCompletionMessageAnnotation

	// Parse the raw JSON to extract annotations
	var deltaData map[string]any
	if err := json.Unmarshal([]byte(delta.RawJSON()), &deltaData); err != nil {
		return annotations
	}

	// Check if annotations exist in the delta
	if annotationsData, ok := deltaData["annotations"].([]any); ok {
		for _, annotationData := range annotationsData {
			if annotationMap, ok := annotationData.(map[string]any); ok {
				if annotationType, ok := annotationMap["type"].(string); ok && annotationType == "url_citation" {
					if urlCitationData, ok := annotationMap["url_citation"].(map[string]any); ok {
						annotation := openai.ChatCompletionMessageAnnotation{
							Type: "url_citation",
							URLCitation: openai.ChatCompletionMessageAnnotationURLCitation{
								URL:   urlCitationData["url"].(string),
								Title: urlCitationData["title"].(string),
							},
						}
						annotations = append(annotations, annotation)
					}
				}
			}
		}
	}

	return annotations
}
