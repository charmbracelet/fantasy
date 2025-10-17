package openai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/charmbracelet/fantasy/ai"
	"github.com/google/uuid"
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/packages/param"
	"github.com/openai/openai-go/v2/responses"
	"github.com/openai/openai-go/v2/shared"
)

const topLogprobsMax = 20

type responsesLanguageModel struct {
	provider string
	modelID  string
	client   openai.Client
}

// newResponsesLanguageModel implements a responses api model
// INFO: (kujtim) currently we do not support stored parameter we default it to false.
func newResponsesLanguageModel(modelID string, provider string, client openai.Client) responsesLanguageModel {
	return responsesLanguageModel{
		modelID:  modelID,
		provider: provider,
		client:   client,
	}
}

func (o responsesLanguageModel) Model() string {
	return o.modelID
}

func (o responsesLanguageModel) Provider() string {
	return o.provider
}

type responsesModelConfig struct {
	isReasoningModel           bool
	systemMessageMode          string
	requiredAutoTruncation     bool
	supportsFlexProcessing     bool
	supportsPriorityProcessing bool
}

func getResponsesModelConfig(modelID string) responsesModelConfig {
	supportsFlexProcessing := strings.HasPrefix(modelID, "o3") ||
		strings.HasPrefix(modelID, "o4-mini") ||
		(strings.HasPrefix(modelID, "gpt-5") && !strings.HasPrefix(modelID, "gpt-5-chat"))

	supportsPriorityProcessing := strings.HasPrefix(modelID, "gpt-4") ||
		strings.HasPrefix(modelID, "gpt-5-mini") ||
		(strings.HasPrefix(modelID, "gpt-5") &&
			!strings.HasPrefix(modelID, "gpt-5-nano") &&
			!strings.HasPrefix(modelID, "gpt-5-chat")) ||
		strings.HasPrefix(modelID, "o3") ||
		strings.HasPrefix(modelID, "o4-mini")

	defaults := responsesModelConfig{
		requiredAutoTruncation:     false,
		systemMessageMode:          "system",
		supportsFlexProcessing:     supportsFlexProcessing,
		supportsPriorityProcessing: supportsPriorityProcessing,
	}

	if strings.HasPrefix(modelID, "gpt-5-chat") {
		return responsesModelConfig{
			isReasoningModel:           false,
			systemMessageMode:          defaults.systemMessageMode,
			requiredAutoTruncation:     defaults.requiredAutoTruncation,
			supportsFlexProcessing:     defaults.supportsFlexProcessing,
			supportsPriorityProcessing: defaults.supportsPriorityProcessing,
		}
	}

	if strings.HasPrefix(modelID, "o") ||
		strings.HasPrefix(modelID, "gpt-5") ||
		strings.HasPrefix(modelID, "codex-") ||
		strings.HasPrefix(modelID, "computer-use") {
		if strings.HasPrefix(modelID, "o1-mini") || strings.HasPrefix(modelID, "o1-preview") {
			return responsesModelConfig{
				isReasoningModel:           true,
				systemMessageMode:          "remove",
				requiredAutoTruncation:     defaults.requiredAutoTruncation,
				supportsFlexProcessing:     defaults.supportsFlexProcessing,
				supportsPriorityProcessing: defaults.supportsPriorityProcessing,
			}
		}

		return responsesModelConfig{
			isReasoningModel:           true,
			systemMessageMode:          "developer",
			requiredAutoTruncation:     defaults.requiredAutoTruncation,
			supportsFlexProcessing:     defaults.supportsFlexProcessing,
			supportsPriorityProcessing: defaults.supportsPriorityProcessing,
		}
	}

	return responsesModelConfig{
		isReasoningModel:           false,
		systemMessageMode:          defaults.systemMessageMode,
		requiredAutoTruncation:     defaults.requiredAutoTruncation,
		supportsFlexProcessing:     defaults.supportsFlexProcessing,
		supportsPriorityProcessing: defaults.supportsPriorityProcessing,
	}
}

func (o responsesLanguageModel) prepareParams(call ai.Call) (*responses.ResponseNewParams, []ai.CallWarning) {
	var warnings []ai.CallWarning
	params := &responses.ResponseNewParams{
		Store: param.NewOpt(false),
	}

	modelConfig := getResponsesModelConfig(o.modelID)

	if call.TopK != nil {
		warnings = append(warnings, ai.CallWarning{
			Type:    ai.CallWarningTypeUnsupportedSetting,
			Setting: "topK",
		})
	}

	if call.PresencePenalty != nil {
		warnings = append(warnings, ai.CallWarning{
			Type:    ai.CallWarningTypeUnsupportedSetting,
			Setting: "presencePenalty",
		})
	}

	if call.FrequencyPenalty != nil {
		warnings = append(warnings, ai.CallWarning{
			Type:    ai.CallWarningTypeUnsupportedSetting,
			Setting: "frequencyPenalty",
		})
	}

	var openaiOptions *ResponsesProviderOptions
	if opts, ok := call.ProviderOptions[Name]; ok {
		if typedOpts, ok := opts.(*ResponsesProviderOptions); ok {
			openaiOptions = typedOpts
		}
	}

	input, inputWarnings := toResponsesPrompt(call.Prompt, modelConfig.systemMessageMode)
	warnings = append(warnings, inputWarnings...)

	var include []IncludeType

	addInclude := func(key IncludeType) {
		include = append(include, key)
	}

	topLogprobs := 0
	if openaiOptions != nil && openaiOptions.Logprobs != nil {
		switch v := openaiOptions.Logprobs.(type) {
		case bool:
			if v {
				topLogprobs = topLogprobsMax
			}
		case float64:
			topLogprobs = int(v)
		case int:
			topLogprobs = v
		}
	}

	if topLogprobs > 0 {
		addInclude(IncludeMessageOutputTextLogprobs)
	}

	params.Model = o.modelID
	params.Input = responses.ResponseNewParamsInputUnion{
		OfInputItemList: input,
	}

	if call.Temperature != nil {
		params.Temperature = param.NewOpt(*call.Temperature)
	}
	if call.TopP != nil {
		params.TopP = param.NewOpt(*call.TopP)
	}
	if call.MaxOutputTokens != nil {
		params.MaxOutputTokens = param.NewOpt(*call.MaxOutputTokens)
	}

	if openaiOptions != nil {
		if openaiOptions.MaxToolCalls != nil {
			params.MaxToolCalls = param.NewOpt(*openaiOptions.MaxToolCalls)
		}
		if openaiOptions.Metadata != nil {
			metadata := make(shared.Metadata)
			for k, v := range openaiOptions.Metadata {
				if str, ok := v.(string); ok {
					metadata[k] = str
				}
			}
			params.Metadata = metadata
		}
		if openaiOptions.ParallelToolCalls != nil {
			params.ParallelToolCalls = param.NewOpt(*openaiOptions.ParallelToolCalls)
		}
		if openaiOptions.User != nil {
			params.User = param.NewOpt(*openaiOptions.User)
		}
		if openaiOptions.Instructions != nil {
			params.Instructions = param.NewOpt(*openaiOptions.Instructions)
		}
		if openaiOptions.ServiceTier != nil {
			params.ServiceTier = responses.ResponseNewParamsServiceTier(*openaiOptions.ServiceTier)
		}
		if openaiOptions.PromptCacheKey != nil {
			params.PromptCacheKey = param.NewOpt(*openaiOptions.PromptCacheKey)
		}
		if openaiOptions.SafetyIdentifier != nil {
			params.SafetyIdentifier = param.NewOpt(*openaiOptions.SafetyIdentifier)
		}
		if topLogprobs > 0 {
			params.TopLogprobs = param.NewOpt(int64(topLogprobs))
		}

		if len(openaiOptions.Include) > 0 {
			include = append(include, openaiOptions.Include...)
		}

		if modelConfig.isReasoningModel && (openaiOptions.ReasoningEffort != nil || openaiOptions.ReasoningSummary != nil) {
			reasoning := shared.ReasoningParam{}
			if openaiOptions.ReasoningEffort != nil {
				reasoning.Effort = shared.ReasoningEffort(*openaiOptions.ReasoningEffort)
			}
			if openaiOptions.ReasoningSummary != nil {
				reasoning.Summary = shared.ReasoningSummary(*openaiOptions.ReasoningSummary)
			}
			params.Reasoning = reasoning
		}
	}

	if modelConfig.requiredAutoTruncation {
		params.Truncation = responses.ResponseNewParamsTruncationAuto
	}

	if len(include) > 0 {
		includeParams := make([]responses.ResponseIncludable, len(include))
		for i, inc := range include {
			includeParams[i] = responses.ResponseIncludable(string(inc))
		}
		params.Include = includeParams
	}

	if modelConfig.isReasoningModel {
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
				Setting: "topP",
				Details: "topP is not supported for reasoning models",
			})
		}
	} else {
		if openaiOptions != nil {
			if openaiOptions.ReasoningEffort != nil {
				warnings = append(warnings, ai.CallWarning{
					Type:    ai.CallWarningTypeUnsupportedSetting,
					Setting: "reasoningEffort",
					Details: "reasoningEffort is not supported for non-reasoning models",
				})
			}

			if openaiOptions.ReasoningSummary != nil {
				warnings = append(warnings, ai.CallWarning{
					Type:    ai.CallWarningTypeUnsupportedSetting,
					Setting: "reasoningSummary",
					Details: "reasoningSummary is not supported for non-reasoning models",
				})
			}
		}
	}

	if openaiOptions != nil && openaiOptions.ServiceTier != nil {
		if *openaiOptions.ServiceTier == ServiceTierFlex && !modelConfig.supportsFlexProcessing {
			warnings = append(warnings, ai.CallWarning{
				Type:    ai.CallWarningTypeUnsupportedSetting,
				Setting: "serviceTier",
				Details: "flex processing is only available for o3, o4-mini, and gpt-5 models",
			})
			params.ServiceTier = ""
		}

		if *openaiOptions.ServiceTier == ServiceTierPriority && !modelConfig.supportsPriorityProcessing {
			warnings = append(warnings, ai.CallWarning{
				Type:    ai.CallWarningTypeUnsupportedSetting,
				Setting: "serviceTier",
				Details: "priority processing is only available for supported models (gpt-4, gpt-5, gpt-5-mini, o3, o4-mini) and requires Enterprise access. gpt-5-nano is not supported",
			})
			params.ServiceTier = ""
		}
	}

	tools, toolChoice, toolWarnings := toResponsesTools(call.Tools, call.ToolChoice, openaiOptions)
	warnings = append(warnings, toolWarnings...)

	if len(tools) > 0 {
		params.Tools = tools
		params.ToolChoice = toolChoice
	}

	return params, warnings
}

func toResponsesPrompt(prompt ai.Prompt, systemMessageMode string) (responses.ResponseInputParam, []ai.CallWarning) {
	var input responses.ResponseInputParam
	var warnings []ai.CallWarning

	for _, msg := range prompt {
		switch msg.Role {
		case ai.MessageRoleSystem:
			var systemText string
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
				if strings.TrimSpace(textPart.Text) != "" {
					systemText += textPart.Text
				}
			}

			if systemText == "" {
				warnings = append(warnings, ai.CallWarning{
					Type:    ai.CallWarningTypeOther,
					Message: "system prompt has no text parts",
				})
				continue
			}

			switch systemMessageMode {
			case "system":
				input = append(input, responses.ResponseInputItemParamOfMessage(systemText, responses.EasyInputMessageRoleSystem))
			case "developer":
				input = append(input, responses.ResponseInputItemParamOfMessage(systemText, responses.EasyInputMessageRoleDeveloper))
			case "remove":
				warnings = append(warnings, ai.CallWarning{
					Type:    ai.CallWarningTypeOther,
					Message: "system messages are removed for this model",
				})
			}

		case ai.MessageRoleUser:
			var contentParts responses.ResponseInputMessageContentListParam
			for i, c := range msg.Content {
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
					contentParts = append(contentParts, responses.ResponseInputContentUnionParam{
						OfInputText: &responses.ResponseInputTextParam{
							Type: "input_text",
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

					if strings.HasPrefix(filePart.MediaType, "image/") {
						base64Encoded := base64.StdEncoding.EncodeToString(filePart.Data)
						imageURL := fmt.Sprintf("data:%s;base64,%s", filePart.MediaType, base64Encoded)
						contentParts = append(contentParts, responses.ResponseInputContentUnionParam{
							OfInputImage: &responses.ResponseInputImageParam{
								Type:     "input_image",
								ImageURL: param.NewOpt(imageURL),
							},
						})
					} else if filePart.MediaType == "application/pdf" {
						base64Encoded := base64.StdEncoding.EncodeToString(filePart.Data)
						fileData := fmt.Sprintf("data:application/pdf;base64,%s", base64Encoded)
						filename := filePart.Filename
						if filename == "" {
							filename = fmt.Sprintf("part-%d.pdf", i)
						}
						contentParts = append(contentParts, responses.ResponseInputContentUnionParam{
							OfInputFile: &responses.ResponseInputFileParam{
								Type:     "input_file",
								Filename: param.NewOpt(filename),
								FileData: param.NewOpt(fileData),
							},
						})
					} else {
						warnings = append(warnings, ai.CallWarning{
							Type:    ai.CallWarningTypeOther,
							Message: fmt.Sprintf("file part media type %s not supported", filePart.MediaType),
						})
					}
				}
			}

			input = append(input, responses.ResponseInputItemParamOfMessage(contentParts, responses.EasyInputMessageRoleUser))

		case ai.MessageRoleAssistant:
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
					input = append(input, responses.ResponseInputItemParamOfMessage(textPart.Text, responses.EasyInputMessageRoleAssistant))

				case ai.ContentTypeToolCall:
					toolCallPart, ok := ai.AsContentType[ai.ToolCallPart](c)
					if !ok {
						warnings = append(warnings, ai.CallWarning{
							Type:    ai.CallWarningTypeOther,
							Message: "assistant message tool call part does not have the right type",
						})
						continue
					}

					if toolCallPart.ProviderExecuted {
						continue
					}

					inputJSON, err := json.Marshal(toolCallPart.Input)
					if err != nil {
						warnings = append(warnings, ai.CallWarning{
							Type:    ai.CallWarningTypeOther,
							Message: fmt.Sprintf("failed to marshal tool call input: %v", err),
						})
						continue
					}

					input = append(input, responses.ResponseInputItemParamOfFunctionCall(string(inputJSON), toolCallPart.ToolCallID, toolCallPart.ToolName))
				case ai.ContentTypeReasoning:
					reasoningMetadata := getReasoningMetadata(c.Options())
					if reasoningMetadata == nil || reasoningMetadata.ItemID == "" {
						continue
					}
					if len(reasoningMetadata.Summary) == 0 && reasoningMetadata.EncryptedContent == nil {
						warnings = append(warnings, ai.CallWarning{
							Type:    ai.CallWarningTypeOther,
							Message: "assistant message reasoning part does is empty",
						})
						continue
					}
					// we want to always send an empty array
					summary := []responses.ResponseReasoningItemSummaryParam{}
					for _, s := range reasoningMetadata.Summary {
						summary = append(summary, responses.ResponseReasoningItemSummaryParam{
							Type: "summary_text",
							Text: s,
						})
					}
					reasoning := &responses.ResponseReasoningItemParam{
						ID:      reasoningMetadata.ItemID,
						Summary: summary,
					}
					if reasoningMetadata.EncryptedContent != nil {
						reasoning.EncryptedContent = param.NewOpt(*reasoningMetadata.EncryptedContent)
					}
					input = append(input, responses.ResponseInputItemUnionParam{
						OfReasoning: reasoning,
					})
				}
			}

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

				var outputStr string
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
					outputStr = output.Text
				case ai.ToolResultContentTypeError:
					output, ok := ai.AsToolResultOutputType[ai.ToolResultOutputContentError](toolResultPart.Output)
					if !ok {
						warnings = append(warnings, ai.CallWarning{
							Type:    ai.CallWarningTypeOther,
							Message: "tool result output does not have the right type",
						})
						continue
					}
					outputStr = output.Error.Error()
				case ai.ToolResultContentTypeMedia:
					output, ok := ai.AsToolResultOutputType[ai.ToolResultOutputContentMedia](toolResultPart.Output)
					if !ok {
						warnings = append(warnings, ai.CallWarning{
							Type:    ai.CallWarningTypeOther,
							Message: "tool result output does not have the right type",
						})
						continue
					}
					// For media content, encode as JSON with data and media type
					mediaContent := map[string]string{
						"data":       output.Data,
						"media_type": output.MediaType,
					}
					jsonBytes, err := json.Marshal(mediaContent)
					if err != nil {
						warnings = append(warnings, ai.CallWarning{
							Type:    ai.CallWarningTypeOther,
							Message: fmt.Sprintf("failed to marshal tool result: %v", err),
						})
						continue
					}
					outputStr = string(jsonBytes)
				}

				input = append(input, responses.ResponseInputItemParamOfFunctionCallOutput(toolResultPart.ToolCallID, outputStr))
			}
		}
	}

	return input, warnings
}

func toResponsesTools(tools []ai.Tool, toolChoice *ai.ToolChoice, options *ResponsesProviderOptions) ([]responses.ToolUnionParam, responses.ResponseNewParamsToolChoiceUnion, []ai.CallWarning) {
	warnings := make([]ai.CallWarning, 0)
	var openaiTools []responses.ToolUnionParam

	if len(tools) == 0 {
		return nil, responses.ResponseNewParamsToolChoiceUnion{}, nil
	}

	strictJSONSchema := false
	if options != nil && options.StrictJSONSchema != nil {
		strictJSONSchema = *options.StrictJSONSchema
	}

	for _, tool := range tools {
		if tool.GetType() == ai.ToolTypeFunction {
			ft, ok := tool.(ai.FunctionTool)
			if !ok {
				continue
			}
			openaiTools = append(openaiTools, responses.ToolUnionParam{
				OfFunction: &responses.FunctionToolParam{
					Name:        ft.Name,
					Description: param.NewOpt(ft.Description),
					Parameters:  ft.InputSchema,
					Strict:      param.NewOpt(strictJSONSchema),
					Type:        "function",
				},
			})
			continue
		}

		warnings = append(warnings, ai.CallWarning{
			Type:    ai.CallWarningTypeUnsupportedTool,
			Tool:    tool,
			Message: "tool is not supported",
		})
	}

	if toolChoice == nil {
		return openaiTools, responses.ResponseNewParamsToolChoiceUnion{}, warnings
	}

	var openaiToolChoice responses.ResponseNewParamsToolChoiceUnion

	switch *toolChoice {
	case ai.ToolChoiceAuto:
		openaiToolChoice = responses.ResponseNewParamsToolChoiceUnion{
			OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsAuto),
		}
	case ai.ToolChoiceNone:
		openaiToolChoice = responses.ResponseNewParamsToolChoiceUnion{
			OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsNone),
		}
	case ai.ToolChoiceRequired:
		openaiToolChoice = responses.ResponseNewParamsToolChoiceUnion{
			OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsRequired),
		}
	default:
		openaiToolChoice = responses.ResponseNewParamsToolChoiceUnion{
			OfFunctionTool: &responses.ToolChoiceFunctionParam{
				Type: "function",
				Name: string(*toolChoice),
			},
		}
	}

	return openaiTools, openaiToolChoice, warnings
}

func (o responsesLanguageModel) handleError(err error) error {
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

func (o responsesLanguageModel) Generate(ctx context.Context, call ai.Call) (*ai.Response, error) {
	params, warnings := o.prepareParams(call)
	response, err := o.client.Responses.New(ctx, *params)
	if err != nil {
		return nil, o.handleError(err)
	}

	if response.Error.Message != "" {
		return nil, o.handleError(fmt.Errorf("response error: %s (code: %s)", response.Error.Message, response.Error.Code))
	}

	var content []ai.Content
	hasFunctionCall := false

	for _, outputItem := range response.Output {
		switch outputItem.Type {
		case "message":
			for _, contentPart := range outputItem.Content {
				if contentPart.Type == "output_text" {
					content = append(content, ai.TextContent{
						Text: contentPart.Text,
					})

					for _, annotation := range contentPart.Annotations {
						switch annotation.Type {
						case "url_citation":
							content = append(content, ai.SourceContent{
								SourceType: ai.SourceTypeURL,
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
							content = append(content, ai.SourceContent{
								SourceType: ai.SourceTypeDocument,
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
			content = append(content, ai.ToolCallContent{
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

			// When there are no summary parts, add an empty reasoning part
			summaries := outputItem.Summary
			if len(summaries) == 0 {
				summaries = []responses.ResponseReasoningItemSummary{{Type: "summary_text", Text: ""}}
			}

			var summaryParts []string

			for _, s := range summaries {
				metadata.Summary = append(metadata.Summary, s.Text)
				summaryParts = append(summaryParts, s.Text)
			}

			content = append(content, ai.ReasoningContent{
				Text: strings.Join(summaryParts, "\n"),
				ProviderMetadata: ai.ProviderMetadata{
					Name: metadata,
				},
			})
		}
	}

	usage := ai.Usage{
		InputTokens:  response.Usage.InputTokens,
		OutputTokens: response.Usage.OutputTokens,
		TotalTokens:  response.Usage.InputTokens + response.Usage.OutputTokens,
	}

	if response.Usage.OutputTokensDetails.ReasoningTokens != 0 {
		usage.ReasoningTokens = response.Usage.OutputTokensDetails.ReasoningTokens
	}
	if response.Usage.InputTokensDetails.CachedTokens != 0 {
		usage.CacheReadTokens = response.Usage.InputTokensDetails.CachedTokens
	}

	finishReason := mapResponsesFinishReason(response.IncompleteDetails.Reason, hasFunctionCall)

	return &ai.Response{
		Content:          content,
		Usage:            usage,
		FinishReason:     finishReason,
		ProviderMetadata: ai.ProviderMetadata{},
		Warnings:         warnings,
	}, nil
}

func mapResponsesFinishReason(reason string, hasFunctionCall bool) ai.FinishReason {
	if hasFunctionCall {
		return ai.FinishReasonToolCalls
	}

	switch reason {
	case "":
		return ai.FinishReasonStop
	case "max_tokens", "max_output_tokens":
		return ai.FinishReasonLength
	case "content_filter":
		return ai.FinishReasonContentFilter
	default:
		return ai.FinishReasonOther
	}
}

func (o responsesLanguageModel) Stream(ctx context.Context, call ai.Call) (ai.StreamResponse, error) {
	params, warnings := o.prepareParams(call)

	stream := o.client.Responses.NewStreaming(ctx, *params)

	finishReason := ai.FinishReasonUnknown
	var usage ai.Usage
	ongoingToolCalls := make(map[int64]*ongoingToolCall)
	hasFunctionCall := false
	activeReasoning := make(map[string]*reasoningState)

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
			event := stream.Current()

			switch event.Type {
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
					if !yield(ai.StreamPart{
						Type:         ai.StreamPartTypeToolInputStart,
						ID:           added.Item.CallID,
						ToolCallName: added.Item.Name,
					}) {
						return
					}

				case "message":
					if !yield(ai.StreamPart{
						Type: ai.StreamPartTypeTextStart,
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

					activeReasoning[added.Item.ID] = &reasoningState{
						metadata: metadata,
					}
					if !yield(ai.StreamPart{
						Type: ai.StreamPartTypeReasoningStart,
						ID:   added.Item.ID,
						ProviderMetadata: ai.ProviderMetadata{
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

						if !yield(ai.StreamPart{
							Type: ai.StreamPartTypeToolInputEnd,
							ID:   done.Item.CallID,
						}) {
							return
						}
						if !yield(ai.StreamPart{
							Type:          ai.StreamPartTypeToolCall,
							ID:            done.Item.CallID,
							ToolCallName:  done.Item.Name,
							ToolCallInput: done.Item.Arguments,
						}) {
							return
						}
					}

				case "message":
					if !yield(ai.StreamPart{
						Type: ai.StreamPartTypeTextEnd,
						ID:   done.Item.ID,
					}) {
						return
					}

				case "reasoning":
					state := activeReasoning[done.Item.ID]
					if state != nil {
						if !yield(ai.StreamPart{
							Type: ai.StreamPartTypeReasoningEnd,
							ID:   done.Item.ID,
							ProviderMetadata: ai.ProviderMetadata{
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
					if !yield(ai.StreamPart{
						Type:  ai.StreamPartTypeToolInputDelta,
						ID:    tc.toolCallID,
						Delta: delta.Delta,
					}) {
						return
					}
				}

			case "response.output_text.delta":
				textDelta := event.AsResponseOutputTextDelta()
				if !yield(ai.StreamPart{
					Type:  ai.StreamPartTypeTextDelta,
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
					activeReasoning[added.ItemID] = state
					if !yield(ai.StreamPart{
						Type:  ai.StreamPartTypeReasoningDelta,
						ID:    added.ItemID,
						Delta: "\n",
						ProviderMetadata: ai.ProviderMetadata{
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
					activeReasoning[textDelta.ItemID] = state
					if !yield(ai.StreamPart{
						Type:  ai.StreamPartTypeReasoningDelta,
						ID:    textDelta.ItemID,
						Delta: textDelta.Delta,
						ProviderMetadata: ai.ProviderMetadata{
							Name: state.metadata,
						},
					}) {
						return
					}
				}

			case "response.completed", "response.incomplete":
				completed := event.AsResponseCompleted()
				finishReason = mapResponsesFinishReason(completed.Response.IncompleteDetails.Reason, hasFunctionCall)
				usage = ai.Usage{
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

			case "error":
				errorEvent := event.AsError()
				if !yield(ai.StreamPart{
					Type:  ai.StreamPartTypeError,
					Error: fmt.Errorf("response error: %s (code: %s)", errorEvent.Message, errorEvent.Code),
				}) {
					return
				}
				return
			}
		}

		err := stream.Err()
		if err != nil {
			yield(ai.StreamPart{
				Type:  ai.StreamPartTypeError,
				Error: o.handleError(err),
			})
			return
		}

		yield(ai.StreamPart{
			Type:         ai.StreamPartTypeFinish,
			Usage:        usage,
			FinishReason: finishReason,
		})
	}, nil
}

func getReasoningMetadata(providerOptions ai.ProviderOptions) *ResponsesReasoningMetadata {
	if openaiResponsesOptions, ok := providerOptions[Name]; ok {
		if reasoning, ok := openaiResponsesOptions.(*ResponsesReasoningMetadata); ok {
			return reasoning
		}
	}
	return nil
}

type ongoingToolCall struct {
	toolName   string
	toolCallID string
}

type reasoningState struct {
	metadata *ResponsesReasoningMetadata
}
