package bedrock

import (
	"encoding/base64"
	"encoding/json"
	"fmt"

	"charm.land/fantasy"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// prepareConverseRequest converts a fantasy.Call to a Converse API request.
// It returns the request, any warnings, and an error if conversion fails.
func (n *novaLanguageModel) prepareConverseRequest(call fantasy.Call) (*bedrockruntime.ConverseInput, []fantasy.CallWarning, error) {
	var warnings []fantasy.CallWarning

	// Convert messages to Converse API format
	messages, systemBlocks, err := convertMessages(call.Prompt)
	if err != nil {
		return nil, warnings, fmt.Errorf("failed to convert messages: %w", err)
	}

	// Build inference configuration
	inferenceConfig := &types.InferenceConfiguration{}
	if call.MaxOutputTokens != nil {
		inferenceConfig.MaxTokens = aws.Int32(int32(*call.MaxOutputTokens))
	}
	if call.Temperature != nil {
		inferenceConfig.Temperature = aws.Float32(float32(*call.Temperature))
	}
	if call.TopP != nil {
		inferenceConfig.TopP = aws.Float32(float32(*call.TopP))
	}

	// Build additional model request fields for top_k
	// Note: Nova models do not support top_k parameter, but we still set it in additional fields
	var additionalFields document.Interface
	if call.TopK != nil {
		// Set top_k in additional fields (even though Nova doesn't support it)
		additionalFieldsMap := map[string]any{
			"top_k": *call.TopK,
		}
		additionalFields = document.NewLazyDocument(additionalFieldsMap)

		// Add warning that top_k is not supported for Nova models
		warnings = append(warnings, fantasy.CallWarning{
			Type:    fantasy.CallWarningTypeUnsupportedSetting,
			Setting: "top_k",
			Message: "top_k parameter is not supported by Amazon Nova models and will be ignored",
		})
	}

	// Build the request
	request := &bedrockruntime.ConverseInput{
		ModelId:                      aws.String(n.modelID),
		Messages:                     messages,
		InferenceConfig:              inferenceConfig,
		AdditionalModelRequestFields: additionalFields,
	}

	// Add system blocks if present
	if len(systemBlocks) > 0 {
		request.System = systemBlocks
	}

	// Add tool configuration if tools are provided
	if len(call.Tools) > 0 {
		toolConfig, toolWarnings := convertTools(call.Tools, call.ToolChoice)
		request.ToolConfig = toolConfig
		warnings = append(warnings, toolWarnings...)
	}

	return request, warnings, nil
}

// convertMessages converts fantasy messages to Converse API messages and system blocks.
func convertMessages(prompt fantasy.Prompt) ([]types.Message, []types.SystemContentBlock, error) {
	var messages []types.Message
	var systemBlocks []types.SystemContentBlock

	for _, msg := range prompt {
		switch msg.Role {
		case fantasy.MessageRoleSystem:
			// Convert system messages to SystemContentBlock
			for _, part := range msg.Content {
				if part.GetType() == fantasy.ContentTypeText {
					if textPart, ok := fantasy.AsMessagePart[fantasy.TextPart](part); ok {
						systemBlocks = append(systemBlocks, &types.SystemContentBlockMemberText{
							Value: textPart.Text,
						})
					}
				}
			}

		case fantasy.MessageRoleUser, fantasy.MessageRoleAssistant:
			// Convert user and assistant messages
			contentBlocks, err := convertMessageContent(msg.Content)
			if err != nil {
				return nil, nil, fmt.Errorf("failed to convert message content: %w", err)
			}

			var role types.ConversationRole
			if msg.Role == fantasy.MessageRoleUser {
				role = types.ConversationRoleUser
			} else {
				role = types.ConversationRoleAssistant
			}

			messages = append(messages, types.Message{
				Role:    role,
				Content: contentBlocks,
			})

		case fantasy.MessageRoleTool:
			// Tool results are included in the previous assistant message
			// or as a separate user message with tool results
			contentBlocks, err := convertMessageContent(msg.Content)
			if err != nil {
				return nil, nil, fmt.Errorf("failed to convert tool message content: %w", err)
			}

			messages = append(messages, types.Message{
				Role:    types.ConversationRoleUser,
				Content: contentBlocks,
			})
		}
	}

	return messages, systemBlocks, nil
}

// convertMessageContent converts fantasy message parts to Converse API content blocks.
func convertMessageContent(content []fantasy.MessagePart) ([]types.ContentBlock, error) {
	var blocks []types.ContentBlock

	for _, part := range content {
		switch part.GetType() {
		case fantasy.ContentTypeText:
			if textPart, ok := fantasy.AsMessagePart[fantasy.TextPart](part); ok {
				blocks = append(blocks, &types.ContentBlockMemberText{
					Value: textPart.Text,
				})
			}

		case fantasy.ContentTypeFile:
			if filePart, ok := fantasy.AsMessagePart[fantasy.FilePart](part); ok {
				// Convert image attachments to Converse image blocks
				if isImageMediaType(filePart.MediaType) {
					imageBlock, err := convertImageAttachment(filePart)
					if err != nil {
						return nil, fmt.Errorf("failed to convert image attachment: %w", err)
					}
					blocks = append(blocks, imageBlock)
				}
				// Note: Non-image files are not supported in Converse API
			}

		case fantasy.ContentTypeToolCall:
			if toolCallPart, ok := fantasy.AsMessagePart[fantasy.ToolCallPart](part); ok {
				toolUseBlock, err := convertToolCall(toolCallPart)
				if err != nil {
					return nil, fmt.Errorf("failed to convert tool call: %w", err)
				}
				blocks = append(blocks, toolUseBlock)
			}

		case fantasy.ContentTypeToolResult:
			if toolResultPart, ok := fantasy.AsMessagePart[fantasy.ToolResultPart](part); ok {
				toolResultBlock, err := convertToolResult(toolResultPart)
				if err != nil {
					return nil, fmt.Errorf("failed to convert tool result: %w", err)
				}
				blocks = append(blocks, toolResultBlock)
			}
		}
	}

	return blocks, nil
}

// isImageMediaType checks if a media type is an image type.
func isImageMediaType(mediaType string) bool {
	switch mediaType {
	case "image/jpeg", "image/png", "image/gif", "image/webp":
		return true
	default:
		return false
	}
}

// convertImageAttachment converts a fantasy FilePart to a Converse image block.
func convertImageAttachment(filePart fantasy.FilePart) (types.ContentBlock, error) {
	// Determine the image format
	var format types.ImageFormat
	switch filePart.MediaType {
	case "image/jpeg", "image/jpg":
		format = types.ImageFormatJpeg
	case "image/png":
		format = types.ImageFormatPng
	case "image/gif":
		format = types.ImageFormatGif
	case "image/webp":
		format = types.ImageFormatWebp
	default:
		return nil, fmt.Errorf("unsupported image media type: %s", filePart.MediaType)
	}

	// Create image source from bytes
	imageSource := &types.ImageSourceMemberBytes{
		Value: filePart.Data,
	}

	return &types.ContentBlockMemberImage{
		Value: types.ImageBlock{
			Format: format,
			Source: imageSource,
		},
	}, nil
}

// convertToolCall converts a fantasy ToolCallPart to a Converse tool use block.
func convertToolCall(toolCallPart fantasy.ToolCallPart) (types.ContentBlock, error) {
	// Parse the input JSON string to a document
	var inputMap map[string]any
	if err := json.Unmarshal([]byte(toolCallPart.Input), &inputMap); err != nil {
		return nil, fmt.Errorf("failed to parse tool call input: %w", err)
	}

	return &types.ContentBlockMemberToolUse{
		Value: types.ToolUseBlock{
			ToolUseId: aws.String(toolCallPart.ToolCallID),
			Name:      aws.String(toolCallPart.ToolName),
			Input:     document.NewLazyDocument(inputMap),
		},
	}, nil
}

// convertToolResult converts a fantasy ToolResultPart to a Converse tool result block.
func convertToolResult(toolResultPart fantasy.ToolResultPart) (types.ContentBlock, error) {
	var contentBlocks []types.ToolResultContentBlock

	switch output := toolResultPart.Output.(type) {
	case fantasy.ToolResultOutputContentText:
		contentBlocks = append(contentBlocks, &types.ToolResultContentBlockMemberText{
			Value: output.Text,
		})

	case fantasy.ToolResultOutputContentError:
		errorText := "Error"
		if output.Error != nil {
			errorText = output.Error.Error()
		}
		contentBlocks = append(contentBlocks, &types.ToolResultContentBlockMemberText{
			Value: errorText,
		})

	case fantasy.ToolResultOutputContentMedia:
		// For media content, decode base64 and create image block
		if output.MediaType != "" && isImageMediaType(output.MediaType) {
			imageData, err := base64.StdEncoding.DecodeString(output.Data)
			if err != nil {
				return nil, fmt.Errorf("failed to decode image data: %w", err)
			}

			var format types.ImageFormat
			switch output.MediaType {
			case "image/jpeg", "image/jpg":
				format = types.ImageFormatJpeg
			case "image/png":
				format = types.ImageFormatPng
			case "image/gif":
				format = types.ImageFormatGif
			case "image/webp":
				format = types.ImageFormatWebp
			}

			contentBlocks = append(contentBlocks, &types.ToolResultContentBlockMemberImage{
				Value: types.ImageBlock{
					Format: format,
					Source: &types.ImageSourceMemberBytes{
						Value: imageData,
					},
				},
			})
		}

		// Add text if present
		if output.Text != "" {
			contentBlocks = append(contentBlocks, &types.ToolResultContentBlockMemberText{
				Value: output.Text,
			})
		}
	}

	return &types.ContentBlockMemberToolResult{
		Value: types.ToolResultBlock{
			ToolUseId: aws.String(toolResultPart.ToolCallID),
			Content:   contentBlocks,
		},
	}, nil
}

// convertTools converts fantasy tools to Converse tool configuration.
func convertTools(tools []fantasy.Tool, toolChoice *fantasy.ToolChoice) (*types.ToolConfiguration, []fantasy.CallWarning) {
	var warnings []fantasy.CallWarning
	var toolSpecs []types.Tool

	for _, tool := range tools {
		if tool.GetType() == fantasy.ToolTypeFunction {
			if funcTool, ok := tool.(fantasy.FunctionTool); ok {
				// Convert input schema to document
				inputSchema := document.NewLazyDocument(funcTool.InputSchema)

				toolSpecs = append(toolSpecs, &types.ToolMemberToolSpec{
					Value: types.ToolSpecification{
						Name:        aws.String(funcTool.Name),
						Description: aws.String(funcTool.Description),
						InputSchema: &types.ToolInputSchemaMemberJson{
							Value: inputSchema,
						},
					},
				})
			}
		} else {
			// Provider-defined tools are not supported
			warnings = append(warnings, fantasy.CallWarning{
				Type:    fantasy.CallWarningTypeUnsupportedTool,
				Tool:    tool,
				Message: fmt.Sprintf("Provider-defined tools are not supported by Converse API: %s", tool.GetName()),
			})
		}
	}

	toolConfig := &types.ToolConfiguration{
		Tools: toolSpecs,
	}

	// Convert tool choice
	if toolChoice != nil {
		switch *toolChoice {
		case fantasy.ToolChoiceAuto:
			toolConfig.ToolChoice = &types.ToolChoiceMemberAuto{
				Value: types.AutoToolChoice{},
			}
		case fantasy.ToolChoiceRequired:
			toolConfig.ToolChoice = &types.ToolChoiceMemberAny{
				Value: types.AnyToolChoice{},
			}
		case fantasy.ToolChoiceNone:
			// No tool choice means don't include tools
			return nil, warnings
		default:
			// Specific tool choice
			toolName := string(*toolChoice)
			toolConfig.ToolChoice = &types.ToolChoiceMemberTool{
				Value: types.SpecificToolChoice{
					Name: aws.String(toolName),
				},
			}
		}
	}

	return toolConfig, warnings
}

// convertConverseResponse converts a Converse API response to a fantasy.Response.
func (n *novaLanguageModel) convertConverseResponse(output *bedrockruntime.ConverseOutput, warnings []fantasy.CallWarning) (*fantasy.Response, error) {
	if output == nil {
		return nil, fmt.Errorf("converse output is nil")
	}

	// Convert content blocks to fantasy content
	var content fantasy.ResponseContent
	if output.Output != nil {
		message := output.Output.(*types.ConverseOutputMemberMessage).Value
		for _, block := range message.Content {
			fantasyContent, err := convertContentBlock(block)
			if err != nil {
				return nil, fmt.Errorf("failed to convert content block: %w", err)
			}
			if fantasyContent != nil {
				content = append(content, fantasyContent)
			}
		}
	}

	// Convert usage statistics
	usage := fantasy.Usage{}
	if output.Usage != nil {
		if output.Usage.InputTokens != nil {
			usage.InputTokens = int64(*output.Usage.InputTokens)
		}
		if output.Usage.OutputTokens != nil {
			usage.OutputTokens = int64(*output.Usage.OutputTokens)
		}
		if output.Usage.TotalTokens != nil {
			usage.TotalTokens = int64(*output.Usage.TotalTokens)
		}
	}

	// Convert stop reason to finish reason
	finishReason := convertStopReason(output.StopReason)

	return &fantasy.Response{
		Content:      content,
		FinishReason: finishReason,
		Usage:        usage,
		Warnings:     warnings,
	}, nil
}

// convertContentBlock converts a Converse API content block to fantasy content.
func convertContentBlock(block types.ContentBlock) (fantasy.Content, error) {
	switch b := block.(type) {
	case *types.ContentBlockMemberText:
		return fantasy.TextContent{
			Text: b.Value,
		}, nil

	case *types.ContentBlockMemberToolUse:
		// Convert tool use to tool call content
		inputBytes, err := json.Marshal(b.Value.Input)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal tool input: %w", err)
		}

		toolCallID := ""
		if b.Value.ToolUseId != nil {
			toolCallID = *b.Value.ToolUseId
		}
		toolName := ""
		if b.Value.Name != nil {
			toolName = *b.Value.Name
		}

		return fantasy.ToolCallContent{
			ToolCallID: toolCallID,
			ToolName:   toolName,
			Input:      string(inputBytes),
		}, nil

	case *types.ContentBlockMemberImage:
		// Convert image block to file content
		var data []byte
		if imageSource, ok := b.Value.Source.(*types.ImageSourceMemberBytes); ok {
			data = imageSource.Value
		}

		// Determine media type from format
		var mediaType string
		switch b.Value.Format {
		case types.ImageFormatJpeg:
			mediaType = "image/jpeg"
		case types.ImageFormatPng:
			mediaType = "image/png"
		case types.ImageFormatGif:
			mediaType = "image/gif"
		case types.ImageFormatWebp:
			mediaType = "image/webp"
		default:
			mediaType = "image/jpeg" // default
		}

		return fantasy.FileContent{
			MediaType: mediaType,
			Data:      data,
		}, nil

	default:
		// Unknown content block type, skip it
		return nil, nil
	}
}

// convertStopReason converts a Converse API stop reason to a fantasy.FinishReason.
func convertStopReason(stopReason types.StopReason) fantasy.FinishReason {
	switch stopReason {
	case types.StopReasonEndTurn:
		return fantasy.FinishReasonStop
	case types.StopReasonMaxTokens:
		return fantasy.FinishReasonLength
	case types.StopReasonStopSequence:
		return fantasy.FinishReasonStop
	case types.StopReasonToolUse:
		return fantasy.FinishReasonToolCalls
	case types.StopReasonContentFiltered:
		return fantasy.FinishReasonContentFilter
	default:
		return fantasy.FinishReasonUnknown
	}
}

// prepareConverseStreamRequest converts a fantasy.Call to a ConverseStream API request.
// It returns the request, any warnings, and an error if conversion fails.
func (n *novaLanguageModel) prepareConverseStreamRequest(call fantasy.Call) (*bedrockruntime.ConverseStreamInput, []fantasy.CallWarning, error) {
	var warnings []fantasy.CallWarning

	// Convert messages to Converse API format
	messages, systemBlocks, err := convertMessages(call.Prompt)
	if err != nil {
		return nil, warnings, fmt.Errorf("failed to convert messages: %w", err)
	}

	// Build inference configuration
	inferenceConfig := &types.InferenceConfiguration{}
	if call.MaxOutputTokens != nil {
		inferenceConfig.MaxTokens = aws.Int32(int32(*call.MaxOutputTokens))
	}
	if call.Temperature != nil {
		inferenceConfig.Temperature = aws.Float32(float32(*call.Temperature))
	}
	if call.TopP != nil {
		inferenceConfig.TopP = aws.Float32(float32(*call.TopP))
	}

	// Build additional model request fields for top_k
	// Note: Nova models do not support top_k parameter, but we still set it in additional fields
	var additionalFields document.Interface
	if call.TopK != nil {
		// Set top_k in additional fields (even though Nova doesn't support it)
		additionalFieldsMap := map[string]any{
			"top_k": *call.TopK,
		}
		additionalFields = document.NewLazyDocument(additionalFieldsMap)

		// Add warning that top_k is not supported for Nova models
		warnings = append(warnings, fantasy.CallWarning{
			Type:    fantasy.CallWarningTypeUnsupportedSetting,
			Setting: "top_k",
			Message: "top_k parameter is not supported by Amazon Nova models and will be ignored",
		})
	}

	// Build the request
	request := &bedrockruntime.ConverseStreamInput{
		ModelId:                      aws.String(n.modelID),
		Messages:                     messages,
		InferenceConfig:              inferenceConfig,
		AdditionalModelRequestFields: additionalFields,
	}

	// Add system blocks if present
	if len(systemBlocks) > 0 {
		request.System = systemBlocks
	}

	// Add tool configuration if tools are provided
	if len(call.Tools) > 0 {
		toolConfig, toolWarnings := convertTools(call.Tools, call.ToolChoice)
		request.ToolConfig = toolConfig
		warnings = append(warnings, toolWarnings...)
	}

	return request, warnings, nil
}

// handleConverseStream handles the ConverseStream API response and yields fantasy.StreamPart events.
func (n *novaLanguageModel) handleConverseStream(output *bedrockruntime.ConverseStreamOutput, warnings []fantasy.CallWarning) fantasy.StreamResponse {
	return func(yield func(fantasy.StreamPart) bool) {
		// Yield warnings as first stream part if present
		if len(warnings) > 0 {
			if !yield(fantasy.StreamPart{
				Type:     fantasy.StreamPartTypeWarnings,
				Warnings: warnings,
			}) {
				return
			}
		}

		// Track accumulated content for final response
		var accumulatedText string
		var accumulatedToolCalls []fantasy.ToolCallContent
		var currentToolCallID string
		var currentToolCallName string
		var currentToolCallInput string
		var usage fantasy.Usage
		var finishReason fantasy.FinishReason

		// Get the event stream
		stream := output.GetStream()
		if stream == nil {
			yield(fantasy.StreamPart{
				Type:  fantasy.StreamPartTypeError,
				Error: fmt.Errorf("stream is nil"),
			})
			return
		}

		// Iterate over stream events
		for event := range stream.Events() {
			switch e := event.(type) {
			case *types.ConverseStreamOutputMemberContentBlockStart:
				// Handle content block start
				if e.Value.Start != nil {
					switch start := e.Value.Start.(type) {
					case *types.ContentBlockStartMemberToolUse:
						// Tool use block started
						if start.Value.ToolUseId != nil {
							currentToolCallID = *start.Value.ToolUseId
						}
						if start.Value.Name != nil {
							currentToolCallName = *start.Value.Name
						}
						currentToolCallInput = ""

						if !yield(fantasy.StreamPart{
							Type:         fantasy.StreamPartTypeToolInputStart,
							ID:           currentToolCallID,
							ToolCallName: currentToolCallName,
						}) {
							return
						}
					}
				}

			case *types.ConverseStreamOutputMemberContentBlockDelta:
				// Handle content block delta
				if e.Value.Delta != nil {
					switch delta := e.Value.Delta.(type) {
					case *types.ContentBlockDeltaMemberText:
						// Text delta
						deltaText := delta.Value
						accumulatedText += deltaText

						if !yield(fantasy.StreamPart{
							Type:  fantasy.StreamPartTypeTextDelta,
							Delta: deltaText,
						}) {
							return
						}
					case *types.ContentBlockDeltaMemberToolUse:
						// Tool use input delta
						if delta.Value.Input != nil {
							deltaText := *delta.Value.Input
							currentToolCallInput += deltaText

							if !yield(fantasy.StreamPart{
								Type:  fantasy.StreamPartTypeToolInputDelta,
								ID:    currentToolCallID,
								Delta: deltaText,
							}) {
								return
							}
						}
					}
				}

			case *types.ConverseStreamOutputMemberContentBlockStop:
				// Handle content block stop
				if currentToolCallID != "" {
					// Tool use block ended
					accumulatedToolCalls = append(accumulatedToolCalls, fantasy.ToolCallContent{
						ToolCallID: currentToolCallID,
						ToolName:   currentToolCallName,
						Input:      currentToolCallInput,
					})

					if !yield(fantasy.StreamPart{
						Type:          fantasy.StreamPartTypeToolInputEnd,
						ID:            currentToolCallID,
						ToolCallInput: currentToolCallInput,
					}) {
						return
					}

					// Reset tool call tracking
					currentToolCallID = ""
					currentToolCallName = ""
					currentToolCallInput = ""
				}

			case *types.ConverseStreamOutputMemberMessageStart:
				// Message started - no action needed for Nova

			case *types.ConverseStreamOutputMemberMessageStop:
				// Message stopped - extract stop reason
				if e.Value.StopReason != "" {
					finishReason = convertStopReason(e.Value.StopReason)
				}

			case *types.ConverseStreamOutputMemberMetadata:
				// Metadata event - extract usage statistics
				if e.Value.Usage != nil {
					if e.Value.Usage.InputTokens != nil {
						usage.InputTokens = int64(*e.Value.Usage.InputTokens)
					}
					if e.Value.Usage.OutputTokens != nil {
						usage.OutputTokens = int64(*e.Value.Usage.OutputTokens)
					}
					if e.Value.Usage.TotalTokens != nil {
						usage.TotalTokens = int64(*e.Value.Usage.TotalTokens)
					}
				}

			default:
				// Unknown event type, skip it
			}
		}

		// Check for stream errors
		if err := stream.Err(); err != nil {
			yield(fantasy.StreamPart{
				Type:  fantasy.StreamPartTypeError,
				Error: convertAWSError(err),
			})
			return
		}

		// Yield finish part with usage statistics
		yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeFinish,
			Usage:        usage,
			FinishReason: finishReason,
		})
	}
}
