package bedrock

import (
	"context"
	"encoding/json"
	"testing"

	"charm.land/fantasy"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"pgregory.net/rapid"
)

// Feature: amazon-nova-bedrock-support, Property 6: Request Format Validity
// For any valid fantasy.Call, the conversion to a Converse API request should produce
// a request that satisfies the Converse API specification (valid message roles, properly
// formatted content blocks, valid inference configuration).
func TestProperty_RequestFormatValidity(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a valid fantasy.Call
		call := generateValidCall(t)

		// Create a nova language model instance
		cfg, err := config.LoadDefaultConfig(context.Background())
		if err != nil {
			t.Skip("AWS configuration not available")
		}

		client := bedrockruntime.NewFromConfig(cfg)
		model := &novaLanguageModel{
			modelID:  "amazon.nova-pro-v1:0",
			provider: Name,
			client:   client,
			options:  options{},
		}

		// Convert to Converse API request
		request, warnings, err := model.prepareConverseRequest(call)

		// The conversion should succeed
		if err != nil {
			t.Fatalf("prepareConverseRequest failed: %v", err)
		}

		// Validate the request format
		validateConverseRequest(t, request, warnings, call)
	})
}

// generateValidCall generates a valid fantasy.Call for property testing.
func generateValidCall(t *rapid.T) fantasy.Call {
	// Generate prompt with at least one message
	numMessages := rapid.IntRange(1, 5).Draw(t, "numMessages")
	var prompt fantasy.Prompt

	// Optionally add system message
	if rapid.Bool().Draw(t, "hasSystem") {
		prompt = append(prompt, fantasy.Message{
			Role: fantasy.MessageRoleSystem,
			Content: []fantasy.MessagePart{
				fantasy.TextPart{
					Text: rapid.String().Draw(t, "systemText"),
				},
			},
		})
	}

	// Add user/assistant messages
	for i := 0; i < numMessages; i++ {
		role := fantasy.MessageRoleUser
		if i%2 == 1 {
			role = fantasy.MessageRoleAssistant
		}

		var content []fantasy.MessagePart
		content = append(content, fantasy.TextPart{
			Text: rapid.String().Draw(t, "messageText"),
		})

		// Optionally add image for user messages
		if role == fantasy.MessageRoleUser && rapid.Bool().Draw(t, "hasImage") {
			content = append(content, fantasy.FilePart{
				Data:      rapid.SliceOfN(rapid.Byte(), 1, 100).Draw(t, "imageData"),
				MediaType: rapid.SampledFrom([]string{"image/jpeg", "image/png", "image/gif", "image/webp"}).Draw(t, "imageType"),
			})
		}

		prompt = append(prompt, fantasy.Message{
			Role:    role,
			Content: content,
		})
	}

	// Generate inference parameters
	var maxTokens *int64
	if rapid.Bool().Draw(t, "hasMaxTokens") {
		val := rapid.Int64Range(1, 4096).Draw(t, "maxTokens")
		maxTokens = &val
	}

	var temperature *float64
	if rapid.Bool().Draw(t, "hasTemperature") {
		val := rapid.Float64Range(0.0, 1.0).Draw(t, "temperature")
		temperature = &val
	}

	var topP *float64
	if rapid.Bool().Draw(t, "hasTopP") {
		val := rapid.Float64Range(0.0, 1.0).Draw(t, "topP")
		topP = &val
	}

	var topK *int64
	if rapid.Bool().Draw(t, "hasTopK") {
		val := rapid.Int64Range(1, 500).Draw(t, "topK")
		topK = &val
	}

	return fantasy.Call{
		Prompt:          prompt,
		MaxOutputTokens: maxTokens,
		Temperature:     temperature,
		TopP:            topP,
		TopK:            topK,
	}
}

// validateConverseRequest validates that a Converse API request is properly formatted.
func validateConverseRequest(t *rapid.T, request *bedrockruntime.ConverseInput, warnings []fantasy.CallWarning, call fantasy.Call) {
	// Model ID must be set
	if request.ModelId == nil || *request.ModelId == "" {
		t.Fatalf("ModelId must be set")
	}

	// Messages must be present and non-empty
	if len(request.Messages) == 0 {
		t.Fatalf("Messages must not be empty")
	}

	// Validate message roles
	for i, msg := range request.Messages {
		if msg.Role != "user" && msg.Role != "assistant" {
			t.Fatalf("Message %d has invalid role: %s", i, msg.Role)
		}

		// Messages must have content
		if len(msg.Content) == 0 {
			t.Fatalf("Message %d has no content", i)
		}
	}

	// Validate inference configuration if parameters were provided
	if request.InferenceConfig != nil {
		if call.MaxOutputTokens != nil {
			if request.InferenceConfig.MaxTokens == nil {
				t.Fatalf("MaxTokens should be set when MaxOutputTokens is provided")
			}
			if *request.InferenceConfig.MaxTokens != int32(*call.MaxOutputTokens) {
				t.Fatalf("MaxTokens mismatch: expected %d, got %d", *call.MaxOutputTokens, *request.InferenceConfig.MaxTokens)
			}
		}

		if call.Temperature != nil {
			if request.InferenceConfig.Temperature == nil {
				t.Fatalf("Temperature should be set when Temperature is provided")
			}
			if *request.InferenceConfig.Temperature != float32(*call.Temperature) {
				t.Fatalf("Temperature mismatch: expected %f, got %f", *call.Temperature, *request.InferenceConfig.Temperature)
			}
		}

		if call.TopP != nil {
			if request.InferenceConfig.TopP == nil {
				t.Fatalf("TopP should be set when TopP is provided")
			}
			if *request.InferenceConfig.TopP != float32(*call.TopP) {
				t.Fatalf("TopP mismatch: expected %f, got %f", *call.TopP, *request.InferenceConfig.TopP)
			}
		}
	}

	// Validate top_k in additional fields if provided
	if call.TopK != nil {
		if request.AdditionalModelRequestFields == nil {
			t.Fatalf("AdditionalModelRequestFields should be set when TopK is provided")
		}
	}

	// System blocks should be present if system messages were in the prompt
	hasSystemMessage := false
	for _, msg := range call.Prompt {
		if msg.Role == fantasy.MessageRoleSystem {
			hasSystemMessage = true
			break
		}
	}

	if hasSystemMessage && len(request.System) == 0 {
		t.Fatalf("System blocks should be present when system messages are in the prompt")
	}
}

// Feature: amazon-nova-bedrock-support, Property 12: Parameter Preservation
// For any fantasy.Call with inference parameters (temperature, top_p, top_k, max_tokens,
// system prompt, multi-turn messages, image attachments), the converted Converse API
// request should include all provided parameters in the appropriate fields.
func TestProperty_ParameterPreservation(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a call with various parameters
		call := generateCallWithParameters(t)

		// Create a nova language model instance
		cfg, err := config.LoadDefaultConfig(context.Background())
		if err != nil {
			t.Skip("AWS configuration not available")
		}

		client := bedrockruntime.NewFromConfig(cfg)
		model := &novaLanguageModel{
			modelID:  "amazon.nova-pro-v1:0",
			provider: Name,
			client:   client,
			options:  options{},
		}

		// Convert to Converse API request
		request, _, err := model.prepareConverseRequest(call)
		if err != nil {
			t.Fatalf("prepareConverseRequest failed: %v", err)
		}

		// Verify all parameters are preserved
		verifyParameterPreservation(t, request, call)
	})
}

// generateCallWithParameters generates a fantasy.Call with various parameters for testing.
func generateCallWithParameters(t *rapid.T) fantasy.Call {
	var prompt fantasy.Prompt

	// Add system message if requested
	hasSystem := rapid.Bool().Draw(t, "hasSystem")
	if hasSystem {
		prompt = append(prompt, fantasy.Message{
			Role: fantasy.MessageRoleSystem,
			Content: []fantasy.MessagePart{
				fantasy.TextPart{
					Text: rapid.StringN(1, 100, -1).Draw(t, "systemPrompt"),
				},
			},
		})
	}

	// Add user message
	var userContent []fantasy.MessagePart
	userContent = append(userContent, fantasy.TextPart{
		Text: rapid.StringN(1, 100, -1).Draw(t, "userText"),
	})

	// Add image attachment if requested
	hasImage := rapid.Bool().Draw(t, "hasImage")
	if hasImage {
		prompt = append(prompt, fantasy.Message{
			Role: fantasy.MessageRoleUser,
			Content: append(userContent, fantasy.FilePart{
				Data:      rapid.SliceOfN(rapid.Byte(), 10, 100).Draw(t, "imageData"),
				MediaType: rapid.SampledFrom([]string{"image/jpeg", "image/png"}).Draw(t, "imageType"),
			}),
		})
	} else {
		prompt = append(prompt, fantasy.Message{
			Role:    fantasy.MessageRoleUser,
			Content: userContent,
		})
	}

	// Add assistant message for multi-turn
	hasMultiTurn := rapid.Bool().Draw(t, "hasMultiTurn")
	if hasMultiTurn {
		prompt = append(prompt, fantasy.Message{
			Role: fantasy.MessageRoleAssistant,
			Content: []fantasy.MessagePart{
				fantasy.TextPart{
					Text: rapid.StringN(1, 100, -1).Draw(t, "assistantText"),
				},
			},
		})

		// Add another user message
		prompt = append(prompt, fantasy.Message{
			Role: fantasy.MessageRoleUser,
			Content: []fantasy.MessagePart{
				fantasy.TextPart{
					Text: rapid.StringN(1, 100, -1).Draw(t, "userText2"),
				},
			},
		})
	}

	// Generate inference parameters
	maxTokens := rapid.Int64Range(1, 4096).Draw(t, "maxTokens")
	temperature := rapid.Float64Range(0.0, 1.0).Draw(t, "temperature")
	topP := rapid.Float64Range(0.0, 1.0).Draw(t, "topP")
	topK := rapid.Int64Range(1, 500).Draw(t, "topK")

	return fantasy.Call{
		Prompt:          prompt,
		MaxOutputTokens: &maxTokens,
		Temperature:     &temperature,
		TopP:            &topP,
		TopK:            &topK,
	}
}

// verifyParameterPreservation verifies that all parameters from the call are preserved in the request.
func verifyParameterPreservation(t *rapid.T, request *bedrockruntime.ConverseInput, call fantasy.Call) {
	// Verify max_tokens
	if call.MaxOutputTokens != nil {
		if request.InferenceConfig == nil || request.InferenceConfig.MaxTokens == nil {
			t.Fatalf("MaxTokens not preserved: expected %d, got nil", *call.MaxOutputTokens)
		}
		if *request.InferenceConfig.MaxTokens != int32(*call.MaxOutputTokens) {
			t.Fatalf("MaxTokens not preserved: expected %d, got %d", *call.MaxOutputTokens, *request.InferenceConfig.MaxTokens)
		}
	}

	// Verify temperature
	if call.Temperature != nil {
		if request.InferenceConfig == nil || request.InferenceConfig.Temperature == nil {
			t.Fatalf("Temperature not preserved: expected %f, got nil", *call.Temperature)
		}
		if *request.InferenceConfig.Temperature != float32(*call.Temperature) {
			t.Fatalf("Temperature not preserved: expected %f, got %f", *call.Temperature, *request.InferenceConfig.Temperature)
		}
	}

	// Verify top_p
	if call.TopP != nil {
		if request.InferenceConfig == nil || request.InferenceConfig.TopP == nil {
			t.Fatalf("TopP not preserved: expected %f, got nil", *call.TopP)
		}
		if *request.InferenceConfig.TopP != float32(*call.TopP) {
			t.Fatalf("TopP not preserved: expected %f, got %f", *call.TopP, *request.InferenceConfig.TopP)
		}
	}

	// Verify top_k (in additional fields)
	if call.TopK != nil {
		if request.AdditionalModelRequestFields == nil {
			t.Fatalf("TopK not preserved: AdditionalModelRequestFields is nil")
		}
	}

	// Verify system prompt
	hasSystemMessage := false
	for _, msg := range call.Prompt {
		if msg.Role == fantasy.MessageRoleSystem {
			hasSystemMessage = true
			break
		}
	}
	if hasSystemMessage {
		if len(request.System) == 0 {
			t.Fatalf("System prompt not preserved")
		}
	}

	// Verify multi-turn conversations (message count)
	userAssistantCount := 0
	for _, msg := range call.Prompt {
		if msg.Role == fantasy.MessageRoleUser || msg.Role == fantasy.MessageRoleAssistant {
			userAssistantCount++
		}
	}
	if len(request.Messages) != userAssistantCount {
		t.Fatalf("Multi-turn messages not preserved: expected %d messages, got %d", userAssistantCount, len(request.Messages))
	}

	// Verify image attachments
	hasImage := false
	for _, msg := range call.Prompt {
		for _, part := range msg.Content {
			if part.GetType() == fantasy.ContentTypeFile {
				if filePart, ok := fantasy.AsMessagePart[fantasy.FilePart](part); ok {
					if isImageMediaType(filePart.MediaType) {
						hasImage = true
						break
					}
				}
			}
		}
		if hasImage {
			break
		}
	}

	if hasImage {
		// Check that at least one message has an image content block
		foundImage := false
		for _, msg := range request.Messages {
			for _, block := range msg.Content {
				if _, ok := block.(*types.ContentBlockMemberImage); ok {
					foundImage = true
					break
				}
			}
			if foundImage {
				break
			}
		}
		if !foundImage {
			t.Fatalf("Image attachment not preserved in request")
		}
	}
}

// Feature: amazon-nova-bedrock-support, Property 7: Response Parsing Success
// For any valid Converse API response, parsing it into a fantasy.Response should succeed
// and produce a response with valid content, usage statistics, and finish reason.
func TestProperty_ResponseParsingSuccess(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a valid Converse API response
		output := generateValidConverseOutput(t)

		// Create a nova language model instance
		cfg, err := config.LoadDefaultConfig(context.Background())
		if err != nil {
			t.Skip("AWS configuration not available")
		}

		client := bedrockruntime.NewFromConfig(cfg)
		model := &novaLanguageModel{
			modelID:  "amazon.nova-pro-v1:0",
			provider: Name,
			client:   client,
			options:  options{},
		}

		// Convert to fantasy.Response
		response, err := model.convertConverseResponse(output, nil)

		// The conversion should succeed
		if err != nil {
			t.Fatalf("convertConverseResponse failed: %v", err)
		}

		// Validate the response
		validateFantasyResponse(t, response, output)
	})
}

// generateValidConverseOutput generates a valid Converse API output for property testing.
func generateValidConverseOutput(t *rapid.T) *bedrockruntime.ConverseOutput {
	// Generate content blocks
	numBlocks := rapid.IntRange(1, 5).Draw(t, "numBlocks")
	var contentBlocks []types.ContentBlock

	for i := 0; i < numBlocks; i++ {
		blockType := rapid.IntRange(0, 2).Draw(t, "blockType")
		switch blockType {
		case 0:
			// Text block
			contentBlocks = append(contentBlocks, &types.ContentBlockMemberText{
				Value: rapid.String().Draw(t, "textContent"),
			})
		case 1:
			// Tool use block
			toolID := rapid.String().Draw(t, "toolID")
			toolName := rapid.String().Draw(t, "toolName")
			contentBlocks = append(contentBlocks, &types.ContentBlockMemberToolUse{
				Value: types.ToolUseBlock{
					ToolUseId: &toolID,
					Name:      &toolName,
					Input:     nil, // simplified for testing
				},
			})
		case 2:
			// Image block
			imageData := rapid.SliceOfN(rapid.Byte(), 1, 100).Draw(t, "imageData")
			format := rapid.SampledFrom([]types.ImageFormat{
				types.ImageFormatJpeg,
				types.ImageFormatPng,
				types.ImageFormatGif,
				types.ImageFormatWebp,
			}).Draw(t, "imageFormat")
			contentBlocks = append(contentBlocks, &types.ContentBlockMemberImage{
				Value: types.ImageBlock{
					Format: format,
					Source: &types.ImageSourceMemberBytes{
						Value: imageData,
					},
				},
			})
		}
	}

	// Generate usage statistics
	inputTokens := int32(rapid.IntRange(1, 10000).Draw(t, "inputTokens"))
	outputTokens := int32(rapid.IntRange(1, 10000).Draw(t, "outputTokens"))
	totalTokens := inputTokens + outputTokens

	// Generate stop reason
	stopReason := rapid.SampledFrom([]types.StopReason{
		types.StopReasonEndTurn,
		types.StopReasonMaxTokens,
		types.StopReasonStopSequence,
		types.StopReasonToolUse,
		types.StopReasonContentFiltered,
	}).Draw(t, "stopReason")

	return &bedrockruntime.ConverseOutput{
		Output: &types.ConverseOutputMemberMessage{
			Value: types.Message{
				Role:    types.ConversationRoleAssistant,
				Content: contentBlocks,
			},
		},
		Usage: &types.TokenUsage{
			InputTokens:  &inputTokens,
			OutputTokens: &outputTokens,
			TotalTokens:  &totalTokens,
		},
		StopReason: stopReason,
	}
}

// validateFantasyResponse validates that a fantasy.Response is properly formatted.
func validateFantasyResponse(t *rapid.T, response *fantasy.Response, output *bedrockruntime.ConverseOutput) {
	// Response must not be nil
	if response == nil {
		t.Fatalf("Response is nil")
	}

	// Content must be present
	if len(response.Content) == 0 {
		t.Fatalf("Response content is empty")
	}

	// Usage statistics must be valid
	if output.Usage != nil {
		if output.Usage.InputTokens != nil && response.Usage.InputTokens != int64(*output.Usage.InputTokens) {
			t.Fatalf("InputTokens mismatch: expected %d, got %d", *output.Usage.InputTokens, response.Usage.InputTokens)
		}
		if output.Usage.OutputTokens != nil && response.Usage.OutputTokens != int64(*output.Usage.OutputTokens) {
			t.Fatalf("OutputTokens mismatch: expected %d, got %d", *output.Usage.OutputTokens, response.Usage.OutputTokens)
		}
		if output.Usage.TotalTokens != nil && response.Usage.TotalTokens != int64(*output.Usage.TotalTokens) {
			t.Fatalf("TotalTokens mismatch: expected %d, got %d", *output.Usage.TotalTokens, response.Usage.TotalTokens)
		}
	}

	// Finish reason must be valid (not empty)
	if response.FinishReason == "" {
		t.Fatalf("FinishReason is empty")
	}

	// Verify finish reason mapping
	expectedFinishReason := convertStopReason(output.StopReason)
	if response.FinishReason != expectedFinishReason {
		t.Fatalf("FinishReason mismatch: expected %s, got %s", expectedFinishReason, response.FinishReason)
	}
}

// Feature: amazon-nova-bedrock-support, Property 10: Finish Reason Mapping Completeness
// For all possible Converse API stop reasons ("end_turn", "max_tokens", "stop_sequence",
// "tool_use", "content_filtered"), there should be a corresponding fantasy.FinishReason value.
func TestProperty_FinishReasonMappingCompleteness(t *testing.T) {
	// Test all known stop reasons
	allStopReasons := []types.StopReason{
		types.StopReasonEndTurn,
		types.StopReasonMaxTokens,
		types.StopReasonStopSequence,
		types.StopReasonToolUse,
		types.StopReasonContentFiltered,
	}

	for _, stopReason := range allStopReasons {
		t.Run(string(stopReason), func(t *testing.T) {
			finishReason := convertStopReason(stopReason)

			// The finish reason must not be empty
			if finishReason == "" {
				t.Fatalf("convertStopReason returned empty string for stop reason: %s", stopReason)
			}

			// The finish reason must be a valid fantasy.FinishReason
			validFinishReasons := []fantasy.FinishReason{
				fantasy.FinishReasonStop,
				fantasy.FinishReasonLength,
				fantasy.FinishReasonContentFilter,
				fantasy.FinishReasonToolCalls,
				fantasy.FinishReasonError,
				fantasy.FinishReasonOther,
				fantasy.FinishReasonUnknown,
			}

			isValid := false
			for _, valid := range validFinishReasons {
				if finishReason == valid {
					isValid = true
					break
				}
			}

			if !isValid {
				t.Fatalf("convertStopReason returned invalid finish reason: %s for stop reason: %s", finishReason, stopReason)
			}

			// Verify specific mappings
			switch stopReason {
			case types.StopReasonEndTurn:
				if finishReason != fantasy.FinishReasonStop {
					t.Fatalf("Expected FinishReasonStop for EndTurn, got %s", finishReason)
				}
			case types.StopReasonMaxTokens:
				if finishReason != fantasy.FinishReasonLength {
					t.Fatalf("Expected FinishReasonLength for MaxTokens, got %s", finishReason)
				}
			case types.StopReasonStopSequence:
				if finishReason != fantasy.FinishReasonStop {
					t.Fatalf("Expected FinishReasonStop for StopSequence, got %s", finishReason)
				}
			case types.StopReasonToolUse:
				if finishReason != fantasy.FinishReasonToolCalls {
					t.Fatalf("Expected FinishReasonToolCalls for ToolUse, got %s", finishReason)
				}
			case types.StopReasonContentFiltered:
				if finishReason != fantasy.FinishReasonContentFilter {
					t.Fatalf("Expected FinishReasonContentFilter for ContentFiltered, got %s", finishReason)
				}
			}
		})
	}

	// Test unknown stop reason
	t.Run("unknown", func(t *testing.T) {
		unknownStopReason := types.StopReason("unknown_reason")
		finishReason := convertStopReason(unknownStopReason)

		if finishReason != fantasy.FinishReasonUnknown {
			t.Fatalf("Expected FinishReasonUnknown for unknown stop reason, got %s", finishReason)
		}
	})
}

// Feature: amazon-nova-bedrock-support, Property 11: Message Format Round-Trip
// For any fantasy message (with text, images, tool calls, or tool results), converting it
// to Converse API format and then back to fantasy format should preserve the essential
// content and structure.
func TestProperty_MessageFormatRoundTrip(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		// Generate a fantasy message with various content types
		message := generateFantasyMessage(t)

		// Create a nova language model instance
		cfg, err := config.LoadDefaultConfig(context.Background())
		if err != nil {
			t.Skip("AWS configuration not available")
		}

		client := bedrockruntime.NewFromConfig(cfg)
		model := &novaLanguageModel{
			modelID:  "amazon.nova-pro-v1:0",
			provider: Name,
			client:   client,
			options:  options{},
		}

		// Convert fantasy message to Converse API format
		call := fantasy.Call{
			Prompt: fantasy.Prompt{message},
		}

		request, _, err := model.prepareConverseRequest(call)
		if err != nil {
			t.Fatalf("prepareConverseRequest failed: %v", err)
		}

		// Extract the converted message content blocks
		if len(request.Messages) == 0 {
			t.Fatalf("No messages in request")
		}

		converseMessage := request.Messages[0]

		// Convert back to fantasy format by simulating a response
		var fantasyContent fantasy.ResponseContent
		for _, block := range converseMessage.Content {
			content, err := convertContentBlock(block)
			if err != nil {
				t.Fatalf("convertContentBlock failed: %v", err)
			}
			if content != nil {
				fantasyContent = append(fantasyContent, content)
			}
		}

		// Verify round-trip preservation
		verifyRoundTripPreservation(t, message, fantasyContent)
	})
}

// generateFantasyMessage generates a fantasy message with various content types.
func generateFantasyMessage(t *rapid.T) fantasy.Message {
	role := rapid.SampledFrom([]fantasy.MessageRole{
		fantasy.MessageRoleUser,
		fantasy.MessageRoleAssistant,
	}).Draw(t, "role")

	var content []fantasy.MessagePart

	// Always include text
	content = append(content, fantasy.TextPart{
		Text: rapid.StringN(1, 100, -1).Draw(t, "text"),
	})

	// Optionally add image (only for user messages)
	if role == fantasy.MessageRoleUser && rapid.Bool().Draw(t, "hasImage") {
		content = append(content, fantasy.FilePart{
			Data:      rapid.SliceOfN(rapid.Byte(), 10, 100).Draw(t, "imageData"),
			MediaType: rapid.SampledFrom([]string{"image/jpeg", "image/png", "image/gif", "image/webp"}).Draw(t, "imageType"),
		})
	}

	// Optionally add tool call (only for assistant messages)
	if role == fantasy.MessageRoleAssistant && rapid.Bool().Draw(t, "hasToolCall") {
		toolInput := map[string]interface{}{
			"param": rapid.String().Draw(t, "toolParam"),
		}
		toolInputJSON, _ := json.Marshal(toolInput)

		content = append(content, fantasy.ToolCallPart{
			ToolCallID: rapid.String().Draw(t, "toolCallID"),
			ToolName:   rapid.String().Draw(t, "toolName"),
			Input:      string(toolInputJSON),
		})
	}

	return fantasy.Message{
		Role:    role,
		Content: content,
	}
}

// verifyRoundTripPreservation verifies that essential content is preserved in round-trip conversion.
func verifyRoundTripPreservation(t *rapid.T, original fantasy.Message, converted fantasy.ResponseContent) {
	// Count content types in original
	originalTextCount := 0
	originalImageCount := 0
	originalToolCallCount := 0

	for _, part := range original.Content {
		switch part.GetType() {
		case fantasy.ContentTypeText:
			originalTextCount++
		case fantasy.ContentTypeFile:
			if filePart, ok := fantasy.AsMessagePart[fantasy.FilePart](part); ok {
				if isImageMediaType(filePart.MediaType) {
					originalImageCount++
				}
			}
		case fantasy.ContentTypeToolCall:
			originalToolCallCount++
		}
	}

	// Count content types in converted
	convertedTextCount := 0
	convertedImageCount := 0
	convertedToolCallCount := 0

	for _, content := range converted {
		switch content.GetType() {
		case fantasy.ContentTypeText:
			convertedTextCount++
		case fantasy.ContentTypeFile:
			convertedImageCount++
		case fantasy.ContentTypeToolCall:
			convertedToolCallCount++
		}
	}

	// Verify counts match
	if originalTextCount != convertedTextCount {
		t.Fatalf("Text count mismatch: original %d, converted %d", originalTextCount, convertedTextCount)
	}

	if originalImageCount != convertedImageCount {
		t.Fatalf("Image count mismatch: original %d, converted %d", originalImageCount, convertedImageCount)
	}

	if originalToolCallCount != convertedToolCallCount {
		t.Fatalf("Tool call count mismatch: original %d, converted %d", originalToolCallCount, convertedToolCallCount)
	}

	// Verify text content is preserved
	if originalTextCount > 0 {
		originalText := ""
		for _, part := range original.Content {
			if part.GetType() == fantasy.ContentTypeText {
				if textPart, ok := fantasy.AsMessagePart[fantasy.TextPart](part); ok {
					originalText = textPart.Text
					break
				}
			}
		}

		convertedText := converted.Text()
		if originalText != convertedText {
			t.Fatalf("Text content not preserved: original '%s', converted '%s'", originalText, convertedText)
		}
	}

	// Verify tool call names are preserved
	if originalToolCallCount > 0 {
		originalToolNames := make(map[string]bool)
		for _, part := range original.Content {
			if part.GetType() == fantasy.ContentTypeToolCall {
				if toolCallPart, ok := fantasy.AsMessagePart[fantasy.ToolCallPart](part); ok {
					originalToolNames[toolCallPart.ToolName] = true
				}
			}
		}

		convertedToolCalls := converted.ToolCalls()
		for _, toolCall := range convertedToolCalls {
			if !originalToolNames[toolCall.ToolName] {
				t.Fatalf("Tool call name not preserved: %s", toolCall.ToolName)
			}
		}
	}
}
