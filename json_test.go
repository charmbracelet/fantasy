package fantasy

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"
)

func TestMessageJSONSerialization(t *testing.T) {
	tests := []struct {
		name    string
		message Message
	}{
		{
			name: "simple text message",
			message: Message{
				Role: MessageRoleUser,
				Content: []MessagePart{
					TextPart{Text: "Hello, world!"},
				},
			},
		},
		{
			name: "message with multiple text parts",
			message: Message{
				Role: MessageRoleAssistant,
				Content: []MessagePart{
					TextPart{Text: "First part"},
					TextPart{Text: "Second part"},
					TextPart{Text: "Third part"},
				},
			},
		},
		{
			name: "message with reasoning part",
			message: Message{
				Role: MessageRoleAssistant,
				Content: []MessagePart{
					ReasoningPart{Text: "Let me think about this..."},
					TextPart{Text: "Here's my answer"},
				},
			},
		},
		{
			name: "message with file part",
			message: Message{
				Role: MessageRoleUser,
				Content: []MessagePart{
					TextPart{Text: "Here's an image:"},
					FilePart{
						Filename:  "test.png",
						Data:      []byte{0x89, 0x50, 0x4E, 0x47}, // PNG header
						MediaType: "image/png",
					},
				},
			},
		},
		{
			name: "message with tool call",
			message: Message{
				Role: MessageRoleAssistant,
				Content: []MessagePart{
					ToolCallPart{
						ToolCallID:       "call_123",
						ToolName:         "get_weather",
						Input:            `{"location": "San Francisco"}`,
						ProviderExecuted: false,
					},
				},
			},
		},
		{
			name: "message with tool result - text output",
			message: Message{
				Role: MessageRoleTool,
				Content: []MessagePart{
					ToolResultPart{
						ToolCallID: "call_123",
						Output: ToolResultOutputContentText{
							Text: "The weather is sunny, 72°F",
						},
					},
				},
			},
		},
		{
			name: "message with tool result - error output",
			message: Message{
				Role: MessageRoleTool,
				Content: []MessagePart{
					ToolResultPart{
						ToolCallID: "call_456",
						Output: ToolResultOutputContentError{
							Error: errors.New("API rate limit exceeded"),
						},
					},
				},
			},
		},
		{
			name: "message with tool result - media output",
			message: Message{
				Role: MessageRoleTool,
				Content: []MessagePart{
					ToolResultPart{
						ToolCallID: "call_789",
						Output: ToolResultOutputContentMedia{
							Data:      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
							MediaType: "image/png",
						},
					},
				},
			},
		},
		{
			name: "complex message with mixed content",
			message: Message{
				Role: MessageRoleAssistant,
				Content: []MessagePart{
					TextPart{Text: "I'll analyze this image and call some tools."},
					ReasoningPart{Text: "First, I need to identify the objects..."},
					ToolCallPart{
						ToolCallID:       "call_001",
						ToolName:         "analyze_image",
						Input:            `{"image_id": "img_123"}`,
						ProviderExecuted: false,
					},
					ToolCallPart{
						ToolCallID:       "call_002",
						ToolName:         "get_context",
						Input:            `{"query": "similar images"}`,
						ProviderExecuted: true,
					},
				},
			},
		},
		{
			name: "system message",
			message: Message{
				Role: MessageRoleSystem,
				Content: []MessagePart{
					TextPart{Text: "You are a helpful assistant."},
				},
			},
		},
		{
			name: "empty content",
			message: Message{
				Role:    MessageRoleUser,
				Content: []MessagePart{},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Marshal the message
			data, err := json.Marshal(tt.message)
			if err != nil {
				t.Fatalf("failed to marshal message: %v", err)
			}

			// Unmarshal back
			var decoded Message
			err = json.Unmarshal(data, &decoded)
			if err != nil {
				t.Fatalf("failed to unmarshal message: %v", err)
			}

			// Compare roles
			if decoded.Role != tt.message.Role {
				t.Errorf("role mismatch: got %v, want %v", decoded.Role, tt.message.Role)
			}

			// Compare content length
			if len(decoded.Content) != len(tt.message.Content) {
				t.Fatalf("content length mismatch: got %d, want %d", len(decoded.Content), len(tt.message.Content))
			}

			// Compare each content part
			for i := range tt.message.Content {
				original := tt.message.Content[i]
				decodedPart := decoded.Content[i]

				if original.GetType() != decodedPart.GetType() {
					t.Errorf("content[%d] type mismatch: got %v, want %v", i, decodedPart.GetType(), original.GetType())
					continue
				}

				compareMessagePart(t, i, original, decodedPart)
			}
		})
	}
}

func compareMessagePart(t *testing.T, index int, original, decoded MessagePart) {
	switch original.GetType() {
	case ContentTypeText:
		orig := original.(TextPart)
		dec := decoded.(TextPart)
		if orig.Text != dec.Text {
			t.Errorf("content[%d] text mismatch: got %q, want %q", index, dec.Text, orig.Text)
		}

	case ContentTypeReasoning:
		orig := original.(ReasoningPart)
		dec := decoded.(ReasoningPart)
		if orig.Text != dec.Text {
			t.Errorf("content[%d] reasoning text mismatch: got %q, want %q", index, dec.Text, orig.Text)
		}

	case ContentTypeFile:
		orig := original.(FilePart)
		dec := decoded.(FilePart)
		if orig.Filename != dec.Filename {
			t.Errorf("content[%d] filename mismatch: got %q, want %q", index, dec.Filename, orig.Filename)
		}
		if orig.MediaType != dec.MediaType {
			t.Errorf("content[%d] media type mismatch: got %q, want %q", index, dec.MediaType, orig.MediaType)
		}
		if !reflect.DeepEqual(orig.Data, dec.Data) {
			t.Errorf("content[%d] file data mismatch", index)
		}

	case ContentTypeToolCall:
		orig := original.(ToolCallPart)
		dec := decoded.(ToolCallPart)
		if orig.ToolCallID != dec.ToolCallID {
			t.Errorf("content[%d] tool call id mismatch: got %q, want %q", index, dec.ToolCallID, orig.ToolCallID)
		}
		if orig.ToolName != dec.ToolName {
			t.Errorf("content[%d] tool name mismatch: got %q, want %q", index, dec.ToolName, orig.ToolName)
		}
		if orig.Input != dec.Input {
			t.Errorf("content[%d] tool input mismatch: got %q, want %q", index, dec.Input, orig.Input)
		}
		if orig.ProviderExecuted != dec.ProviderExecuted {
			t.Errorf("content[%d] provider executed mismatch: got %v, want %v", index, dec.ProviderExecuted, orig.ProviderExecuted)
		}

	case ContentTypeToolResult:
		orig := original.(ToolResultPart)
		dec := decoded.(ToolResultPart)
		if orig.ToolCallID != dec.ToolCallID {
			t.Errorf("content[%d] tool result call id mismatch: got %q, want %q", index, dec.ToolCallID, orig.ToolCallID)
		}
		compareToolResultOutput(t, index, orig.Output, dec.Output)
	}
}

func compareToolResultOutput(t *testing.T, index int, original, decoded ToolResultOutputContent) {
	if original.GetType() != decoded.GetType() {
		t.Errorf("content[%d] tool result output type mismatch: got %v, want %v", index, decoded.GetType(), original.GetType())
		return
	}

	switch original.GetType() {
	case ToolResultContentTypeText:
		orig := original.(ToolResultOutputContentText)
		dec := decoded.(ToolResultOutputContentText)
		if orig.Text != dec.Text {
			t.Errorf("content[%d] tool result text mismatch: got %q, want %q", index, dec.Text, orig.Text)
		}

	case ToolResultContentTypeError:
		orig := original.(ToolResultOutputContentError)
		dec := decoded.(ToolResultOutputContentError)
		if orig.Error.Error() != dec.Error.Error() {
			t.Errorf("content[%d] tool result error mismatch: got %q, want %q", index, dec.Error.Error(), orig.Error.Error())
		}

	case ToolResultContentTypeMedia:
		orig := original.(ToolResultOutputContentMedia)
		dec := decoded.(ToolResultOutputContentMedia)
		if orig.Data != dec.Data {
			t.Errorf("content[%d] tool result media data mismatch", index)
		}
		if orig.MediaType != dec.MediaType {
			t.Errorf("content[%d] tool result media type mismatch: got %q, want %q", index, dec.MediaType, orig.MediaType)
		}
	}
}

func TestHelperFunctions(t *testing.T) {
	t.Run("NewUserMessage - text only", func(t *testing.T) {
		msg := NewUserMessage("Hello")

		data, err := json.Marshal(msg)
		if err != nil {
			t.Fatalf("failed to marshal: %v", err)
		}

		var decoded Message
		if err := json.Unmarshal(data, &decoded); err != nil {
			t.Fatalf("failed to unmarshal: %v", err)
		}

		if decoded.Role != MessageRoleUser {
			t.Errorf("role mismatch: got %v, want %v", decoded.Role, MessageRoleUser)
		}

		if len(decoded.Content) != 1 {
			t.Fatalf("expected 1 content part, got %d", len(decoded.Content))
		}

		textPart := decoded.Content[0].(TextPart)
		if textPart.Text != "Hello" {
			t.Errorf("text mismatch: got %q, want %q", textPart.Text, "Hello")
		}
	})

	t.Run("NewUserMessage - with files", func(t *testing.T) {
		msg := NewUserMessage("Check this image",
			FilePart{
				Filename:  "image1.jpg",
				Data:      []byte{0xFF, 0xD8, 0xFF},
				MediaType: "image/jpeg",
			},
			FilePart{
				Filename:  "image2.png",
				Data:      []byte{0x89, 0x50, 0x4E, 0x47},
				MediaType: "image/png",
			},
		)

		data, err := json.Marshal(msg)
		if err != nil {
			t.Fatalf("failed to marshal: %v", err)
		}

		var decoded Message
		if err := json.Unmarshal(data, &decoded); err != nil {
			t.Fatalf("failed to unmarshal: %v", err)
		}

		if len(decoded.Content) != 3 {
			t.Fatalf("expected 3 content parts, got %d", len(decoded.Content))
		}

		// Check text part
		textPart := decoded.Content[0].(TextPart)
		if textPart.Text != "Check this image" {
			t.Errorf("text mismatch: got %q, want %q", textPart.Text, "Check this image")
		}

		// Check first file
		file1 := decoded.Content[1].(FilePart)
		if file1.Filename != "image1.jpg" {
			t.Errorf("file1 name mismatch: got %q, want %q", file1.Filename, "image1.jpg")
		}

		// Check second file
		file2 := decoded.Content[2].(FilePart)
		if file2.Filename != "image2.png" {
			t.Errorf("file2 name mismatch: got %q, want %q", file2.Filename, "image2.png")
		}
	})

	t.Run("NewSystemMessage - single prompt", func(t *testing.T) {
		msg := NewSystemMessage("You are a helpful assistant.")

		data, err := json.Marshal(msg)
		if err != nil {
			t.Fatalf("failed to marshal: %v", err)
		}

		var decoded Message
		if err := json.Unmarshal(data, &decoded); err != nil {
			t.Fatalf("failed to unmarshal: %v", err)
		}

		if decoded.Role != MessageRoleSystem {
			t.Errorf("role mismatch: got %v, want %v", decoded.Role, MessageRoleSystem)
		}

		if len(decoded.Content) != 1 {
			t.Fatalf("expected 1 content part, got %d", len(decoded.Content))
		}

		textPart := decoded.Content[0].(TextPart)
		if textPart.Text != "You are a helpful assistant." {
			t.Errorf("text mismatch: got %q, want %q", textPart.Text, "You are a helpful assistant.")
		}
	})

	t.Run("NewSystemMessage - multiple prompts", func(t *testing.T) {
		msg := NewSystemMessage("First instruction", "Second instruction", "Third instruction")

		data, err := json.Marshal(msg)
		if err != nil {
			t.Fatalf("failed to marshal: %v", err)
		}

		var decoded Message
		if err := json.Unmarshal(data, &decoded); err != nil {
			t.Fatalf("failed to unmarshal: %v", err)
		}

		if len(decoded.Content) != 3 {
			t.Fatalf("expected 3 content parts, got %d", len(decoded.Content))
		}

		expected := []string{"First instruction", "Second instruction", "Third instruction"}
		for i, exp := range expected {
			textPart := decoded.Content[i].(TextPart)
			if textPart.Text != exp {
				t.Errorf("content[%d] text mismatch: got %q, want %q", i, textPart.Text, exp)
			}
		}
	})
}

func TestEdgeCases(t *testing.T) {
	t.Run("empty text part", func(t *testing.T) {
		msg := Message{
			Role: MessageRoleUser,
			Content: []MessagePart{
				TextPart{Text: ""},
			},
		}

		data, err := json.Marshal(msg)
		if err != nil {
			t.Fatalf("failed to marshal: %v", err)
		}

		var decoded Message
		if err := json.Unmarshal(data, &decoded); err != nil {
			t.Fatalf("failed to unmarshal: %v", err)
		}

		textPart := decoded.Content[0].(TextPart)
		if textPart.Text != "" {
			t.Errorf("expected empty text, got %q", textPart.Text)
		}
	})

	t.Run("nil error in tool result", func(t *testing.T) {
		msg := Message{
			Role: MessageRoleTool,
			Content: []MessagePart{
				ToolResultPart{
					ToolCallID: "call_123",
					Output: ToolResultOutputContentError{
						Error: nil,
					},
				},
			},
		}

		data, err := json.Marshal(msg)
		if err != nil {
			t.Fatalf("failed to marshal: %v", err)
		}

		var decoded Message
		if err := json.Unmarshal(data, &decoded); err != nil {
			t.Fatalf("failed to unmarshal: %v", err)
		}

		toolResult := decoded.Content[0].(ToolResultPart)
		errorOutput := toolResult.Output.(ToolResultOutputContentError)
		if errorOutput.Error != nil {
			t.Errorf("expected nil error, got %v", errorOutput.Error)
		}
	})

	t.Run("empty file data", func(t *testing.T) {
		msg := Message{
			Role: MessageRoleUser,
			Content: []MessagePart{
				FilePart{
					Filename:  "empty.txt",
					Data:      []byte{},
					MediaType: "text/plain",
				},
			},
		}

		data, err := json.Marshal(msg)
		if err != nil {
			t.Fatalf("failed to marshal: %v", err)
		}

		var decoded Message
		if err := json.Unmarshal(data, &decoded); err != nil {
			t.Fatalf("failed to unmarshal: %v", err)
		}

		filePart := decoded.Content[0].(FilePart)
		if len(filePart.Data) != 0 {
			t.Errorf("expected empty data, got %d bytes", len(filePart.Data))
		}
	})

	t.Run("unicode in text", func(t *testing.T) {
		msg := Message{
			Role: MessageRoleUser,
			Content: []MessagePart{
				TextPart{Text: "Hello 世界! 🌍 Привет"},
			},
		}

		data, err := json.Marshal(msg)
		if err != nil {
			t.Fatalf("failed to marshal: %v", err)
		}

		var decoded Message
		if err := json.Unmarshal(data, &decoded); err != nil {
			t.Fatalf("failed to unmarshal: %v", err)
		}

		textPart := decoded.Content[0].(TextPart)
		if textPart.Text != "Hello 世界! 🌍 Привет" {
			t.Errorf("unicode text mismatch: got %q, want %q", textPart.Text, "Hello 世界! 🌍 Привет")
		}
	})
}

func TestInvalidJSONHandling(t *testing.T) {
	t.Run("unknown message part type", func(t *testing.T) {
		invalidJSON := `{
			"role": "user",
			"content": [
				{
					"type": "unknown-type",
					"data": {}
				}
			],
			"provider_options": null
		}`

		var msg Message
		err := json.Unmarshal([]byte(invalidJSON), &msg)
		if err == nil {
			t.Error("expected error for unknown message part type, got nil")
		}
	})

	t.Run("unknown tool result output type", func(t *testing.T) {
		invalidJSON := `{
			"role": "tool",
			"content": [
				{
					"type": "tool-result",
					"data": {
						"tool_call_id": "call_123",
						"output": {
							"type": "unknown-output-type",
							"data": {}
						},
						"provider_options": null
					}
				}
			],
			"provider_options": null
		}`

		var msg Message
		err := json.Unmarshal([]byte(invalidJSON), &msg)
		if err == nil {
			t.Error("expected error for unknown tool result output type, got nil")
		}
	})

	t.Run("malformed JSON", func(t *testing.T) {
		invalidJSON := `{"role": "user", "content": [`

		var msg Message
		err := json.Unmarshal([]byte(invalidJSON), &msg)
		if err == nil {
			t.Error("expected error for malformed JSON, got nil")
		}
	})
}

// Mock provider data for testing provider options
type mockProviderData struct {
	Key string `json:"key"`
}

func (m mockProviderData) Options()     {}
func (m mockProviderData) Type() string { return "mock" }
func (m mockProviderData) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type string `json:"type"`
		mockProviderData
	}{
		Type:             "mock",
		mockProviderData: m,
	})
}

func (m *mockProviderData) UnmarshalJSON(data []byte) error {
	var aux struct {
		Type string `json:"type"`
		mockProviderData
	}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	*m = aux.mockProviderData
	return nil
}

func TestPromptSerialization(t *testing.T) {
	t.Run("serialize prompt (message slice)", func(t *testing.T) {
		prompt := Prompt{
			NewSystemMessage("You are helpful"),
			NewUserMessage("Hello"),
			Message{
				Role: MessageRoleAssistant,
				Content: []MessagePart{
					TextPart{Text: "Hi there!"},
				},
			},
		}

		data, err := json.Marshal(prompt)
		if err != nil {
			t.Fatalf("failed to marshal prompt: %v", err)
		}

		var decoded Prompt
		if err := json.Unmarshal(data, &decoded); err != nil {
			t.Fatalf("failed to unmarshal prompt: %v", err)
		}

		if len(decoded) != 3 {
			t.Fatalf("expected 3 messages, got %d", len(decoded))
		}

		if decoded[0].Role != MessageRoleSystem {
			t.Errorf("message 0 role mismatch: got %v, want %v", decoded[0].Role, MessageRoleSystem)
		}

		if decoded[1].Role != MessageRoleUser {
			t.Errorf("message 1 role mismatch: got %v, want %v", decoded[1].Role, MessageRoleUser)
		}

		if decoded[2].Role != MessageRoleAssistant {
			t.Errorf("message 2 role mismatch: got %v, want %v", decoded[2].Role, MessageRoleAssistant)
		}
	})
}

func TestStreamPartErrorSerialization(t *testing.T) {
	t.Run("stream part with ProviderError containing OpenAI API error", func(t *testing.T) {
		// Create a mock OpenAI API error
		openaiErr := errors.New("invalid_api_key: Incorrect API key provided")

		// Wrap in ProviderError
		providerErr := &ProviderError{
			Title:       "unauthorized",
			Message:     "Incorrect API key provided",
			Cause:       openaiErr,
			URL:         "https://api.openai.com/v1/chat/completions",
			StatusCode:  401,
			RequestBody: []byte(`{"model":"gpt-4","messages":[]}`),
			ResponseHeaders: map[string]string{
				"content-type": "application/json",
			},
			ResponseBody: []byte(`{"error":{"message":"Incorrect API key provided","type":"invalid_request_error"}}`),
		}

		// Create StreamPart with error
		streamPart := StreamPart{
			Type:  StreamPartTypeError,
			Error: providerErr,
		}

		// Marshal the stream part
		data, err := json.Marshal(streamPart)
		if err != nil {
			t.Fatalf("failed to marshal stream part: %v", err)
		}

		// Unmarshal back
		var decoded StreamPart
		err = json.Unmarshal(data, &decoded)
		if err != nil {
			t.Fatalf("failed to unmarshal stream part: %v", err)
		}

		// Verify the stream part type
		if decoded.Type != StreamPartTypeError {
			t.Errorf("type mismatch: got %v, want %v", decoded.Type, StreamPartTypeError)
		}

		// Verify error exists
		if decoded.Error == nil {
			t.Fatal("expected error to be present, got nil")
		}

		// Verify error message
		expectedMsg := "unauthorized: Incorrect API key provided"
		if decoded.Error.Error() != expectedMsg {
			t.Errorf("error message mismatch: got %q, want %q", decoded.Error.Error(), expectedMsg)
		}
	})

	t.Run("unmarshal stream part with error from JSON", func(t *testing.T) {
		// JSON representing a StreamPart with an error
		jsonData := `{
			"type": "error",
			"error": "unauthorized: Incorrect API key provided",
			"id": "",
			"tool_call_name": "",
			"tool_call_input": "",
			"delta": "",
			"provider_executed": false,
			"usage": {
				"input_tokens": 0,
				"output_tokens": 0,
				"total_tokens": 0,
				"reasoning_tokens": 0,
				"cache_creation_tokens": 0,
				"cache_read_tokens": 0
			},
			"finish_reason": "",
			"warnings": null,
			"source_type": "",
			"url": "",
			"title": "",
			"provider_metadata": null
		}`

		var streamPart StreamPart
		err := json.Unmarshal([]byte(jsonData), &streamPart)
		if err != nil {
			t.Fatalf("failed to unmarshal stream part: %v", err)
		}

		// Verify the stream part type
		if streamPart.Type != StreamPartTypeError {
			t.Errorf("type mismatch: got %v, want %v", streamPart.Type, StreamPartTypeError)
		}

		// Verify error exists
		if streamPart.Error == nil {
			t.Fatal("expected error to be present, got nil")
		}

		// Verify error message
		expectedMsg := "unauthorized: Incorrect API key provided"
		if streamPart.Error.Error() != expectedMsg {
			t.Errorf("error message mismatch: got %q, want %q", streamPart.Error.Error(), expectedMsg)
		}
	})
}
