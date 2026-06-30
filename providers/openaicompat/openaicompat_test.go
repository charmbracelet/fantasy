package openaicompat

import (
	"errors"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestToPromptFunc_ReasoningContent(t *testing.T) {
	t.Parallel()

	t.Run("should add reasoning_content field to assistant messages", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "What is 2+2?"},
				},
			},
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.ReasoningPart{Text: "Let me think... 2+2 equals 4."},
					fantasy.TextPart{Text: "The answer is 4."},
				},
			},
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "What about 3+3?"},
				},
			},
		}

		messages, warnings := ToPromptFunc(prompt, "", "")

		require.Empty(t, warnings)
		require.Len(t, messages, 3)

		// First message (user) - no reasoning
		msg1 := messages[0].OfUser
		require.NotNil(t, msg1)
		require.Equal(t, "What is 2+2?", msg1.Content.OfString.Value)

		// Second message (assistant) - with reasoning
		msg2 := messages[1].OfAssistant
		require.NotNil(t, msg2)
		require.Equal(t, "The answer is 4.", msg2.Content.OfString.Value)
		// Check reasoning_content in extra fields
		extraFields := msg2.ExtraFields()
		reasoningContent, hasReasoning := extraFields["reasoning_content"]
		require.True(t, hasReasoning)
		require.Equal(t, "Let me think... 2+2 equals 4.", reasoningContent)

		// Third message (user) - no reasoning
		msg3 := messages[2].OfUser
		require.NotNil(t, msg3)
		require.Equal(t, "What about 3+3?", msg3.Content.OfString.Value)
	})

	t.Run("should handle assistant messages with only reasoning content", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Hello"},
				},
			},
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.ReasoningPart{Text: "Internal reasoning only..."},
				},
			},
		}

		messages, warnings := ToPromptFunc(prompt, "", "")

		require.Len(t, warnings, 1)
		require.Contains(t, warnings[0].Message, "dropping empty assistant message")
		require.Len(t, messages, 1) // Only user message, assistant message dropped

		// User message - unchanged
		msg := messages[0].OfUser
		require.NotNil(t, msg)
		require.Equal(t, "Hello", msg.Content.OfString.Value)
	})

	t.Run("should not add reasoning_content to messages without reasoning", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Hello"},
				},
			},
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Hi there!"},
				},
			},
		}

		messages, warnings := ToPromptFunc(prompt, "", "")

		require.Empty(t, warnings)
		require.Len(t, messages, 2)

		// Assistant message without reasoning
		msg := messages[1].OfAssistant
		require.NotNil(t, msg)
		require.Equal(t, "Hi there!", msg.Content.OfString.Value)
		extraFields := msg.ExtraFields()
		_, hasReasoning := extraFields["reasoning_content"]
		require.False(t, hasReasoning)
	})

	t.Run("should preserve system and user messages unchanged", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleSystem,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "You are helpful."},
				},
			},
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Hello"},
				},
			},
		}

		messages, warnings := ToPromptFunc(prompt, "", "")

		require.Empty(t, warnings)
		require.Len(t, messages, 2)

		// System message - unchanged
		systemMsg := messages[0].OfSystem
		require.NotNil(t, systemMsg)
		require.Equal(t, "You are helpful.", systemMsg.Content.OfString.Value)

		// User message - unchanged
		userMsg := messages[1].OfUser
		require.NotNil(t, userMsg)
		require.Equal(t, "Hello", userMsg.Content.OfString.Value)
	})

	t.Run("should use last assistant TextPart only", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Hello"},
				},
			},
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "First part. "},
					fantasy.TextPart{Text: "Second part. "},
					fantasy.TextPart{Text: "Third part."},
				},
			},
		}

		messages, warnings := ToPromptFunc(prompt, "", "")

		require.Empty(t, warnings)
		require.Len(t, messages, 2)

		// Assistant message should use only the last TextPart (matching openai behavior)
		assistantMsg := messages[1].OfAssistant
		require.NotNil(t, assistantMsg)
		require.Equal(t, "Third part.", assistantMsg.Content.OfString.Value)
	})

	t.Run("should include user messages with only unsupported attachments", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Hello"},
				},
			},
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.FilePart{
						MediaType: "application/x-unsupported",
						Data:      []byte("unsupported data"),
					},
				},
			},
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "After unsupported"},
				},
			},
		}

		messages, warnings := ToPromptFunc(prompt, "", "")

		require.Len(t, warnings, 2) // unsupported type + empty message
		require.Contains(t, warnings[0].Message, "not supported")
		require.Contains(t, warnings[1].Message, "dropping empty user message")
		// Should have only 2 messages (empty content message is now dropped)
		require.Len(t, messages, 2)

		msg1 := messages[0].OfUser
		require.NotNil(t, msg1)
		require.Equal(t, "Hello", msg1.Content.OfString.Value)

		msg2 := messages[1].OfUser
		require.NotNil(t, msg2)
		require.Equal(t, "After unsupported", msg2.Content.OfString.Value)
	})

	t.Run("should detect PDF file IDs using strings.HasPrefix", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Check this PDF"},
					fantasy.FilePart{
						MediaType: "application/pdf",
						Data:      []byte("file-abc123xyz"),
						Filename:  "test.pdf",
					},
				},
			},
		}

		messages, warnings := ToPromptFunc(prompt, "", "")

		require.Empty(t, warnings)
		require.Len(t, messages, 1)

		userMsg := messages[0].OfUser
		require.NotNil(t, userMsg)

		content := userMsg.Content.OfArrayOfContentParts
		require.Len(t, content, 2)

		// Second content part should be file with file_id
		filePart := content[1].OfFile
		require.NotNil(t, filePart)
		require.Equal(t, "file-abc123xyz", filePart.File.FileID.Value)
	})
}

func TestToPromptFunc_DropsEmptyMessages(t *testing.T) {
	t.Parallel()

	t.Run("should drop truly empty assistant messages", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Hello"},
				},
			},
			{
				Role:    fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{},
			},
		}

		messages, warnings := ToPromptFunc(prompt, "", "")

		require.Len(t, messages, 1, "should only have user message")
		require.Len(t, warnings, 1)
		require.Equal(t, fantasy.CallWarningTypeOther, warnings[0].Type)
		require.Contains(t, warnings[0].Message, "dropping empty assistant message")
	})

	t.Run("should keep assistant messages with text content", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Hello"},
				},
			},
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Hi there!"},
				},
			},
		}

		messages, warnings := ToPromptFunc(prompt, "", "")

		require.Len(t, messages, 2, "should have both user and assistant messages")
		require.Empty(t, warnings)
	})

	t.Run("should keep assistant messages with tool calls", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "What's the weather?"},
				},
			},
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.ToolCallPart{
						ToolCallID: "call_123",
						ToolName:   "get_weather",
						Input:      `{"location":"NYC"}`,
					},
				},
			},
		}

		messages, warnings := ToPromptFunc(prompt, "", "")

		require.Len(t, messages, 2, "should have both user and assistant messages")
		require.Empty(t, warnings)
	})

	t.Run("should add empty reasoning_content to tool call messages when thinking is enabled", func(t *testing.T) {
		t.Parallel()

		// When thinking is enabled (reasoning parts exist in history),
		// tool call messages without their own reasoning must still include
		// reasoning_content. Providers like Kimi require it.
		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "What is 2+2?"},
				},
			},
			{
				// First turn has reasoning
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.ReasoningPart{Text: "Simple math."},
					fantasy.TextPart{Text: "Four"},
				},
			},
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Now try a tool call"},
				},
			},
			{
				// Tool call WITHOUT reasoning on this turn
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.ToolCallPart{
						ToolCallID: "call_1",
						ToolName:   "execute",
						Input:      `{"command":"echo 4"}`,
					},
				},
			},
		}

		messages, warnings := ToPromptFunc(prompt, "", "")

		require.Empty(t, warnings)
		require.Len(t, messages, 4)

		// Tool call message must have reasoning_content (empty) since
		// thinking is enabled in this conversation
		msg := messages[3].OfAssistant
		require.NotNil(t, msg)
		extraFields := msg.ExtraFields()
		reasoningContent, hasReasoning := extraFields["reasoning_content"]
		require.True(t, hasReasoning, "reasoning_content must be present on tool call messages when thinking is enabled")
		require.Equal(t, "", reasoningContent)
	})

	t.Run("should drop user messages without visible content", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.FilePart{
						Data:      []byte("not supported"),
						MediaType: "application/unknown",
					},
				},
			},
		}

		messages, warnings := ToPromptFunc(prompt, "", "")

		require.Empty(t, messages)
		require.Len(t, warnings, 2) // unsupported type + empty message
		require.Contains(t, warnings[1].Message, "dropping empty user message")
	})

	t.Run("should keep user messages with image content", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.FilePart{
						Data:      []byte{0x01, 0x02, 0x03},
						MediaType: "image/png",
					},
				},
			},
		}

		messages, warnings := ToPromptFunc(prompt, "", "")

		require.Len(t, messages, 1)
		require.Empty(t, warnings)
	})

	t.Run("should keep user messages with tool results", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleTool,
				Content: []fantasy.MessagePart{
					fantasy.ToolResultPart{
						ToolCallID: "call_123",
						Output:     fantasy.ToolResultOutputContentText{Text: "done"},
					},
				},
			},
		}

		messages, warnings := ToPromptFunc(prompt, "", "")

		require.Len(t, messages, 1)
		require.Empty(t, warnings)
	})

	t.Run("should keep user messages with tool error results", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleTool,
				Content: []fantasy.MessagePart{
					fantasy.ToolResultPart{
						ToolCallID: "call_456",
						Output:     fantasy.ToolResultOutputContentError{Error: errors.New("boom")},
					},
				},
			},
		}

		messages, warnings := ToPromptFunc(prompt, "", "")

		require.Len(t, messages, 1)
		require.Empty(t, warnings)
	})
}

func TestToPromptFunc_ContentExtraFields(t *testing.T) {
	t.Parallel()

	t.Run("should add cache_control to user message with content array", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{
						Text: "Analyze this document.",
						ProviderOptions: fantasy.ProviderOptions{
							Name: &ContentExtraFields{
								Fields: map[string]any{"cache_control": map[string]string{"type": "ephemeral"}},
							},
						},
					},
				},
			},
		}

		messages, warnings := ToPromptFunc(prompt, "", "")

		require.Empty(t, warnings)
		require.Len(t, messages, 1)

		msg := messages[0].OfUser
		require.NotNil(t, msg)
		// Should use content array format when cache_control is present
		require.Equal(t, "", msg.Content.OfString.Value)
		require.NotNil(t, msg.Content.OfArrayOfContentParts)
		require.Len(t, msg.Content.OfArrayOfContentParts, 1)

		textBlock := msg.Content.OfArrayOfContentParts[0].OfText
		require.NotNil(t, textBlock)
		require.Equal(t, "Analyze this document.", textBlock.Text)

		// Check cache_control in extra fields
		extraFields := textBlock.ExtraFields()
		cacheControl, hasCacheControl := extraFields["cache_control"]
		require.True(t, hasCacheControl)
		cacheMap, ok := cacheControl.(map[string]string)
		require.True(t, ok)
		require.Equal(t, "ephemeral", cacheMap["type"])
	})

	t.Run("should add cache_control to system message with content array", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleSystem,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{
						Text: "You are a helpful assistant.",
						ProviderOptions: fantasy.ProviderOptions{
							Name: &ContentExtraFields{
								Fields: map[string]any{"cache_control": map[string]string{"type": "ephemeral"}},
							},
						},
					},
				},
			},
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Hello"},
				},
			},
		}

		messages, warnings := ToPromptFunc(prompt, "", "")

		require.Empty(t, warnings)
		require.Len(t, messages, 2)

		systemMsg := messages[0].OfSystem
		require.NotNil(t, systemMsg)
		// Should use content array format when cache_control is present
		require.Equal(t, "", systemMsg.Content.OfString.Value)
		require.NotNil(t, systemMsg.Content.OfArrayOfContentParts)
		require.Len(t, systemMsg.Content.OfArrayOfContentParts, 1)

		textBlock := systemMsg.Content.OfArrayOfContentParts[0]
		require.Equal(t, "You are a helpful assistant.", textBlock.Text)

		// Check cache_control in extra fields
		extraFields := textBlock.ExtraFields()
		cacheControl, hasCacheControl := extraFields["cache_control"]
		require.True(t, hasCacheControl)
		cacheMap, ok := cacheControl.(map[string]string)
		require.True(t, ok)
		require.Equal(t, "ephemeral", cacheMap["type"])
	})

	t.Run("should add cache_control to assistant message with content array", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Hello"},
				},
			},
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{
						Text: "Hi there!",
						ProviderOptions: fantasy.ProviderOptions{
							Name: &ContentExtraFields{
								Fields: map[string]any{"cache_control": map[string]string{"type": "ephemeral"}},
							},
						},
					},
				},
			},
		}

		messages, warnings := ToPromptFunc(prompt, "", "")

		require.Empty(t, warnings)
		require.Len(t, messages, 2)

		assistantMsg := messages[1].OfAssistant
		require.NotNil(t, assistantMsg)
		// Should use content array format when cache_control is present
		require.Equal(t, "", assistantMsg.Content.OfString.Value)
		require.NotNil(t, assistantMsg.Content.OfArrayOfContentParts)
		require.Len(t, assistantMsg.Content.OfArrayOfContentParts, 1)

		textBlock := assistantMsg.Content.OfArrayOfContentParts[0].OfText
		require.NotNil(t, textBlock)
		require.Equal(t, "Hi there!", textBlock.Text)

		// Check cache_control in extra fields
		extraFields := textBlock.ExtraFields()
		cacheControl, hasCacheControl := extraFields["cache_control"]
		require.True(t, hasCacheControl)
		cacheMap, ok := cacheControl.(map[string]string)
		require.True(t, ok)
		require.Equal(t, "ephemeral", cacheMap["type"])
	})

	t.Run("should not use content array for messages without cache_control", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Hello"},
				},
			},
		}

		messages, warnings := ToPromptFunc(prompt, "", "")

		require.Empty(t, warnings)
		require.Len(t, messages, 1)

		msg := messages[0].OfUser
		require.NotNil(t, msg)
		// Should use string format when no cache_control
		require.NotEqual(t, "", msg.Content.OfString.Value)
		require.Equal(t, "Hello", msg.Content.OfString.Value)
		require.Nil(t, msg.Content.OfArrayOfContentParts)
	})

	t.Run("should handle multiple content parts with mixed cache_control", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Context: "},
					fantasy.TextPart{
						Text: "Large document here...",
						ProviderOptions: fantasy.ProviderOptions{
							Name: &ContentExtraFields{
								Fields: map[string]any{"cache_control": map[string]string{"type": "ephemeral"}},
							},
						},
					},
				},
			},
		}

		messages, warnings := ToPromptFunc(prompt, "", "")

		require.Empty(t, warnings)
		require.Len(t, messages, 1)

		msg := messages[0].OfUser
		require.NotNil(t, msg)
		// Should use content array format since one part has cache_control
		require.Equal(t, "", msg.Content.OfString.Value)
		require.NotNil(t, msg.Content.OfArrayOfContentParts)
		require.Len(t, msg.Content.OfArrayOfContentParts, 2)

		// First part should not have cache_control
		firstBlock := msg.Content.OfArrayOfContentParts[0].OfText
		require.NotNil(t, firstBlock)
		require.Equal(t, "Context: ", firstBlock.Text)
		require.Empty(t, firstBlock.ExtraFields())

		// Second part should have cache_control
		secondBlock := msg.Content.OfArrayOfContentParts[1].OfText
		require.NotNil(t, secondBlock)
		require.Equal(t, "Large document here...", secondBlock.Text)
		extraFields := secondBlock.ExtraFields()
		cacheControl, hasCacheControl := extraFields["cache_control"]
		require.True(t, hasCacheControl)
		cacheMap, ok := cacheControl.(map[string]string)
		require.True(t, ok)
		require.Equal(t, "ephemeral", cacheMap["type"])
	})

	t.Run("should fall back to message-level cache_control when part has none", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{
						Text: "Hello",
						// No cache_control on this part
					},
				},
				// But cache_control on the message itself
				ProviderOptions: fantasy.ProviderOptions{
					Name: &ContentExtraFields{
						Fields: map[string]any{"cache_control": map[string]string{"type": "ephemeral"}},
					},
				},
			},
		}

		messages, warnings := ToPromptFunc(prompt, "", "")

		require.Empty(t, warnings)
		require.Len(t, messages, 1)

		msg := messages[0].OfUser
		require.NotNil(t, msg)
		// Should use content array format due to message-level cache_control
		require.Equal(t, "", msg.Content.OfString.Value)
		require.NotNil(t, msg.Content.OfArrayOfContentParts)
		require.Len(t, msg.Content.OfArrayOfContentParts, 1)

		textBlock := msg.Content.OfArrayOfContentParts[0].OfText
		require.NotNil(t, textBlock)
		require.Equal(t, "Hello", textBlock.Text)

		// Check cache_control in extra fields
		extraFields := textBlock.ExtraFields()
		cacheControl, hasCacheControl := extraFields["cache_control"]
		require.True(t, hasCacheControl)
		cacheMap, ok := cacheControl.(map[string]string)
		require.True(t, ok)
		require.Equal(t, "ephemeral", cacheMap["type"])
	})

	t.Run("should prefer part-level cache_control over message-level", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{
						Text: "First part",
						// Part-level cache_control with different type
						ProviderOptions: fantasy.ProviderOptions{
							Name: &ContentExtraFields{
								Fields: map[string]any{"cache_control": map[string]string{"type": "ephemeral"}},
							},
						},
					},
					fantasy.TextPart{Text: "Second part"}, // No part-level
				},
				// Message-level cache_control
				ProviderOptions: fantasy.ProviderOptions{
					Name: &ContentExtraFields{
						Fields: map[string]any{"cache_control": map[string]string{"type": "persistent"}},
					},
				},
			},
		}

		messages, warnings := ToPromptFunc(prompt, "", "")

		require.Empty(t, warnings)
		require.Len(t, messages, 1)

		msg := messages[0].OfUser
		require.NotNil(t, msg)
		require.NotNil(t, msg.Content.OfArrayOfContentParts)
		require.Len(t, msg.Content.OfArrayOfContentParts, 2)

		// First part should have ephemeral (part-level wins)
		firstBlock := msg.Content.OfArrayOfContentParts[0].OfText
		require.NotNil(t, firstBlock)
		extraFields := firstBlock.ExtraFields()
		cacheControl := extraFields["cache_control"].(map[string]string)
		require.Equal(t, "ephemeral", cacheControl["type"])

		// Second part should have persistent (message-level fallback)
		secondBlock := msg.Content.OfArrayOfContentParts[1].OfText
		require.NotNil(t, secondBlock)
		extraFields = secondBlock.ExtraFields()
		cacheControl = extraFields["cache_control"].(map[string]string)
		require.Equal(t, "persistent", cacheControl["type"])
	})

	t.Run("should add cache_control to multi-part assistant message with tool calls", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{
						Text: "Let me check that.",
						ProviderOptions: fantasy.ProviderOptions{
							Name: &ContentExtraFields{
								Fields: map[string]any{"cache_control": map[string]string{"type": "ephemeral"}},
							},
						},
					},
					fantasy.ToolCallPart{
						ToolCallID: "call_1",
						ToolName:   "lookup",
						Input:      "{}",
					},
				},
			},
		}

		messages, warnings := ToPromptFunc(prompt, "", "")

		require.Empty(t, warnings)
		require.Len(t, messages, 1)

		assistantMsg := messages[0].OfAssistant
		require.NotNil(t, assistantMsg)
		// Tool calls force the multi-part path; cache_control must still survive.
		require.Equal(t, "", assistantMsg.Content.OfString.Value)
		require.Len(t, assistantMsg.Content.OfArrayOfContentParts, 1)
		require.Len(t, assistantMsg.ToolCalls, 1)

		textBlock := assistantMsg.Content.OfArrayOfContentParts[0].OfText
		require.NotNil(t, textBlock)
		require.Equal(t, "Let me check that.", textBlock.Text)
		cacheControl := textBlock.ExtraFields()["cache_control"].(map[string]string)
		require.Equal(t, "ephemeral", cacheControl["type"])
	})
}
