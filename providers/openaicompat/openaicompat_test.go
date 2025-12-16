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
