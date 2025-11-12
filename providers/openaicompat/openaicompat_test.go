package openaicompat

import (
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

		require.Empty(t, warnings)
		require.Len(t, messages, 2)

		// Assistant message with only reasoning
		msg := messages[1].OfAssistant
		require.NotNil(t, msg)
		extraFields := msg.ExtraFields()
		reasoningContent, hasReasoning := extraFields["reasoning_content"]
		require.True(t, hasReasoning)
		require.Equal(t, "Internal reasoning only...", reasoningContent)
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

		require.Len(t, warnings, 1)
		require.Contains(t, warnings[0].Message, "not supported")
		// Should have all 3 messages (matching openai behavior - don't skip empty content)
		require.Len(t, messages, 3)

		msg1 := messages[0].OfUser
		require.NotNil(t, msg1)
		require.Equal(t, "Hello", msg1.Content.OfString.Value)

		// Second message has empty content (unsupported attachment was skipped)
		msg2 := messages[1].OfUser
		require.NotNil(t, msg2)
		content2 := msg2.Content.OfArrayOfContentParts
		require.Len(t, content2, 0)

		msg3 := messages[2].OfUser
		require.NotNil(t, msg3)
		require.Equal(t, "After unsupported", msg3.Content.OfString.Value)
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
