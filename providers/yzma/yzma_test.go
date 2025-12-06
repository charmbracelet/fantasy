package yzma

import (
	"testing"

	"charm.land/fantasy"
	"github.com/hybridgroup/yzma/pkg/message"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestConvertMessageContent(t *testing.T) {
	t.Parallel()

	t.Run("empty prompt", func(t *testing.T) {
		prompt := fantasy.Prompt{}
		result := convertMessageContent(prompt)
		assert.Empty(t, result)
	})

	t.Run("single user text message", func(t *testing.T) {
		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Hello, world!"},
				},
			},
		}
		result := convertMessageContent(prompt)
		require.Len(t, result, 1)

		chat, ok := result[0].(message.Chat)
		require.True(t, ok)
		assert.Equal(t, "user", chat.Role)
		assert.Equal(t, "Hello, world!", chat.Content)
	})

	t.Run("system and user messages", func(t *testing.T) {
		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleSystem,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "You are a helpful assistant."},
				},
			},
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "What is 2+2?"},
				},
			},
		}
		result := convertMessageContent(prompt)
		require.Len(t, result, 2)

		systemChat, ok := result[0].(message.Chat)
		require.True(t, ok)
		assert.Equal(t, "system", systemChat.Role)
		assert.Equal(t, "You are a helpful assistant.", systemChat.Content)

		userChat, ok := result[1].(message.Chat)
		require.True(t, ok)
		assert.Equal(t, "user", userChat.Role)
		assert.Equal(t, "What is 2+2?", userChat.Content)
	})

	t.Run("assistant message", func(t *testing.T) {
		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "I can help with that."},
				},
			},
		}
		result := convertMessageContent(prompt)
		require.Len(t, result, 1)

		chat, ok := result[0].(message.Chat)
		require.True(t, ok)
		assert.Equal(t, "assistant", chat.Role)
		assert.Equal(t, "I can help with that.", chat.Content)
	})

	t.Run("conversation with multiple roles", func(t *testing.T) {
		prompt := fantasy.Prompt{
			{
				Role:    fantasy.MessageRoleSystem,
				Content: []fantasy.MessagePart{fantasy.TextPart{Text: "Be concise."}},
			},
			{
				Role:    fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{fantasy.TextPart{Text: "Hi"}},
			},
			{
				Role:    fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{fantasy.TextPart{Text: "Hello!"}},
			},
			{
				Role:    fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{fantasy.TextPart{Text: "How are you?"}},
			},
		}
		result := convertMessageContent(prompt)
		require.Len(t, result, 4)

		assert.Equal(t, "system", result[0].(message.Chat).Role)
		assert.Equal(t, "user", result[1].(message.Chat).Role)
		assert.Equal(t, "assistant", result[2].(message.Chat).Role)
		assert.Equal(t, "user", result[3].(message.Chat).Role)
	})

	t.Run("tool call in assistant message", func(t *testing.T) {
		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.ToolCallPart{
						ToolCallID: "call_123",
						ToolName:   "get_weather",
						Input:      "{\"location\": \"New York\"}",
					},
				},
			},
		}
		result := convertMessageContent(prompt)
		// Should produce a message with tool call info
		require.Len(t, result, 1, "tool call should produce a message")
	})

	t.Run("tool result message", func(t *testing.T) {
		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleTool,
				Content: []fantasy.MessagePart{
					fantasy.ToolResultPart{
						ToolCallID: "call_123",
						Output:     fantasy.ToolResultOutputContentText{Text: "Sunny, 25°C"},
					},
				},
			},
		}
		result := convertMessageContent(prompt)
		// Should produce a message with tool result
		require.Len(t, result, 1, "tool result should produce a message")
	})

	t.Run("assistant with text and tool call", func(t *testing.T) {
		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Let me check the weather."},
					fantasy.ToolCallPart{
						ToolCallID: "call_456",
						ToolName:   "get_weather",
						Input:      "{\"location\": \"San Francisco\"}",
					},
				},
			},
		}
		result := convertMessageContent(prompt)
		// Should handle both text and tool call
		require.GreaterOrEqual(t, len(result), 1, "should produce at least one message")
	})

	t.Run("full tool use conversation", func(t *testing.T) {
		prompt := fantasy.Prompt{
			{
				Role:    fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{fantasy.TextPart{Text: "What's the weather in Tokyo?"}},
			},
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.ToolCallPart{
						ToolCallID: "call_456",
						ToolName:   "get_weather",
						Input:      "{\"location\": \"Tokyo\"}",
					},
				},
			},
			{
				Role: fantasy.MessageRoleTool,
				Content: []fantasy.MessagePart{
					fantasy.ToolResultPart{
						ToolCallID: "call_123",
						Output:     fantasy.ToolResultOutputContentText{Text: "Rainy, 18°C"},
					},
				},
			},
			{
				Role:    fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{fantasy.TextPart{Text: "The weather in Tokyo is rainy with 18°C."}},
			},
		}
		result := convertMessageContent(prompt)
		require.Len(t, result, 4, "should handle full tool conversation")
	})
}
