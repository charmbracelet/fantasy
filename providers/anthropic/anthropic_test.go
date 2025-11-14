package anthropic

import (
	"errors"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestToPrompt_DropsEmptyMessages(t *testing.T) {
	t.Parallel()

	t.Run("should drop assistant messages with only reasoning content", func(t *testing.T) {
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
					fantasy.ReasoningPart{
						Text: "Let me think about this...",
						ProviderOptions: fantasy.ProviderOptions{
							Name: &ReasoningOptionMetadata{
								Signature: "abc123",
							},
						},
					},
				},
			},
		}

		systemBlocks, messages, warnings := toPrompt(prompt, true)

		require.Empty(t, systemBlocks)
		require.Len(t, messages, 1, "should only have user message, assistant message should be dropped")
		require.Len(t, warnings, 1)
		require.Equal(t, fantasy.CallWarningTypeOther, warnings[0].Type)
		require.Contains(t, warnings[0].Message, "dropping empty assistant message")
		require.Contains(t, warnings[0].Message, "neither user-facing content nor tool calls")
	})

	t.Run("should drop assistant reasoning when sendReasoning disabled", func(t *testing.T) {
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
					fantasy.ReasoningPart{
						Text: "Let me think about this...",
						ProviderOptions: fantasy.ProviderOptions{
							Name: &ReasoningOptionMetadata{
								Signature: "def456",
							},
						},
					},
				},
			},
		}

		systemBlocks, messages, warnings := toPrompt(prompt, false)

		require.Empty(t, systemBlocks)
		require.Len(t, messages, 1, "should only have user message, assistant message should be dropped")
		require.Len(t, warnings, 2)
		require.Equal(t, fantasy.CallWarningTypeOther, warnings[0].Type)
		require.Contains(t, warnings[0].Message, "sending reasoning content is disabled")
		require.Equal(t, fantasy.CallWarningTypeOther, warnings[1].Type)
		require.Contains(t, warnings[1].Message, "dropping empty assistant message")
	})

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

		systemBlocks, messages, warnings := toPrompt(prompt, true)

		require.Empty(t, systemBlocks)
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

		systemBlocks, messages, warnings := toPrompt(prompt, true)

		require.Empty(t, systemBlocks)
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

		systemBlocks, messages, warnings := toPrompt(prompt, true)

		require.Empty(t, systemBlocks)
		require.Len(t, messages, 2, "should have both user and assistant messages")
		require.Empty(t, warnings)
	})

	t.Run("should drop assistant messages with invalid tool input", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Hi"},
				},
			},
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.ToolCallPart{
						ToolCallID: "call_123",
						ToolName:   "get_weather",
						Input:      "{not-json",
					},
				},
			},
		}

		systemBlocks, messages, warnings := toPrompt(prompt, true)

		require.Empty(t, systemBlocks)
		require.Len(t, messages, 1, "should only have user message")
		require.Len(t, warnings, 1)
		require.Equal(t, fantasy.CallWarningTypeOther, warnings[0].Type)
		require.Contains(t, warnings[0].Message, "dropping empty assistant message")
	})

	t.Run("should keep assistant messages with reasoning and text", func(t *testing.T) {
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
					fantasy.ReasoningPart{
						Text: "Let me think...",
						ProviderOptions: fantasy.ProviderOptions{
							Name: &ReasoningOptionMetadata{
								Signature: "abc123",
							},
						},
					},
					fantasy.TextPart{Text: "Hi there!"},
				},
			},
		}

		systemBlocks, messages, warnings := toPrompt(prompt, true)

		require.Empty(t, systemBlocks)
		require.Len(t, messages, 2, "should have both user and assistant messages")
		require.Empty(t, warnings)
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

		systemBlocks, messages, warnings := toPrompt(prompt, true)

		require.Empty(t, systemBlocks)
		require.Len(t, messages, 1)
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
						MediaType: "application/pdf",
					},
				},
			},
		}

		systemBlocks, messages, warnings := toPrompt(prompt, true)

		require.Empty(t, systemBlocks)
		require.Empty(t, messages)
		require.Len(t, warnings, 1)
		require.Equal(t, fantasy.CallWarningTypeOther, warnings[0].Type)
		require.Contains(t, warnings[0].Message, "dropping empty user message")
		require.Contains(t, warnings[0].Message, "neither user-facing content nor tool results")
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

		systemBlocks, messages, warnings := toPrompt(prompt, true)

		require.Empty(t, systemBlocks)
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

		systemBlocks, messages, warnings := toPrompt(prompt, true)

		require.Empty(t, systemBlocks)
		require.Len(t, messages, 1)
		require.Empty(t, warnings)
	})

	t.Run("should keep user messages with tool media results", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleTool,
				Content: []fantasy.MessagePart{
					fantasy.ToolResultPart{
						ToolCallID: "call_789",
						Output: fantasy.ToolResultOutputContentMedia{
							Data:      "AQID",
							MediaType: "image/png",
						},
					},
				},
			},
		}

		systemBlocks, messages, warnings := toPrompt(prompt, true)

		require.Empty(t, systemBlocks)
		require.Len(t, messages, 1)
		require.Empty(t, warnings)
	})
}
