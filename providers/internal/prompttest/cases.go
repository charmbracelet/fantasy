package prompttest

import (
	"encoding/base64"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/anthropic"
)

// The corpus: one prompt per rule worth pinning, exercised against every
// chat-completions converter by Golden.

func cacheControl() fantasy.ProviderOptions {
	return fantasy.ProviderOptions{
		anthropic.Name: &anthropic.ProviderCacheControlOptions{
			CacheControl: anthropic.CacheControl{Type: "ephemeral"},
		},
	}
}

func png() []byte { return []byte{0x89, 0x50, 0x4e, 0x47} }

func b64(data []byte) string { return base64.StdEncoding.EncodeToString(data) }

// Cases returns the shared corpus. Every converter should be able to handle
// every case; the point of the corpus is that the cases a converter handles
// badly show up in its golden files rather than going unnoticed.
func Cases() []Case {
	return []Case{
		{
			Name:  "system_single_text",
			Model: "gpt-4o",
			Prompt: fantasy.Prompt{
				{Role: fantasy.MessageRoleSystem, Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "you are helpful"},
				}},
				{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "hello"},
				}},
			},
		},
		{
			Name:  "system_multiple_text_parts",
			Model: "gpt-4o",
			Prompt: fantasy.Prompt{
				{Role: fantasy.MessageRoleSystem, Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "first"},
					fantasy.TextPart{Text: "second"},
				}},
				{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "hello"},
				}},
			},
		},
		{
			Name:  "system_cache_control",
			Model: "anthropic/claude-sonnet-4",
			Prompt: fantasy.Prompt{
				{
					Role:            fantasy.MessageRoleSystem,
					ProviderOptions: cacheControl(),
					Content: []fantasy.MessagePart{
						fantasy.TextPart{Text: "cached system prompt"},
					},
				},
				{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "hello"},
				}},
			},
		},
		{
			Name:  "user_text_and_image",
			Model: "gpt-4o",
			Prompt: fantasy.Prompt{
				{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "what is this"},
					fantasy.FilePart{Filename: "a.png", MediaType: "image/png", Data: png()},
				}},
			},
		},
		{
			Name:  "user_pdf",
			Model: "gpt-4o",
			Prompt: fantasy.Prompt{
				{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{
					fantasy.FilePart{Filename: "doc.pdf", MediaType: "application/pdf", Data: []byte("%PDF-1.4")},
				}},
			},
		},
		{
			Name:  "user_audio_wav",
			Model: "gpt-4o-audio-preview",
			Prompt: fantasy.Prompt{
				{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{
					fantasy.FilePart{Filename: "a.wav", MediaType: "audio/wav", Data: []byte{1, 2, 3}},
				}},
			},
		},
		{
			Name:  "user_unsupported_media",
			Model: "gpt-4o",
			Prompt: fantasy.Prompt{
				{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{
					fantasy.FilePart{Filename: "a.mp4", MediaType: "video/mp4", Data: []byte{1, 2, 3}},
				}},
			},
		},
		{
			Name:  "user_cache_control",
			Model: "anthropic/claude-sonnet-4",
			Prompt: fantasy.Prompt{
				{
					Role:            fantasy.MessageRoleUser,
					ProviderOptions: cacheControl(),
					Content: []fantasy.MessagePart{
						fantasy.TextPart{Text: "cached user turn"},
					},
				},
			},
		},
		{
			// Every converter should drop this and warn: an empty user message
			// is not valid input and silently forwarding it produces an API
			// error far from the cause.
			Name:  "user_empty_is_dropped",
			Model: "gpt-4o",
			Prompt: fantasy.Prompt{
				{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: ""},
				}},
				{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "real turn"},
				}},
			},
		},
		{
			Name:  "assistant_text_only",
			Model: "gpt-4o",
			Prompt: fantasy.Prompt{
				{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "hi"},
				}},
				{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "hello there"},
				}},
			},
		},
		{
			Name:  "assistant_text_and_tool_calls",
			Model: "gpt-4o",
			Prompt: fantasy.Prompt{
				{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "look at these"},
				}},
				{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "on it"},
					fantasy.ToolCallPart{ToolCallID: "c1", ToolName: "view", Input: `{"path":"a"}`},
					fantasy.ToolCallPart{ToolCallID: "c2", ToolName: "view", Input: `{"path":"b"}`},
				}},
				{Role: fantasy.MessageRoleTool, Content: []fantasy.MessagePart{
					fantasy.ToolResultPart{ToolCallID: "c1", Output: fantasy.ToolResultOutputContentText{Text: "a body"}},
				}},
				{Role: fantasy.MessageRoleTool, Content: []fantasy.MessagePart{
					fantasy.ToolResultPart{ToolCallID: "c2", Output: fantasy.ToolResultOutputContentText{Text: "b body"}},
				}},
			},
		},
		{
			// Every converter should drop this and warn. An assistant message
			// with neither content nor tool_calls is rejected outright, and
			// because it stays in history the rejection repeats on every
			// later request (charmbracelet/crush#3794).
			Name:  "assistant_empty_is_dropped",
			Model: "gpt-4o",
			Prompt: fantasy.Prompt{
				{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "hi"},
				}},
				{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{}},
				{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "still here"},
				}},
			},
		},
		{
			// The assistant turn is the one role whose cache hint travels on
			// the message rather than on a content part, so it needs its own
			// case to keep that path honest.
			Name:  "assistant_cache_control",
			Model: "anthropic/claude-sonnet-4",
			Prompt: fantasy.Prompt{
				{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "hi"},
				}},
				{
					Role:            fantasy.MessageRoleAssistant,
					ProviderOptions: cacheControl(),
					Content: []fantasy.MessagePart{
						fantasy.TextPart{Text: "cached assistant turn"},
					},
				},
			},
		},
		{
			Name:  "assistant_reasoning_anthropic",
			Model: "anthropic/claude-sonnet-4",
			Prompt: fantasy.Prompt{
				{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "think"},
				}},
				{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{
					fantasy.ReasoningPart{Text: "thinking about it"},
					fantasy.TextPart{Text: "the answer"},
				}},
			},
		},
		{
			Name:  "assistant_reasoning_only",
			Model: "anthropic/claude-sonnet-4",
			Prompt: fantasy.Prompt{
				{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "think"},
				}},
				{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{
					fantasy.ReasoningPart{Text: "only reasoning, no text"},
				}},
			},
		},
		{
			Name:  "tool_result_error",
			Model: "gpt-4o",
			Prompt: fantasy.Prompt{
				{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{
					fantasy.ToolCallPart{ToolCallID: "c1", ToolName: "view", Input: "{}"},
				}},
				{Role: fantasy.MessageRoleTool, Content: []fantasy.MessagePart{
					fantasy.ToolResultPart{ToolCallID: "c1", Output: fantasy.ToolResultOutputContentError{Error: errFixture{}}},
				}},
			},
		},
		{
			// The tool_call declared above must be answered. A converter that
			// drops the media result leaves an unanswered tool_call_id, which
			// strict backends reject.
			Name:  "tool_result_media_image",
			Model: "gpt-4o",
			Prompt: fantasy.Prompt{
				{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{
					fantasy.ToolCallPart{ToolCallID: "c1", ToolName: "view", Input: "{}"},
				}},
				{Role: fantasy.MessageRoleTool, Content: []fantasy.MessagePart{
					fantasy.ToolResultPart{ToolCallID: "c1", Output: fantasy.ToolResultOutputContentMedia{
						Data:      b64(png()),
						MediaType: "image/png",
					}},
				}},
			},
		},
		{
			// Accompanying text wins over the generated placeholder, so the
			// tool message reads as the tool's own answer.
			Name:  "tool_result_media_image_with_text",
			Model: "gpt-4o",
			Prompt: fantasy.Prompt{
				{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{
					fantasy.ToolCallPart{ToolCallID: "c1", ToolName: "view", Input: "{}"},
				}},
				{Role: fantasy.MessageRoleTool, Content: []fantasy.MessagePart{
					fantasy.ToolResultPart{ToolCallID: "c1", Output: fantasy.ToolResultOutputContentMedia{
						Data:      b64(png()),
						MediaType: "image/png",
						Text:      "Screenshot of the blockquote element.",
					}},
				}},
			},
		},
		{
			// Audio results travel as an input_audio part, format "wav".
			Name:  "tool_result_media_audio_wav",
			Model: "gpt-4o-audio-preview",
			Prompt: fantasy.Prompt{
				{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{
					fantasy.ToolCallPart{ToolCallID: "c1", ToolName: "say", Input: "{}"},
				}},
				{Role: fantasy.MessageRoleTool, Content: []fantasy.MessagePart{
					fantasy.ToolResultPart{ToolCallID: "c1", Output: fantasy.ToolResultOutputContentMedia{
						Data:      b64([]byte{1, 2, 3}),
						MediaType: "audio/wav",
					}},
				}},
			},
		},
		{
			// audio/mpeg and audio/mp3 both map to format "mp3", the branch
			// the wav case above does not reach.
			Name:  "tool_result_media_audio_mpeg",
			Model: "gpt-4o-audio-preview",
			Prompt: fantasy.Prompt{
				{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{
					fantasy.ToolCallPart{ToolCallID: "c1", ToolName: "say", Input: "{}"},
				}},
				{Role: fantasy.MessageRoleTool, Content: []fantasy.MessagePart{
					fantasy.ToolResultPart{ToolCallID: "c1", Output: fantasy.ToolResultOutputContentMedia{
						Data:      b64([]byte{1, 2, 3}),
						MediaType: "audio/mpeg",
					}},
				}},
			},
		},
		{
			// An unsupported media type still has to answer the tool_call, so
			// the text message goes out alone with a warning naming the type.
			Name:  "tool_result_media_unsupported",
			Model: "gpt-4o",
			Prompt: fantasy.Prompt{
				{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{
					fantasy.ToolCallPart{ToolCallID: "c1", ToolName: "record", Input: "{}"},
				}},
				{Role: fantasy.MessageRoleTool, Content: []fantasy.MessagePart{
					fantasy.ToolResultPart{ToolCallID: "c1", Output: fantasy.ToolResultOutputContentMedia{
						Data:      b64([]byte{1, 2, 3}),
						MediaType: "video/mp4",
					}},
				}},
			},
		},
		{
			// The ordering case from the media fix: a parallel batch must keep
			// its tool messages contiguous.
			Name:  "tool_result_media_parallel_batch",
			Model: "gpt-4o",
			Prompt: fantasy.Prompt{
				{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{
					fantasy.ToolCallPart{ToolCallID: "c1", ToolName: "view", Input: "{}"},
					fantasy.ToolCallPart{ToolCallID: "c2", ToolName: "view", Input: "{}"},
				}},
				{Role: fantasy.MessageRoleTool, Content: []fantasy.MessagePart{
					fantasy.ToolResultPart{ToolCallID: "c1", Output: fantasy.ToolResultOutputContentMedia{
						Data:      b64(png()),
						MediaType: "image/png",
					}},
				}},
				{Role: fantasy.MessageRoleTool, Content: []fantasy.MessagePart{
					fantasy.ToolResultPart{ToolCallID: "c2", Output: fantasy.ToolResultOutputContentText{Text: "plain"}},
				}},
				{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "next"},
				}},
			},
		},
		{
			Name:  "tool_result_cache_control",
			Model: "anthropic/claude-sonnet-4",
			Prompt: fantasy.Prompt{
				{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{
					fantasy.ToolCallPart{ToolCallID: "c1", ToolName: "view", Input: "{}"},
				}},
				{
					Role:            fantasy.MessageRoleTool,
					ProviderOptions: cacheControl(),
					Content: []fantasy.MessagePart{
						fantasy.ToolResultPart{ToolCallID: "c1", Output: fantasy.ToolResultOutputContentText{Text: "cached result"}},
					},
				},
			},
		},
	}
}

type errFixture struct{}

func (errFixture) Error() string { return "tool blew up" }
