// Package prompttest provides a shared corpus of prompts and a golden-file
// harness for the chat-completions prompt builders.
//
// Four providers (openai, openaicompat, openrouter, vercel) convert a
// fantasy.Prompt into the same OpenAI SDK message type, and each has its own
// copy of the conversion. Running one corpus through all four and recording
// the result turns the differences between them into a diffable artifact:
// testdata/prompt/<case>.json under each provider. Reviewing a change to any
// converter means reading the golden diff rather than the control flow, and a
// behaviour that only one provider has becomes visible instead of buried.
package prompttest

import (
	"encoding/base64"
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/anthropic"
)

var update = flag.Bool("update", false, "rewrite prompt golden files")

// Case is one prompt fixture. Model is passed through to the converter because
// several providers branch on the model ID.
type Case struct {
	Name   string
	Model  string
	Prompt fantasy.Prompt
}

// Result is the recorded output of a converter, in the shape that gets
// serialised to the golden file. Messages is stored as raw JSON so the harness
// records exactly what the SDK would put on the wire, including the extra
// fields providers attach for cache control and reasoning details.
type Result struct {
	Messages json.RawMessage `json:"messages"`
	Warnings []string        `json:"warnings"`
}

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

// Golden compares one converter's output for one case against the recorded
// file, or rewrites it under -update. Warnings are recorded alongside the
// messages because dropping content silently and dropping it with a warning
// are very different behaviours, and only the golden file makes that visible.
func Golden(t *testing.T, dir, name string, messages any, warnings []fantasy.CallWarning) {
	t.Helper()

	raw, err := json.Marshal(messages)
	if err != nil {
		t.Fatalf("marshal messages: %v", err)
	}
	texts := make([]string, 0, len(warnings))
	for _, w := range warnings {
		texts = append(texts, string(w.Type)+": "+w.Message)
	}
	got, err := json.MarshalIndent(Result{Messages: raw, Warnings: texts}, "", "  ")
	if err != nil {
		t.Fatalf("marshal result: %v", err)
	}
	got = append(got, '\n')

	path := filepath.Join(dir, name+".json")
	if *update {
		if err := os.MkdirAll(dir, 0o750); err != nil {
			t.Fatalf("create golden dir: %v", err)
		}
		if err := os.WriteFile(path, got, 0o600); err != nil {
			t.Fatalf("write golden: %v", err)
		}
		return
	}

	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read golden (run with -update to create): %v", err)
	}
	if string(got) != string(want) {
		t.Errorf("golden mismatch for %s\n--- want ---\n%s\n--- got ---\n%s", name, want, got)
	}
}
