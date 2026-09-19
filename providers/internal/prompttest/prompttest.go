// Package prompttest provides a shared corpus of prompts and a golden-file
// harness for the chat-completions prompt builders.
//
// Four providers (openai, openaicompat, openrouter, vercel) convert a
// fantasy.Prompt into the same OpenAI SDK message type, and each has its own
// copy of the conversion. One corpus runs through all four and records every
// converter's output for a case in a single file, testdata/<case>.json, keyed
// by provider. The file is the behaviour matrix: a difference between
// providers is a difference between adjacent keys, visible without running a
// diff across four directories.
//
// Reviewing a change to any converter means reading the golden diff rather
// than the control flow, and a behaviour that only one provider has becomes
// visible instead of buried. Regenerate with:
//
//	go test ./providers/internal/prompttest -update
package prompttest

import (
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/anthropic"
	"charm.land/fantasy/providers/openai"
)

var update = flag.Bool("update", false, "rewrite prompt golden files")

// Case is one prompt fixture. Model is passed through to the converter because
// several providers branch on the model ID.
type Case struct {
	Name   string
	Model  string
	Prompt fantasy.Prompt
}

// Result is one converter's output for one case. Messages is stored as raw
// JSON so the harness records exactly what the SDK would put on the wire,
// including the extra fields providers attach for cache control and reasoning
// details.
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

// Converter is one named prompt builder under test.
type Converter struct {
	Name    string
	Convert openai.LanguageModelToPromptFunc
}

// Golden runs every case through every converter and compares the recorded
// matrix against testdata, or rewrites it under -update.
//
// Warnings are recorded alongside the messages because dropping content
// silently and dropping it with a warning are very different behaviours, and
// only the record makes that difference visible.
func Golden(t *testing.T, converters []Converter) {
	t.Helper()

	seen := make(map[string]bool, len(Cases()))
	for _, tc := range Cases() {
		if seen[tc.Name] {
			t.Fatalf("duplicate case name %q: both would write the same golden file", tc.Name)
		}
		seen[tc.Name] = true
	}

	for _, tc := range Cases() {
		t.Run(tc.Name, func(t *testing.T) {
			t.Parallel()
			got := record(t, tc, converters)
			path := filepath.Join(goldenDir, tc.Name+".json")
			if *update {
				if err := os.MkdirAll(goldenDir, 0o750); err != nil {
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
			if diff := lineDiff(string(want), string(got)); diff != "" {
				t.Errorf("golden mismatch for %s:\n%s", tc.Name, diff)
			}
		})
	}

	t.Run("no_stale_goldens", func(t *testing.T) {
		if *update {
			t.Skip("rewriting goldens")
		}
		entries, err := os.ReadDir(goldenDir)
		if err != nil {
			t.Fatalf("read golden dir: %v", err)
		}
		for _, e := range entries {
			name := strings.TrimSuffix(e.Name(), ".json")
			if !seen[name] {
				t.Errorf("golden %s has no matching case; delete it or restore the case", e.Name())
			}
		}
	})
}

const goldenDir = "testdata"

// record converts one case with every converter and marshals the matrix.
//
// Converters that produce identical output share one entry, keyed by their
// comma-joined names. Most cases end up as a single entry naming all four,
// which is the point: agreement collapses to one block, and any file with
// more than one key is showing you exactly where the providers diverge.
func record(t *testing.T, tc Case, converters []Converter) []byte {
	t.Helper()
	matrix := map[string]Result{}
	names := map[string][]string{}
	for _, c := range converters {
		messages, warnings := c.Convert(tc.Prompt, c.Name, tc.Model)
		raw, err := json.Marshal(messages)
		if err != nil {
			t.Fatalf("marshal %s messages: %v", c.Name, err)
		}
		texts := make([]string, 0, len(warnings))
		for _, w := range warnings {
			texts = append(texts, string(w.Type)+": "+w.Message)
		}
		result := Result{Messages: raw, Warnings: texts}
		key, err := json.Marshal(result)
		if err != nil {
			t.Fatalf("key %s result: %v", c.Name, err)
		}
		matrix[string(key)] = result
		names[string(key)] = append(names[string(key)], c.Name)
	}

	grouped := make(map[string]Result, len(matrix))
	for key, result := range matrix {
		grouped[strings.Join(names[key], ",")] = result
	}
	got, err := json.MarshalIndent(grouped, "", "  ")
	if err != nil {
		t.Fatalf("marshal matrix: %v", err)
	}
	return append(got, '\n')
}

// lineDiff reports the differing lines between want and got, or "" when they
// match. Golden files here run to hundreds of base64-heavy lines, so printing
// both in full buries the one field that moved.
func lineDiff(want, got string) string {
	if want == got {
		return ""
	}
	wantLines, gotLines := strings.Split(want, "\n"), strings.Split(got, "\n")
	var b strings.Builder
	for i := 0; i < max(len(wantLines), len(gotLines)); i++ {
		w, g := at(wantLines, i), at(gotLines, i)
		if w == g {
			continue
		}
		fmt.Fprintf(&b, "  line %d:\n    want: %s\n    got:  %s\n", i+1, w, g)
	}
	return b.String()
}

func at(lines []string, i int) string {
	if i < len(lines) {
		return lines[i]
	}
	return "<missing>"
}
