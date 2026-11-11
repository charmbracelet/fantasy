package google

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"charm.land/fantasy"
)

// streamedUsageChunks are the usage reports from a recorded Gemini stream
// (testdata/TestGoogleObjectGeneration/gemini-3-pro-preview). Each one is a
// running total for the response so far, so the last is the whole response:
// 41 candidates plus 604 thoughts reported apart, or 645 output tokens.
var streamedUsageChunks = []struct {
	text                                string
	prompt, candidates, thoughts, total int
}{
	{`{"name":`, 38, 23, 604, 665},
	{`"fantasy"}`, 38, 41, 604, 683},
	{``, 38, 41, 604, 683},
}

// newStreamedUsageServer answers any streaming request with
// streamedUsageChunks as server-sent events.
func newStreamedUsageServer(t *testing.T) *httptest.Server {
	t.Helper()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for i, chunk := range streamedUsageChunks {
			candidate := map[string]any{
				"content": map[string]any{
					"role":  "model",
					"parts": []map[string]any{{"text": chunk.text}},
				},
			}
			if i == len(streamedUsageChunks)-1 {
				candidate["finishReason"] = "STOP"
			}
			data, err := json.Marshal(map[string]any{
				"candidates": []map[string]any{candidate},
				"usageMetadata": map[string]any{
					"promptTokenCount":     chunk.prompt,
					"candidatesTokenCount": chunk.candidates,
					"thoughtsTokenCount":   chunk.thoughts,
					"totalTokenCount":      chunk.total,
				},
			})
			if err != nil {
				t.Errorf("marshal chunk %d: %v", i, err)
				return
			}
			fmt.Fprintf(w, "data: %s\n\n", data)
		}
	}))
	t.Cleanup(server.Close)
	return server
}

func newStreamedUsageModel(t *testing.T) fantasy.LanguageModel {
	t.Helper()

	server := newStreamedUsageServer(t)
	provider, err := New(WithGeminiAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	model, err := provider.LanguageModel(t.Context(), "gemini-3-pro-preview")
	if err != nil {
		t.Fatalf("LanguageModel: %v", err)
	}
	return model
}

var streamedUsagePrompt = fantasy.Prompt{
	{
		Role:    fantasy.MessageRoleUser,
		Content: []fantasy.MessagePart{fantasy.TextPart{Text: "Hi"}},
	},
}

func assertStreamedUsage(t *testing.T, usage fantasy.Usage) {
	t.Helper()

	if usage.InputTokens != 38 {
		t.Errorf("InputTokens = %d, want 38", usage.InputTokens)
	}
	if usage.OutputTokens != 645 {
		t.Errorf("OutputTokens = %d, want 645", usage.OutputTokens)
	}
	if usage.ReasoningTokens != 604 {
		t.Errorf("ReasoningTokens = %d, want 604", usage.ReasoningTokens)
	}
	if usage.TotalTokens != 683 {
		t.Errorf("TotalTokens = %d, want 683", usage.TotalTokens)
	}
}

// Gemini repeats running totals on every chunk, so adding them up would
// report a multiple of what the response used.
func TestStream_UsageIsTheLastChunksTotals(t *testing.T) {
	t.Parallel()

	stream, err := newStreamedUsageModel(t).Stream(t.Context(), fantasy.Call{Prompt: streamedUsagePrompt})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	var finish *fantasy.StreamPart
	for part := range stream {
		if part.Type == fantasy.StreamPartTypeError {
			t.Fatalf("stream error: %v", part.Error)
		}
		if part.Type == fantasy.StreamPartTypeFinish {
			finish = &part
		}
	}
	if finish == nil {
		t.Fatal("stream ended without a finish part")
	}
	assertStreamedUsage(t, finish.Usage)
}

// Object streaming reads usage off the same chunks and must not add them up
// either.
func TestStreamObject_UsageIsTheLastChunksTotals(t *testing.T) {
	t.Parallel()

	stream, err := newStreamedUsageModel(t).StreamObject(t.Context(), fantasy.ObjectCall{
		Prompt: streamedUsagePrompt,
		Schema: fantasy.Schema{
			Type:       "object",
			Properties: map[string]*fantasy.Schema{"name": {Type: "string"}},
			Required:   []string{"name"},
		},
	})
	if err != nil {
		t.Fatalf("StreamObject: %v", err)
	}

	var finish *fantasy.ObjectStreamPart
	for part := range stream {
		if part.Type == fantasy.ObjectStreamPartTypeError {
			t.Fatalf("stream error: %v", part.Error)
		}
		if part.Type == fantasy.ObjectStreamPartTypeFinish {
			finish = &part
		}
	}
	if finish == nil {
		t.Fatal("stream ended without a finish part")
	}
	assertStreamedUsage(t, finish.Usage)
}
