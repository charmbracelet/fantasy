package google

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// newFinishServer serves a Gemini response whose candidate carries a functionCall and
// the given finishReason. Gemini reports the finish reason on a later chunk than the
// part it belongs to, and streaming requests are answered with SSE.
func newFinishServer(t *testing.T, finishReason string) *httptest.Server {
	t.Helper()

	part := map[string]any{"functionCall": map[string]any{
		"name": "test-tool",
		"args": map[string]any{"value": "trunc"},
	}}
	usage := map[string]any{
		"promptTokenCount":     5,
		"candidatesTokenCount": 2,
		"totalTokenCount":      7,
	}
	chunk := func(candidate map[string]any) map[string]any {
		return map[string]any{"candidates": []map[string]any{candidate}, "usageMetadata": usage}
	}

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := chunk(map[string]any{
			"content": map[string]any{"role": "model", "parts": []map[string]any{part}},
		})
		last := chunk(map[string]any{
			"content":      map[string]any{"role": "model", "parts": []map[string]any{}},
			"finishReason": finishReason,
		})

		encoded := func(v map[string]any) string {
			b, err := json.Marshal(v)
			require.NoError(t, err)
			return string(b)
		}

		if r.URL.Query().Get("alt") == "sse" {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: " + encoded(body) + "\r\n\r\n"))
			_, _ = w.Write([]byte("data: " + encoded(last) + "\r\n\r\n"))
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{{
				"content":      map[string]any{"role": "model", "parts": []map[string]any{part}},
				"finishReason": finishReason,
			}},
			"usageMetadata": usage,
		})
	}))
}

func testModel(t *testing.T, baseURL string) fantasy.LanguageModel {
	t.Helper()

	p, err := New(WithVertex("test-project", "us-central1"), WithBaseURL(baseURL), WithSkipAuth(true))
	require.NoError(t, err)
	model, err := p.LanguageModel(t.Context(), "gemini-2.5-flash")
	require.NoError(t, err)
	return model
}

func testPrompt() fantasy.Prompt {
	return fantasy.Prompt{{
		Role:    fantasy.MessageRoleUser,
		Content: []fantasy.MessagePart{fantasy.TextPart{Text: "Hi"}},
	}}
}

// A function call that is cut off by the token limit must keep its terminal
// finish reason, so the agent records it without dispatching truncated
// arguments (CHARM-2020).
func TestGeneratePreservesTerminalFinishReasonWithToolCalls(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		providerReason string
		want           fantasy.FinishReason
	}{
		{"MAX_TOKENS", fantasy.FinishReasonLength},
		{"SAFETY", fantasy.FinishReasonContentFilter},
		{"MALFORMED_FUNCTION_CALL", fantasy.FinishReasonError},
		{"STOP", fantasy.FinishReasonToolCalls},
	} {
		t.Run(tc.providerReason, func(t *testing.T) {
			t.Parallel()
			server := newFinishServer(t, tc.providerReason)
			defer server.Close()

			resp, err := testModel(t, server.URL).Generate(
				t.Context(), fantasy.Call{Prompt: testPrompt()},
			)
			require.NoError(t, err)

			assert.Equal(t, tc.want, resp.FinishReason)
			if tc.want != fantasy.FinishReasonToolCalls {
				require.Len(t, resp.Warnings, 1)
				assert.Contains(t, resp.Warnings[0].Message, "arguments may be truncated")
			}
		})
	}
}

func TestStreamPreservesTerminalFinishReasonWithToolCalls(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		providerReason string
		want           fantasy.FinishReason
	}{
		{"MAX_TOKENS", fantasy.FinishReasonLength},
		{"STOP", fantasy.FinishReasonToolCalls},
	} {
		t.Run(tc.providerReason, func(t *testing.T) {
			t.Parallel()
			server := newFinishServer(t, tc.providerReason)
			defer server.Close()

			stream, err := testModel(t, server.URL).Stream(
				t.Context(), fantasy.Call{Prompt: testPrompt()},
			)
			require.NoError(t, err)

			var (
				finish   fantasy.FinishReason
				warnings []fantasy.CallWarning
			)
			for part := range stream {
				switch part.Type {
				case fantasy.StreamPartTypeFinish:
					finish = part.FinishReason
				case fantasy.StreamPartTypeWarnings:
					warnings = part.Warnings
				}
			}
			require.NotEmpty(t, finish)

			assert.Equal(t, tc.want, finish)
			if tc.want != fantasy.FinishReasonToolCalls {
				require.Len(t, warnings, 1)
				assert.Contains(t, warnings[0].Message, "arguments may be truncated")
			}
		})
	}
}
