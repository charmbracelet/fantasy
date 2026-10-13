package google

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// newJSONModeServer serves a Gemini JSON-mode structured-output stream. The
// candidate carries the schema-shaped text and, when finishReason is non-empty,
// that reason on a trailing chunk. An empty finishReason models an upstream
// that closed before it ever sent candidate.finishReason.
func newJSONModeServer(t *testing.T, finishReason string) *httptest.Server {
	t.Helper()

	usage := map[string]any{
		"promptTokenCount":     5,
		"candidatesTokenCount": 4,
		"totalTokenCount":      9,
	}
	chunk := func(candidate map[string]any) map[string]any {
		return map[string]any{"candidates": []map[string]any{candidate}, "usageMetadata": usage}
	}

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		body := chunk(map[string]any{
			"content": map[string]any{"role": "model", "parts": []map[string]any{
				{"text": `{"answer":"hello"}`},
			}},
		})
		encoded, err := json.Marshal(body)
		require.NoError(t, err)

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: " + string(encoded) + "\r\n\r\n"))

		if finishReason == "" {
			return
		}
		last, err := json.Marshal(chunk(map[string]any{
			"content":      map[string]any{"role": "model", "parts": []map[string]any{}},
			"finishReason": finishReason,
		}))
		require.NoError(t, err)
		_, _ = w.Write([]byte("data: " + string(last) + "\r\n\r\n"))
	}))
}

func objectStreamModel(t *testing.T, baseURL string) fantasy.LanguageModel {
	t.Helper()

	p, err := New(WithVertex("test-project", "us-central1"), WithBaseURL(baseURL), WithSkipAuth(true))
	require.NoError(t, err)
	model, err := p.LanguageModel(t.Context(), "gemini-2.5-flash")
	require.NoError(t, err)
	return model
}

func objectStreamCall() fantasy.ObjectCall {
	return fantasy.ObjectCall{
		Prompt: fantasy.Prompt{{
			Role:    fantasy.MessageRoleUser,
			Content: []fantasy.MessagePart{fantasy.TextPart{Text: "Hi"}},
		}},
		Schema: fantasy.Schema{
			Type: "object",
			Properties: map[string]*fantasy.Schema{
				"answer": {Type: "string"},
			},
			Required: []string{"answer"},
		},
	}
}

// A JSON-mode object stream that closed before any candidate carried a
// finishReason was reported as a clean "stop", so a truncated object looked
// like a completed one and the retry middleware never engaged. The main Stream
// path already surfaces a retryable incomplete-stream error for the same input
// (google.go), and the OpenAI adapter has a regression test for it
// (TestChatCompletionsStreamObject_RequiresFinishReasonBeforeFinish).
func TestStreamObjectWithoutFinishReasonIsIncomplete(t *testing.T) {
	t.Parallel()

	server := newJSONModeServer(t, "")
	defer server.Close()

	stream, err := objectStreamModel(t, server.URL).StreamObject(t.Context(), objectStreamCall())
	require.NoError(t, err)

	var finishes []fantasy.ObjectStreamPart
	var errorParts []fantasy.ObjectStreamPart
	for part := range stream {
		switch part.Type {
		case fantasy.ObjectStreamPartTypeFinish:
			finishes = append(finishes, part)
		case fantasy.ObjectStreamPartTypeError:
			errorParts = append(errorParts, part)
		}
	}

	require.Len(t, errorParts, 1, "a stream with no candidate.finishReason must report an error")
	assert.Empty(t, finishes, "a stream with no candidate.finishReason must not report finish")

	var providerErr *fantasy.ProviderError
	require.True(t, errors.As(errorParts[0].Error, &providerErr),
		"want a *fantasy.ProviderError, got %T", errorParts[0].Error)
	assert.True(t, providerErr.IsRetryable(), "the incomplete-stream error must be retryable")
}

// A stream that does carry a terminal reason must still finish normally.
func TestStreamObjectWithFinishReasonFinishes(t *testing.T) {
	t.Parallel()

	server := newJSONModeServer(t, "STOP")
	defer server.Close()

	stream, err := objectStreamModel(t, server.URL).StreamObject(t.Context(), objectStreamCall())
	require.NoError(t, err)

	var finishes []fantasy.ObjectStreamPart
	for part := range stream {
		switch part.Type {
		case fantasy.ObjectStreamPartTypeFinish:
			finishes = append(finishes, part)
		case fantasy.ObjectStreamPartTypeError:
			t.Fatalf("unexpected error part: %v", part.Error)
		}
	}

	require.Len(t, finishes, 1)
	assert.Equal(t, fantasy.FinishReasonStop, finishes[0].FinishReason)
}
