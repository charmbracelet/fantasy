package openrouter

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openai"
	"github.com/stretchr/testify/require"
)

func TestStreamReasoningKeepsFirstTextChunk(t *testing.T) {
	t.Parallel()

	// DeepSeek and other models without a dedicated format send reasoning in
	// reasoning_details[].text with no summary.
	chunk := func(delta string) string {
		return fmt.Sprintf(`data: {"id":"g","object":"chat.completion.chunk","created":1,"model":"deepseek/deepseek-v4-pro","choices":[{"index":0,"delta":%s,"finish_reason":null}]}`+"\n\n", delta)
	}
	stream := chunk(`{"role":"assistant","content":"","reasoning":"The","reasoning_details":[{"type":"reasoning.text","text":"The","format":"unknown","index":0}]}`) +
		chunk(`{"content":"","reasoning":" user wants","reasoning_details":[{"type":"reasoning.text","text":" user wants","format":"unknown","index":0}]}`) +
		chunk(`{"content":" a haiku."}`) +
		`data: {"id":"g","object":"chat.completion.chunk","created":1,"model":"deepseek/deepseek-v4-pro","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}` + "\n\n" +
		"data: [DONE]\n\n"

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, stream)
	}))
	defer server.Close()

	p, err := New(WithAPIKey("k"), func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithBaseURL(server.URL))
	})
	require.NoError(t, err)
	model, err := p.LanguageModel(t.Context(), "deepseek/deepseek-v4-pro")
	require.NoError(t, err)

	parts, err := model.Stream(t.Context(), fantasy.Call{Prompt: fantasy.Prompt{fantasy.NewUserMessage("hi")}})
	require.NoError(t, err)

	var reasoning strings.Builder
	for part := range parts {
		switch part.Type {
		case fantasy.StreamPartTypeReasoningStart, fantasy.StreamPartTypeReasoningDelta:
			reasoning.WriteString(part.Delta)
		case fantasy.StreamPartTypeError:
			t.Fatalf("stream error: %v", part.Error)
		}
	}
	require.Equal(t, "The user wants", reasoning.String())
}
