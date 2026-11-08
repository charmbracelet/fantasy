package openrouter

import (
	"encoding/json"
	"strings"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/google"
	"charm.land/fantasy/providers/openai"
	openaisdk "github.com/charmbracelet/openai-go"
	"github.com/stretchr/testify/require"
)

func mustChunk(t *testing.T, raw string) openaisdk.ChatCompletionChunk {
	t.Helper()
	var chunk openaisdk.ChatCompletionChunk
	require.NoError(t, json.Unmarshal([]byte(raw), &chunk))
	return chunk
}

// openReasoningCtx returns a context carrying an open reasoning block in the
// given format, as a previous reasoning chunk would have left it.
func openReasoningCtx(format string) map[string]any {
	state := &currentReasoningState{format: format}
	if strings.HasPrefix(format, "openai-responses") || strings.HasPrefix(format, "xai-responses") {
		state.metadata = &openai.ResponsesReasoningMetadata{}
	}
	if strings.HasPrefix(format, "google-gemini") {
		state.googleMetadata = &google.ReasoningMetadata{}
	}
	return map[string]any{reasoningStartedCtx: state}
}

// TestLanguageModelStreamExtra_ClosesReasoningOnFinish covers the provider
// side of a lost reasoning block: when the finish chunk carries no content and
// no reasoning_details, an open reasoning block must still be closed,
// otherwise consumers drop it from the recorded step content.
func TestLanguageModelStreamExtra_ClosesReasoningOnFinish(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		format    string
		chunk     string
		wantEnd   bool
		wantDelta bool
	}{
		{
			name:    "finish chunk with empty delta closes an openai-responses block",
			format:  "openai-responses",
			chunk:   `{"id":"x","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}`,
			wantEnd: true,
		},
		{
			name:    "finish chunk with empty delta closes an anthropic block missing its signature",
			format:  "anthropic-claude",
			chunk:   `{"id":"x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			wantEnd: true,
		},
		{
			name:    "finish chunk with empty delta closes a google block",
			format:  "google-gemini",
			chunk:   `{"id":"x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			wantEnd: true,
		},
		{
			name:    "finish chunk with empty delta closes any other format block",
			format:  "deepseek-chat",
			chunk:   `{"id":"x","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}`,
			wantEnd: true,
		},
		{
			name:   "empty non-finish chunk keeps an openai-responses block open",
			format: "openai-responses",
			chunk:  `{"id":"x","choices":[{"index":0,"delta":{},"finish_reason":null}]}`,
		},
		{
			name:   "empty non-finish chunk keeps an anthropic block open for its late signature",
			format: "anthropic-claude",
			chunk:  `{"id":"x","choices":[{"index":0,"delta":{},"finish_reason":null}]}`,
		},
		{
			name:    "content boundary still closes the block",
			format:  "openai-responses",
			chunk:   `{"id":"x","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}`,
			wantEnd: true,
		},
		{
			name:      "reasoning delta batched with the finish reason closes after the delta",
			format:    "deepseek-chat",
			chunk:     `{"id":"x","choices":[{"index":0,"delta":{"reasoning_details":[{"format":"deepseek-chat","text":"tail"}]},"finish_reason":"length"}]}`,
			wantEnd:   true,
			wantDelta: true,
		},
		{
			name:      "anthropic reasoning delta batched with the finish reason closes after the delta",
			format:    "anthropic-claude",
			chunk:     `{"id":"x","choices":[{"index":0,"delta":{"reasoning_details":[{"format":"anthropic-claude","text":"more"}]},"finish_reason":"stop"}]}`,
			wantEnd:   true,
			wantDelta: true,
		},
		{
			name:      "google reasoning delta batched with the finish reason closes after the delta",
			format:    "google-gemini",
			chunk:     `{"id":"x","choices":[{"index":0,"delta":{"reasoning_details":[{"format":"google-gemini","type":"reasoning.text","text":"tail"}]},"finish_reason":"stop"}]}`,
			wantEnd:   true,
			wantDelta: true,
		},
		{
			name:      "openai-responses summary batched with the finish reason closes after the delta",
			format:    "openai-responses",
			chunk:     `{"id":"x","choices":[{"index":0,"delta":{"reasoning_details":[{"format":"openai-responses","summary":"tail","index":0}]},"finish_reason":"length"}]}`,
			wantEnd:   true,
			wantDelta: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			ctx := openReasoningCtx(tt.format)
			var parts []fantasy.StreamPart
			yield := func(part fantasy.StreamPart) bool {
				parts = append(parts, part)
				return true
			}

			updated, shouldContinue := languageModelStreamExtra(mustChunk(t, tt.chunk), yield, ctx)
			require.True(t, shouldContinue)

			var ends, deltas int
			for _, part := range parts {
				switch part.Type {
				case fantasy.StreamPartTypeReasoningEnd:
					ends++
				case fantasy.StreamPartTypeReasoningDelta:
					deltas++
				}
			}

			if tt.wantEnd {
				require.Equal(t, 1, ends, "expected exactly one ReasoningEnd, parts: %+v", parts)
				require.Nil(t, extractReasoningContext(updated), "the block must be marked closed")
			} else {
				require.Equal(t, 0, ends, "block must stay open, parts: %+v", parts)
				require.NotNil(t, extractReasoningContext(updated))
			}
			if tt.wantDelta {
				require.Equal(t, 1, deltas, "the batched reasoning delta must not be lost, parts: %+v", parts)
				require.Equal(t, fantasy.StreamPartTypeReasoningDelta, parts[0].Type,
					"delta must precede the close, parts: %+v", parts)
				require.Equal(t, fantasy.StreamPartTypeReasoningEnd, parts[1].Type,
					"delta must precede the close, parts: %+v", parts)
			}
		})
	}
}
