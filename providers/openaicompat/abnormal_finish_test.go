package openaicompat

import (
	"fmt"
	"strings"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

// abnormalFinishSSE renders a text-only stream (reasoning, then an answer)
// that ends with the given finish reason. The final chunk carries nothing but
// the finish reason, which is where truncation used to go unnoticed.
func abnormalFinishSSE(finishReason string) string {
	return `{"id":"x","created":1,"model":"m","choices":[{"index":0,"delta":{"reasoning_content":"thinking"},"finish_reason":null}]}` + "\n" +
		`{"id":"x","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"partial answer"},"finish_reason":null}]}` + "\n" +
		fmt.Sprintf(`{"id":"x","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":%q}]}`, finishReason)
}

// Text-only turns ending abnormally (token limit, content filter, or an
// unrecognized reason) must warn: without it a truncated answer is
// indistinguishable from a complete one downstream.
func TestReplay_AbnormalFinishWarnsWithoutToolCalls(t *testing.T) {
	tests := []struct {
		finishReason    string
		wantFinish      fantasy.FinishReason
		wantMsgContains string
	}{
		{finishReason: "length", wantFinish: fantasy.FinishReasonLength, wantMsgContains: "token limit"},
		{finishReason: "content_filter", wantFinish: fantasy.FinishReasonContentFilter, wantMsgContains: "content filter"},
		{finishReason: "auto", wantFinish: fantasy.FinishReasonUnknown, wantMsgContains: "unrecognized finish reason"},
	}

	for _, tt := range tests {
		t.Run(tt.finishReason, func(t *testing.T) {
			srv, _ := serveSSE(t, abnormalFinishSSE(tt.finishReason))
			provider, err := New(WithBaseURL(srv.URL), WithAPIKey("x"))
			require.NoError(t, err)
			lm, err := provider.LanguageModel(t.Context(), "m")
			require.NoError(t, err)

			parts := streamParts(t, lm, fantasy.Prompt{
				{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "x"}}},
			})

			var finish *fantasy.StreamPart
			var warnings []fantasy.CallWarning
			for i, p := range parts {
				switch p.Type {
				case fantasy.StreamPartTypeFinish:
					finish = &parts[i]
				case fantasy.StreamPartTypeWarnings:
					warnings = append(warnings, p.Warnings...)
				}
			}
			require.NotNil(t, finish, "types: %v", partTypes(parts))
			require.Equal(t, tt.wantFinish, finish.FinishReason)
			require.Len(t, warnings, 1, "types: %v", partTypes(parts))
			require.Contains(t, warnings[0].Message, tt.wantMsgContains)
			require.Contains(t, warnings[0].Message, tt.finishReason)
			require.Equal(t, tt.finishReason, warnings[0].RawFinishReason)
		})
	}
}

func TestReplay_StopFinishStaysQuiet(t *testing.T) {
	srv, _ := serveSSE(t, abnormalFinishSSE("stop"))
	provider, err := New(WithBaseURL(srv.URL), WithAPIKey("x"))
	require.NoError(t, err)
	lm, err := provider.LanguageModel(t.Context(), "m")
	require.NoError(t, err)

	parts := streamParts(t, lm, fantasy.Prompt{
		{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "x"}}},
	})

	var warnings []fantasy.CallWarning
	for _, p := range parts {
		if p.Type == fantasy.StreamPartTypeWarnings {
			warnings = append(warnings, p.Warnings...)
		}
	}
	require.Empty(t, warnings, "types: %v", partTypes(parts))
}

// A finish chunk carrying neither content nor reasoning must close an open
// reasoning block; the recorded step otherwise loses the block entirely.
func TestReplay_ReasoningClosedByFinishChunk(t *testing.T) {
	const sse = `{"id":"x","created":1,"model":"m","choices":[{"index":0,"delta":{"reasoning_content":"thinking"},"finish_reason":null}]}` + "\n" +
		`{"id":"x","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}`

	srv, _ := serveSSE(t, sse)
	provider, err := New(WithBaseURL(srv.URL), WithAPIKey("x"))
	require.NoError(t, err)
	lm, err := provider.LanguageModel(t.Context(), "m")
	require.NoError(t, err)

	parts := streamParts(t, lm, fantasy.Prompt{
		{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "x"}}},
	})

	var starts, deltas, ends int
	var sb strings.Builder
	for _, p := range parts {
		switch p.Type {
		case fantasy.StreamPartTypeReasoningStart:
			starts++
		case fantasy.StreamPartTypeReasoningDelta:
			deltas++
			sb.WriteString(p.Delta)
		case fantasy.StreamPartTypeReasoningEnd:
			ends++
		}
	}
	require.Equal(t, 1, starts, "types: %v", partTypes(parts))
	require.Equal(t, 1, deltas, "types: %v", partTypes(parts))
	require.Equal(t, "thinking", sb.String())
	require.Equal(t, 1, ends, "the finish chunk must close the open reasoning block; types: %v", partTypes(parts))
}
