package openai

import (
	"context"
	"fmt"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

// finishReasonJSON renders a finish_reason JSON value: null when empty, the
// quoted string otherwise.
func finishReasonJSON(finishReason string) string {
	if finishReason == "" {
		return "null"
	}
	return `"` + finishReason + `"`
}

// chatCompletionToolStream builds SSE chunks for a turn with a single tool
// call whose arguments are complete JSON, ending with the given finish_reason.
func chatCompletionToolStream(finishReason string) []string {
	return []string{
		`data: {"id":"chatcmpl-fw","object":"chat.completion.chunk","created":1711357598,"model":"gpt-3.5-turbo-0125","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_fw","type":"function","function":{"name":"f","arguments":""}}]},"finish_reason":null}]}` + "\n\n",
		`data: {"id":"chatcmpl-fw","object":"chat.completion.chunk","created":1711357598,"model":"gpt-3.5-turbo-0125","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"x\":1}"}}]},"finish_reason":null}]}` + "\n\n",
		fmt.Sprintf(`data: {"id":"chatcmpl-fw","object":"chat.completion.chunk","created":1711357598,"model":"gpt-3.5-turbo-0125","choices":[{"index":0,"delta":{},"finish_reason":%s}]}`+"\n\n", finishReasonJSON(finishReason)),
		`data: {"id":"chatcmpl-fw","object":"chat.completion.chunk","created":1711357598,"model":"gpt-3.5-turbo-0125","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":5,"total_tokens":10}}` + "\n\n",
		"data: [DONE]\n\n",
	}
}

// TestChatCompletionsStream_AbnormalFinishWarnings covers CHARM truncation
// visibility: a turn that ends with finish_reason "length", "content_filter",
// or an unrecognized reason must surface a warning so consumers can tell a
// truncated answer apart from a complete one without checking the finish
// reason themselves — even when the turn produced only text and no tool calls
// were seen.
func TestChatCompletionsStream_AbnormalFinishWarnings(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name            string
		finishReason    string
		toolCall        bool
		wantFinish      fantasy.FinishReason
		wantWarning     bool
		wantRawReason   string
		wantToolCalls   int
		wantMsgContains string
	}{
		{
			name:            "length with text only warns",
			finishReason:    "length",
			wantFinish:      fantasy.FinishReasonLength,
			wantWarning:     true,
			wantRawReason:   "length",
			wantMsgContains: "token limit",
		},
		{
			name:            "content filter with text only warns",
			finishReason:    "content_filter",
			wantFinish:      fantasy.FinishReasonContentFilter,
			wantWarning:     true,
			wantRawReason:   "content_filter",
			wantMsgContains: "content filter",
		},
		{
			name:            "provider error with text only warns",
			finishReason:    "insufficient_system_resource",
			wantFinish:      fantasy.FinishReasonError,
			wantWarning:     true,
			wantRawReason:   "insufficient_system_resource",
			wantMsgContains: "provider reported an error",
		},
		{
			name:            "unrecognized finish with text only warns and keeps the raw reason",
			finishReason:    "credits_exhausted",
			wantFinish:      fantasy.FinishReasonUnknown,
			wantWarning:     true,
			wantRawReason:   "credits_exhausted",
			wantMsgContains: "unrecognized finish reason",
		},
		{
			name:         "stop with text only is quiet",
			finishReason: "stop",
			wantFinish:   fantasy.FinishReasonStop,
		},
		{
			name:          "tool_calls with a tool call is quiet",
			finishReason:  "tool_calls",
			toolCall:      true,
			wantFinish:    fantasy.FinishReasonToolCalls,
			wantToolCalls: 1,
		},
		{
			name:            "length with a tool call warns and suppresses the call",
			finishReason:    "length",
			toolCall:        true,
			wantFinish:      fantasy.FinishReasonLength,
			wantWarning:     true,
			wantRawReason:   "length",
			wantMsgContains: "arguments may be truncated",
		},
		{
			name:            "unrecognized finish with a tool call warns and keeps the raw reason",
			finishReason:    "credits_exhausted",
			toolCall:        true,
			wantFinish:      fantasy.FinishReasonUnknown,
			wantWarning:     true,
			wantRawReason:   "credits_exhausted",
			wantToolCalls:   1,
			wantMsgContains: "unrecognized finish reason",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			server := newStreamingMockServer()
			defer server.close()

			if tt.toolCall {
				server.chunks = chatCompletionToolStream(tt.finishReason)
			} else {
				server.prepareStreamResponse(map[string]any{
					"content":       []string{"Hello", ", World!"},
					"finish_reason": tt.finishReason,
				})
			}

			provider, err := New(
				WithAPIKey("test-api-key"),
				WithBaseURL(server.server.URL),
			)
			require.NoError(t, err)
			model, err := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")
			require.NoError(t, err)

			stream, err := model.Stream(context.Background(), fantasy.Call{
				Prompt: testPrompt,
			})
			require.NoError(t, err)

			parts, err := collectStreamParts(stream)
			require.NoError(t, err)

			var finish *fantasy.StreamPart
			var warnings []fantasy.CallWarning
			toolCalls := 0
			for i, part := range parts {
				switch part.Type {
				case fantasy.StreamPartTypeFinish:
					finish = &parts[i]
				case fantasy.StreamPartTypeWarnings:
					warnings = append(warnings, part.Warnings...)
				case fantasy.StreamPartTypeToolCall:
					toolCalls++
				}
			}

			require.NotNil(t, finish, "expected a Finish part")
			require.Equal(t, tt.wantFinish, finish.FinishReason)
			require.Equal(t, tt.wantToolCalls, toolCalls)

			if !tt.wantWarning {
				require.Empty(t, warnings, "healthy finish must not warn: %+v", warnings)
				return
			}
			require.Len(t, warnings, 1, "expected exactly one abnormal-finish warning")
			require.Equal(t, fantasy.CallWarningTypeOther, warnings[0].Type)
			require.Contains(t, warnings[0].Message, tt.wantMsgContains)
			require.Equal(t, tt.wantRawReason, warnings[0].RawFinishReason,
				"the raw provider finish reason must survive for clients to log")
		})
	}
}

// TestAbnormalFinishWarning pins the message text per finish category and the
// nil cases for healthy finishes.
func TestAbnormalFinishWarning(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		raw          string
		mapped       fantasy.FinishReason
		wantNil      bool
		wantContains string
	}{
		{
			name:         "token limit",
			raw:          "length",
			mapped:       fantasy.FinishReasonLength,
			wantContains: "token limit",
		},
		{
			name:         "content filter",
			raw:          "content_filter",
			mapped:       fantasy.FinishReasonContentFilter,
			wantContains: "content filter",
		},
		{
			name:         "provider error",
			raw:          "insufficient_system_resource",
			mapped:       fantasy.FinishReasonError,
			wantContains: "provider reported an error",
		},
		{
			name:         "unrecognized reason keeps the raw string",
			raw:          "credits_exhausted",
			mapped:       fantasy.FinishReasonUnknown,
			wantContains: "unrecognized finish reason",
		},
		{name: "no finish reason at all is handled elsewhere", raw: "", mapped: fantasy.FinishReasonUnknown, wantNil: true},
		{name: "healthy stop is quiet", raw: "stop", mapped: fantasy.FinishReasonStop, wantNil: true},
		{name: "healthy tool-calls finish is quiet", raw: "tool_calls", mapped: fantasy.FinishReasonToolCalls, wantNil: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			warning := AbnormalFinishWarning(tt.raw, tt.mapped)
			if tt.wantNil {
				require.Nil(t, warning)
				return
			}
			require.NotNil(t, warning)
			require.Equal(t, fantasy.CallWarningTypeOther, warning.Type)
			require.Contains(t, warning.Message, tt.wantContains)
			require.Contains(t, warning.Message, fmt.Sprintf("%q", tt.raw))
			require.Equal(t, tt.raw, warning.RawFinishReason)
		})
	}
}
