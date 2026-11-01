package openai

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestStreamRequireFinishReason(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name      string
		strict    bool
		reason    string
		done      bool
		wantError bool
		wantTool  bool
	}{
		{name: "strict socket close", strict: true, wantError: true},
		{name: "strict done without reason", strict: true, done: true, wantError: true},
		{name: "compatible default", wantTool: true},
		{name: "strict tool completion", strict: true, reason: "tool_calls", done: true, wantTool: true},
		{name: "strict stop with tool", strict: true, reason: "stop", done: true, wantTool: true},
		{name: "strict truncated tool", strict: true, reason: "length", done: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprint(w, `data: {"id":"chat_1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"echo","arguments":"{\"value\":7}"}}]},"finish_reason":null}]}`+"\n\n")
				if tc.reason != "" {
					fmt.Fprintf(w, "data: {\"id\":\"chat_1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":%q}]}\n\n", tc.reason)
				}
				if tc.done {
					fmt.Fprint(w, "data: [DONE]\n\n")
				}
			}))
			defer s.Close()
			opts := []Option{WithBaseURL(s.URL), WithAPIKey("fixture")}
			if tc.strict {
				opts = append(opts, WithLanguageModelOptions(WithLanguageModelRequireFinishReason()))
			}
			p, err := New(opts...)
			require.NoError(t, err)
			m, err := p.LanguageModel(t.Context(), "fixture-model")
			require.NoError(t, err)
			stream, err := m.Stream(t.Context(), fantasy.Call{Prompt: fantasy.Prompt{fantasy.NewUserMessage("echo")}})
			require.NoError(t, err)
			var errors, finishes, tools, ends int
			for part := range stream {
				switch part.Type {
				case fantasy.StreamPartTypeError:
					errors++
					requireRetryableUnexpectedEOF(t, part.Error)
				case fantasy.StreamPartTypeFinish:
					finishes++
				case fantasy.StreamPartTypeToolCall:
					tools++
				case fantasy.StreamPartTypeToolInputEnd:
					ends++
				}
			}
			if tc.wantError {
				require.Equal(t, 1, errors)
				require.Zero(t, finishes)
				require.Zero(t, ends)
			} else {
				require.Zero(t, errors)
				require.Equal(t, 1, finishes)
			}
			if tc.wantTool {
				require.Equal(t, 1, tools)
			} else {
				require.Zero(t, tools)
			}
		})
	}
}
