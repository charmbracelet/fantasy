package openai

import (
	"context"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestStream_ToolCallsWithoutIndex(t *testing.T) {
	t.Parallel()

	chunk := func(body string) string {
		return `data: {"id":"chatcmpl-noindex","object":"chat.completion.chunk","created":1711357598,"model":"gemini-3.5-flash","choices":[{"index":0,"delta":` + body + `,"finish_reason":null}]}` + "\n\n"
	}
	finish := []string{
		`data: {"id":"chatcmpl-noindex","object":"chat.completion.chunk","created":1711357598,"model":"gemini-3.5-flash","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n",
		"data: [DONE]\n\n",
	}

	tests := []struct {
		name   string
		chunks []string
		want   map[string]string
	}{
		{
			// Gemini's OpenAI-compatible endpoint sends every parallel call whole,
			// in its own chunk, with no index.
			name: "whole calls",
			chunks: []string{
				chunk(`{"role":"assistant","tool_calls":[{"id":"call_paris","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]}`),
				chunk(`{"tool_calls":[{"id":"call_london","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"London\"}"}}]}`),
			},
			want: map[string]string{
				"call_paris":  `{"city":"Paris"}`,
				"call_london": `{"city":"London"}`,
			},
		},
		{
			name: "argument fragments continue the last call",
			chunks: []string{
				chunk(`{"role":"assistant","tool_calls":[{"id":"call_paris","type":"function","function":{"name":"get_weather","arguments":"{\"city\":"}}]}`),
				chunk(`{"tool_calls":[{"id":"call_paris","function":{"arguments":"\"Paris\"}"}}]}`),
				chunk(`{"tool_calls":[{"id":"call_london","type":"function","function":{"name":"get_weather","arguments":"{\"ci"}}]}`),
				chunk(`{"tool_calls":[{"function":{"arguments":"ty\":\"London\"}"}}]}`),
			},
			want: map[string]string{
				"call_paris":  `{"city":"Paris"}`,
				"call_london": `{"city":"London"}`,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			server := newStreamingMockServer()
			defer server.close()
			server.chunks = append(append([]string(nil), tt.chunks...), finish...)

			provider, err := New(
				WithAPIKey("test-api-key"),
				WithBaseURL(server.server.URL),
			)
			require.NoError(t, err)
			model, _ := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")

			stream, err := model.Stream(context.Background(), fantasy.Call{
				Prompt: testPrompt,
				Tools:  []fantasy.Tool{fantasy.FunctionTool{Name: "get_weather"}},
			})
			require.NoError(t, err)

			parts, err := collectStreamParts(stream)
			require.NoError(t, err)

			got := map[string]string{}
			for _, part := range parts {
				if part.Type == fantasy.StreamPartTypeToolCall {
					require.NotContains(t, got, part.ID, "duplicate tool call %s", part.ID)
					got[part.ID] = part.ToolCallInput
				}
			}
			require.Equal(t, tt.want, got)
		})
	}
}
