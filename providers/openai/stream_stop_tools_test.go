package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestStreamStopWithTools(t *testing.T) {
	for _, args := range []string{`{"a":1}`, `{"a":`, ``} {
		t.Run(fmt.Sprintf("arguments_%q", args), func(t *testing.T) {
			server := newStreamingMockServer()
			defer server.close()
			encoded, err := json.Marshal(args)
			require.NoError(t, err)
			server.chunks = []string{
				fmt.Sprintf(`data: {"id":"test","object":"chat.completion.chunk","created":1,"model":"test","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_x","type":"function","function":{"name":"echo","arguments":%s}}]},"finish_reason":null}]}`, encoded) + "\n\n",
				`data: {"id":"test","object":"chat.completion.chunk","created":1,"model":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n",
				"data: [DONE]\n\n",
			}
			provider, err := New(WithAPIKey("test"), WithBaseURL(server.server.URL))
			require.NoError(t, err)
			model, err := provider.LanguageModel(t.Context(), "test")
			require.NoError(t, err)
			stream, err := model.Stream(context.Background(), fantasy.Call{Prompt: testPrompt})
			require.NoError(t, err)
			parts, err := collectStreamParts(stream)
			require.NoError(t, err)
			calls, finishes, failures, ends := 0, 0, 0, 0
			for _, part := range parts {
				switch part.Type {
				case fantasy.StreamPartTypeToolCall:
					calls++
				case fantasy.StreamPartTypeToolInputEnd:
					ends++
				case fantasy.StreamPartTypeFinish:
					finishes++
					require.Equal(t, fantasy.FinishReasonToolCalls, part.FinishReason)
				case fantasy.StreamPartTypeError:
					failures++
					var e *fantasy.ProviderError
					require.ErrorAs(t, part.Error, &e)
					require.True(t, e.IsRetryable())
				}
			}
			if json.Valid([]byte(args)) {
				require.Equal(t, 1, calls)
				require.Equal(t, 1, finishes)
				require.Zero(t, failures)
			} else {
				require.Zero(t, calls)
				require.Zero(t, finishes)
				require.Zero(t, ends)
				require.Equal(t, 1, failures)
			}
		})
	}
}

func TestAgentStreamStopWithToolsDispatchesAndContinues(t *testing.T) {
	t.Parallel()
	server := newStreamingMockServer()
	defer server.close()
	server.chunks = []string{
		`data: {"id":"test","object":"chat.completion.chunk","created":1,"model":"test","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_x","type":"function","function":{"name":"echo","arguments":"{\"a\":1}"}}]},"finish_reason":null}]}` + "\n\n",
		`data: {"id":"test","object":"chat.completion.chunk","created":1,"model":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n",
		"data: [DONE]\n\n",
	}
	provider, err := New(WithAPIKey("test"), WithBaseURL(server.server.URL))
	require.NoError(t, err)
	model, err := provider.LanguageModel(t.Context(), "test")
	require.NoError(t, err)
	calls := 0
	tool := fantasy.NewAgentTool("echo", "Echo the argument", func(ctx context.Context, input struct {
		A int `json:"a"`
	}, call fantasy.ToolCall) (fantasy.ToolResponse, error) { calls++; require.Equal(t, 1, input.A); return fantasy.NewTextResponse("ok"), nil })
	result, err := fantasy.NewAgent(model, fantasy.WithTools(tool)).Stream(t.Context(), fantasy.AgentStreamCall{
		Prompt: "Use echo", StopWhen: []fantasy.StopCondition{fantasy.StepCountIs(2)},
	})
	require.NoError(t, err)
	require.Len(t, result.Steps, 2, "the normalized tool turn must continue")
	require.Equal(t, 2, calls)
	for _, step := range result.Steps {
		require.Equal(t, fantasy.FinishReasonToolCalls, step.FinishReason)
		require.Len(t, step.Content.ToolResults(), 1)
	}
}
