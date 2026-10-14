package openai

import (
	"context"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

// The Responses API documents `incomplete_details.reason` as one of
// "max_output_tokens", "max_messages", "content_filter", "steered". A turn cut
// short by a content filter can carry a partially-serialized function_call, and
// fantasy's agent suppresses dispatch on FinishReasonContentFilter
// (agent.go: `abnormalFinish`). Rewriting that reason into FinishReasonToolCalls
// makes a truncated call look like a complete tool-call turn again.
func TestResponsesGenerate_ContentFilterToolCallsKeepFilterReason(t *testing.T) {
	t.Parallel()

	server := newMockServer()
	defer server.close()
	server.response = map[string]any{
		"id":     "resp_filter",
		"object": "response",
		"model":  "gpt-4.1",
		"output": []any{
			map[string]any{
				"type":      "function_call",
				"id":        "fc_filter",
				"call_id":   "call_filter",
				"name":      "test-tool",
				"arguments": `{"value":"trunc`,
				"status":    "completed",
			},
		},
		"status":             "incomplete",
		"incomplete_details": map[string]any{"reason": "content_filter"},
		"usage": map[string]any{
			"input_tokens":  10,
			"output_tokens": 20,
			"total_tokens":  30,
		},
	}

	model := newResponsesProvider(t, server.server.URL)

	resp, err := model.Generate(context.Background(), fantasy.Call{
		Prompt: testPrompt,
		Tools: []fantasy.Tool{fantasy.FunctionTool{
			Name: "test-tool",
			InputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"value": map[string]any{"type": "string"}},
			},
		}},
	})
	require.NoError(t, err)
	require.Equal(t, fantasy.FinishReasonContentFilter, resp.FinishReason)
	require.NotEmpty(t, resp.Warnings)
	require.Contains(t, resp.Warnings[0].Message, "arguments may be truncated")
	for _, c := range resp.Content {
		require.NotEqual(t, fantasy.ContentTypeToolCall, c.GetType(), "truncated tool call should be suppressed")
	}
}

func TestResponsesStream_ContentFilterToolCallsKeepFilterReason(t *testing.T) {
	t.Parallel()

	chunks := []string{
		responsesSSEEvent("response.created", `{"type":"response.created","response":{"id":"resp_filter","status":"in_progress","output":[]}}`),
		responsesSSEEvent("response.output_item.added", `{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_filter","call_id":"call_filter","name":"test-tool","status":"in_progress","arguments":""}}`),
		responsesSSEEvent("response.function_call_arguments.delta", `{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"value\":\"tr"}`),
		responsesSSEEvent("response.output_item.done", `{"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_filter","call_id":"call_filter","name":"test-tool","status":"completed","arguments":"{\"value\":\"tr"}}`),
		responsesSSEEvent("response.incomplete", `{"type":"response.incomplete","response":{"id":"resp_filter","status":"incomplete","output":[],"incomplete_details":{"reason":"content_filter"},"usage":{"input_tokens":10,"output_tokens":20,"total_tokens":30}}}`),
	}

	sms := newStreamingMockServer()
	defer sms.close()
	sms.chunks = chunks

	model := newResponsesProvider(t, sms.server.URL)

	stream, err := model.Stream(context.Background(), fantasy.Call{
		Prompt: testPrompt,
		Tools: []fantasy.Tool{fantasy.FunctionTool{
			Name: "test-tool",
			InputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"value": map[string]any{"type": "string"}},
			},
		}},
	})
	require.NoError(t, err)

	var parts []fantasy.StreamPart
	stream(func(part fantasy.StreamPart) bool {
		parts = append(parts, part)
		return true
	})

	var finish *fantasy.StreamPart
	var hasWarning bool
	for i, part := range parts {
		if part.Type == fantasy.StreamPartTypeFinish {
			finish = &parts[i]
		}
		if part.Type == fantasy.StreamPartTypeWarnings {
			hasWarning = true
		}
	}
	require.NotNil(t, finish)
	require.Equal(t, fantasy.FinishReasonContentFilter, finish.FinishReason)
	require.True(t, hasWarning, "expected truncation warning")
}
