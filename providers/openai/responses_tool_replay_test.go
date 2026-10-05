package openai

import (
	"encoding/json"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestResponsesFunctionCallMetadata(t *testing.T) {
	t.Parallel()
	server := newMockServer()
	defer server.close()
	server.response = map[string]any{
		"id": "resp_1", "status": "completed", "output": []any{
			map[string]any{"id": "rs_1", "type": "reasoning", "encrypted_content": "encrypted", "summary": []any{}},
			map[string]any{"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "echo", "arguments": "{}"},
		},
	}
	model := newResponsesProvider(t, server.server.URL)
	response, err := model.Generate(t.Context(), fantasy.Call{Prompt: testPrompt})
	require.NoError(t, err)
	call := response.Content[1].(fantasy.ToolCallContent)
	require.Equal(t, "fc_1", call.ProviderMetadata[Name].(*ResponsesToolCallMetadata).ItemID)
	reasoning := response.Content[0].(fantasy.ReasoningContent)
	for _, id := range []string{"fc_1", "ctc_1", "foreign", ""} {
		for _, store := range []bool{false, true} {
			msg := fantasy.Message{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{
				fantasy.ReasoningPart{ProviderOptions: fantasy.ProviderOptions(reasoning.ProviderMetadata)},
				fantasy.ToolCallPart{ToolCallID: call.ToolCallID, ToolName: call.ToolName, Input: call.Input, ProviderOptions: fantasy.ProviderOptions{Name: &ResponsesToolCallMetadata{ItemID: id}}},
			}}
			data, err := json.Marshal(msg)
			require.NoError(t, err)
			var saved fantasy.Message
			require.NoError(t, json.Unmarshal(data, &saved))
			_, err = model.Generate(t.Context(), fantasy.Call{Prompt: fantasy.Prompt{saved}, ProviderOptions: NewResponsesProviderOptions(&ResponsesProviderOptions{Store: &store})})
			require.NoError(t, err)
			input := server.calls[len(server.calls)-1].body["input"].([]any)
			index := 0
			if !store {
				require.Equal(t, "reasoning", input[0].(map[string]any)["type"])
				index = 1
			}
			wire := input[index].(map[string]any)
			require.Equal(t, "function_call", wire["type"])
			require.Equal(t, "call_1", wire["call_id"])
			if !store && id == "fc_1" {
				require.Equal(t, id, wire["id"])
			} else {
				require.NotContains(t, wire, "id")
			}
		}
	}
}

func TestResponsesStreamFunctionCallMetadata(t *testing.T) {
	t.Parallel()
	server := newStreamingMockServer()
	defer server.close()
	server.chunks = []string{
		responsesSSEEvent("response.output_item.added", `{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"echo","arguments":""}}`),
		responsesSSEEvent("response.function_call_arguments.delta", `{"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc_1","delta":"{}"}`),
		responsesSSEEvent("response.output_item.done", `{"type":"response.output_item.done","output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"echo","arguments":"{}"}}`),
		responsesSSEEvent("response.completed", `{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[]}}`),
	}
	stream, err := newResponsesProvider(t, server.server.URL).Stream(t.Context(), fantasy.Call{Prompt: testPrompt})
	require.NoError(t, err)
	seen := 0
	for part := range stream {
		require.NoError(t, part.Error)
		switch part.Type {
		case fantasy.StreamPartTypeToolInputStart, fantasy.StreamPartTypeToolInputDelta, fantasy.StreamPartTypeToolInputEnd, fantasy.StreamPartTypeToolCall:
			seen++
			require.Equal(t, "call_1", part.ID)
			require.Equal(t, "fc_1", part.ProviderMetadata[Name].(*ResponsesToolCallMetadata).ItemID)
		}
	}
	require.Equal(t, 4, seen)
}

func TestResponsesGenerateKeepsReasoningCallOrder(t *testing.T) {
	t.Parallel()
	server := newMockServer()
	defer server.close()
	server.response = map[string]any{"id": "resp_1", "status": "completed", "output": []any{
		map[string]any{"id": "rs_1", "type": "reasoning", "encrypted_content": "first", "summary": []any{}},
		map[string]any{"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "echo", "arguments": "{}"},
		map[string]any{"id": "rs_2", "type": "reasoning", "encrypted_content": "second", "summary": []any{}},
		map[string]any{"id": "fc_2", "type": "function_call", "call_id": "call_2", "name": "echo", "arguments": "{}"},
	}}
	response, err := newResponsesProvider(t, server.server.URL).Generate(t.Context(), fantasy.Call{Prompt: testPrompt})
	require.NoError(t, err)
	require.Len(t, response.Content, 4)
	require.Equal(t, fantasy.ContentTypeReasoning, response.Content[0].GetType())
	require.Equal(t, "call_1", response.Content[1].(fantasy.ToolCallContent).ToolCallID)
	require.Equal(t, fantasy.ContentTypeReasoning, response.Content[2].GetType())
	require.Equal(t, "call_2", response.Content[3].(fantasy.ToolCallContent).ToolCallID)
}

func TestResponsesNilToolCallMetadata(t *testing.T) {
	t.Parallel()
	input, warnings := toResponsesPrompt(fantasy.Prompt{{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{
		fantasy.ToolCallPart{ToolCallID: "call_1", ToolName: "echo", Input: "{}", ProviderOptions: fantasy.ProviderOptions{Name: (*ResponsesToolCallMetadata)(nil)}},
	}}}, "system", false)
	require.Empty(t, warnings)
	require.Len(t, input, 1)
	require.False(t, input[0].OfFunctionCall.ID.Valid())
}
