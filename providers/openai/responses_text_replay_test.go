package openai

import (
	"encoding/json"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestResponsesTextReplay(t *testing.T) {
	t.Parallel()
	server := newMockServer()
	defer server.close()
	server.response = map[string]any{
		"id": "resp_1", "status": "completed",
		"output": []any{map[string]any{"id": "msg_1", "type": "message", "role": "assistant", "status": "completed", "phase": "final_answer", "content": []any{map[string]any{"type": "output_text", "text": "done", "annotations": []any{}}, map[string]any{"type": "output_text", "text": "again", "annotations": []any{}}}}},
	}
	model := newResponsesProvider(t, server.server.URL)
	response, err := model.Generate(t.Context(), fantasy.Call{Prompt: testPrompt})
	require.NoError(t, err)
	text := response.Content[0].(fantasy.TextContent)
	require.NotEmpty(t, text.ProviderMetadata)
	// Use the public JSON format that history storage uses.
	data, err := json.Marshal(text.ProviderMetadata)
	require.NoError(t, err)
	var raw map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(data, &raw))
	metadata, err := fantasy.UnmarshalProviderOptions(raw)
	require.NoError(t, err)
	for _, store := range []bool{false, true} {
		_, err := model.Generate(t.Context(), fantasy.Call{
			Prompt:          fantasy.Prompt{{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{fantasy.TextPart{Text: text.Text, ProviderOptions: fantasy.ProviderOptions(metadata)}, fantasy.TextPart{Text: "again", ProviderOptions: fantasy.ProviderOptions(metadata)}, fantasy.TextPart{Text: "next", ProviderOptions: fantasy.ProviderOptions{Name: &ResponsesTextMetadata{ItemID: "msg_2"}}}}}},
			ProviderOptions: NewResponsesProviderOptions(&ResponsesProviderOptions{Store: &store}),
		})
		require.NoError(t, err)
		input := server.calls[len(server.calls)-1].body["input"].([]any)[0].(map[string]any)
		if store {
			require.NotContains(t, input, "id")
			require.NotContains(t, input, "phase")
		} else {
			require.Len(t, server.calls[len(server.calls)-1].body["input"], 2)
			require.Len(t, input["content"], 2)
			require.Equal(t, "again", input["content"].([]any)[1].(map[string]any)["text"])
			require.Equal(t, "msg_1", input["id"])
			require.Equal(t, "final_answer", input["phase"])
			require.Equal(t, "completed", input["status"])
			require.Equal(t, "output_text", input["content"].([]any)[0].(map[string]any)["type"])
		}
	}
}

func TestResponsesStreamTextMetadata(t *testing.T) {
	t.Parallel()
	server := newStreamingMockServer()
	defer server.close()
	server.chunks = []string{
		responsesSSEEvent("response.output_item.added", `{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","type":"message","phase":"commentary","role":"assistant","content":[]}}`),
		responsesSSEEvent("response.output_text.delta", `{"type":"response.output_text.delta","item_id":"msg_1","delta":"done"}`),
		responsesSSEEvent("response.output_item.done", `{"type":"response.output_item.done","output_index":0,"item":{"id":"msg_1","type":"message","phase":"final_answer","role":"assistant","content":[]}}`),
		responsesSSEEvent("response.completed", `{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[]}}`),
	}
	stream, err := newResponsesProvider(t, server.server.URL).Stream(t.Context(), fantasy.Call{Prompt: testPrompt})
	require.NoError(t, err)
	seen := 0
	for part := range stream {
		require.NoError(t, part.Error)
		if part.Type != fantasy.StreamPartTypeTextStart && part.Type != fantasy.StreamPartTypeTextEnd {
			continue
		}
		seen++
		require.NotEmpty(t, part.ProviderMetadata)
		data, err := json.Marshal(part.ProviderMetadata[Name])
		require.NoError(t, err)
		var fields map[string]any
		require.NoError(t, json.Unmarshal(data, &fields))
		require.Equal(t, "msg_1", fields["data"].(map[string]any)["item_id"])
		phase := "commentary"
		if part.Type == fantasy.StreamPartTypeTextEnd {
			phase = "final_answer"
		}
		require.Equal(t, phase, fields["data"].(map[string]any)["phase"])
	}
	require.Equal(t, 2, seen)
}
