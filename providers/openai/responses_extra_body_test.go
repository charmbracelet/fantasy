package openai

import (
	"encoding/json"
	"fmt"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestResponsesExtraBody(t *testing.T) {
	t.Parallel()
	options, err := ParseResponsesOptions(map[string]any{"extra_body": map[string]any{
		"prompt_cache_retention": "24h", "prompt_cache_options": map[string]any{"ttl": "30m"},
		"model": "override-model", "temperature": 0.2, "store": false,
		"text": map[string]any{"format": map[string]any{"type": "text"}},
	}})
	require.NoError(t, err)
	data, err := json.Marshal(options)
	require.NoError(t, err)
	savedData, err := fantasy.UnmarshalProviderOptions(map[string]json.RawMessage{Name: data})
	require.NoError(t, err)
	saved := savedData[Name].(*ResponsesProviderOptions)
	server := newMockServer()
	defer server.close()
	server.response = map[string]any{"id": "resp_1", "status": "completed", "output": []any{map[string]any{"type": "message", "id": "msg_1", "content": []any{map[string]any{"type": "output_text", "text": `{"answer":"yes"}`}}}}}
	model := newResponsesProvider(t, server.server.URL)
	_, err = model.Generate(t.Context(), fantasy.Call{Prompt: testPrompt, ProviderOptions: NewResponsesProviderOptions(saved)})
	require.NoError(t, err)
	streamServer := newStreamingMockServer()
	defer streamServer.close()
	streamServer.chunks = []string{responsesSSEEvent("response.completed", `{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[]}}`)}
	stream, err := newResponsesProvider(t, streamServer.server.URL).Stream(t.Context(), fantasy.Call{Prompt: testPrompt, ProviderOptions: NewResponsesProviderOptions(saved)})
	require.NoError(t, err)
	for part := range stream {
		require.NoError(t, part.Error)
	}
	objectSchema := fantasy.Schema{Type: "object", Properties: map[string]*fantasy.Schema{"answer": {Type: "string"}}, Required: []string{"answer"}}
	_, err = model.GenerateObject(t.Context(), fantasy.ObjectCall{Prompt: testPrompt, Schema: objectSchema, ProviderOptions: NewResponsesProviderOptions(saved)})
	require.NoError(t, err)
	for _, call := range append(server.calls, streamServer.calls...) {
		require.Equal(t, "24h", call.body["prompt_cache_retention"])
		require.Equal(t, map[string]any{"ttl": "30m"}, call.body["prompt_cache_options"])
		require.Equal(t, "override-model", call.body["model"])
		require.Equal(t, 0.2, call.body["temperature"])
		require.Equal(t, false, call.body["store"])
		require.Equal(t, "text", call.body["text"].(map[string]any)["format"].(map[string]any)["type"])
	}
}

func TestResponsesExtraBodyStoreReplay(t *testing.T) {
	t.Parallel()
	for _, store := range []bool{false, true} {
		t.Run(fmt.Sprint(store), func(t *testing.T) {
			t.Parallel()
			options := &ResponsesProviderOptions{Store: new(!store), ExtraBody: map[string]any{"store": store}}
			prompt := fantasy.Prompt{{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{
				fantasy.ReasoningPart{ProviderOptions: fantasy.ProviderOptions{Name: &ResponsesReasoningMetadata{ItemID: "rs_1", Summary: []string{}, EncryptedContent: new("encrypted"), Finalized: true}}},
				fantasy.ToolCallPart{ToolCallID: "web_1", ToolName: "web_search", ProviderExecuted: true},
			}}}
			model := responsesLanguageModel{modelID: "gpt-5"}
			params, _, err := model.prepareParams(fantasy.Call{Prompt: prompt, ProviderOptions: NewResponsesProviderOptions(options)})
			require.NoError(t, err)
			input := params.Input.OfInputItemList
			require.Len(t, input, 1)
			if store {
				require.NotNil(t, input[0].OfItemReference)
			} else {
				require.NotNil(t, input[0].OfReasoning)
			}
			options.PreviousResponseID = new("resp_1")
			_, _, err = model.prepareParams(fantasy.Call{Prompt: testPrompt, ProviderOptions: NewResponsesProviderOptions(options)})
			if store {
				require.NoError(t, err)
			} else {
				require.EqualError(t, err, previousResponseIDStoreError)
			}
		})
	}
}
