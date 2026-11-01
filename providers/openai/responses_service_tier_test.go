package openai

import (
	"encoding/json"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestResponsesEchoedServiceTier(t *testing.T) {
	t.Parallel()
	for _, tier := range []string{"default", "flex", "priority", ""} {
		for _, status := range []string{"completed", "incomplete"} {
			t.Run(status+"/"+tier, func(t *testing.T) {
				t.Parallel()
				response := map[string]any{"id": "resp_1", "status": status, "output": []any{map[string]any{"type": "message", "id": "msg_1", "content": []any{map[string]any{"type": "output_text", "text": `{"answer":"yes"}`}}}}}
				if tier != "" {
					response["service_tier"] = tier
				}
				options := NewResponsesProviderOptions(&ResponsesProviderOptions{ServiceTier: new(ServiceTierPriority)})
				server := newMockServer()
				defer server.close()
				server.response = response
				model := newResponsesProvider(t, server.server.URL)
				generated, err := model.Generate(t.Context(), fantasy.Call{Prompt: testPrompt, ProviderOptions: options})
				require.NoError(t, err)
				checkResponsesServiceTier(t, generated.ProviderMetadata, tier)
				schema := fantasy.Schema{Type: "object", Properties: map[string]*fantasy.Schema{"answer": {Type: "string"}}, Required: []string{"answer"}}
				object, err := model.GenerateObject(t.Context(), fantasy.ObjectCall{Prompt: testPrompt, Schema: schema, ProviderOptions: options})
				require.NoError(t, err)
				checkResponsesServiceTier(t, object.ProviderMetadata, tier)
				streamServer := newStreamingMockServer()
				defer streamServer.close()
				event := "response." + status
				data, err := json.Marshal(map[string]any{"type": event, "response": response})
				require.NoError(t, err)
				streamServer.chunks = []string{
					responsesSSEEvent("response.created", `{"type":"response.created","response":{"id":"resp_1","status":"in_progress","service_tier":"auto"}}`),
					responsesSSEEvent("response.output_text.delta", `{"type":"response.output_text.delta","item_id":"msg_1","delta":"{\"answer\":\"yes\"}"}`),
					responsesSSEEvent(event, string(data)),
				}
				streamModel := newResponsesProvider(t, streamServer.server.URL)
				stream, err := streamModel.Stream(t.Context(), fantasy.Call{Prompt: testPrompt, ProviderOptions: options})
				require.NoError(t, err)
				seen := false
				for part := range stream {
					require.NoError(t, part.Error)
					if part.Type == fantasy.StreamPartTypeFinish {
						seen = true
						checkResponsesServiceTier(t, part.ProviderMetadata, tier)
					}
				}
				require.True(t, seen)
				objects, err := streamModel.StreamObject(t.Context(), fantasy.ObjectCall{Prompt: testPrompt, Schema: schema, ProviderOptions: options})
				require.NoError(t, err)
				seen = false
				for part := range objects {
					require.NoError(t, part.Error)
					if part.Type == fantasy.ObjectStreamPartTypeFinish {
						seen = true
						checkResponsesServiceTier(t, part.ProviderMetadata, tier)
					}
				}
				require.True(t, seen)
			})
		}
	}
}

func checkResponsesServiceTier(t *testing.T, metadata fantasy.ProviderMetadata, tier string) {
	t.Helper()
	data, err := json.Marshal(metadata)
	require.NoError(t, err)
	var raw map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(data, &raw))
	restored, err := fantasy.UnmarshalProviderMetadata(raw)
	require.NoError(t, err)
	data, err = json.Marshal(restored[Name])
	require.NoError(t, err)
	var wrapper struct {
		Data map[string]any `json:"data"`
	}
	require.NoError(t, json.Unmarshal(data, &wrapper))
	if tier == "" {
		require.NotContains(t, wrapper.Data, "service_tier")
	} else {
		require.Equal(t, tier, wrapper.Data["service_tier"])
	}
}
