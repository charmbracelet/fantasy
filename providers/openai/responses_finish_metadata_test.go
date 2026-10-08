package openai

import (
	"encoding/json"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestResponsesRawFinishMetadata(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		status, reason string
		finish         fantasy.FinishReason
	}{
		{"completed", "", fantasy.FinishReasonStop},
		{"incomplete", "max_output_tokens", fantasy.FinishReasonLength},
		{"incomplete", "content_filter", fantasy.FinishReasonContentFilter},
		{"incomplete", "future_reason", fantasy.FinishReasonOther},
	} {
		t.Run(tc.status+tc.reason, func(t *testing.T) {
			t.Parallel()
			response := map[string]any{"id": "resp_1", "status": tc.status, "output": []any{map[string]any{"type": "message", "id": "msg_1", "content": []any{map[string]any{"type": "output_text", "text": `{"answer":"yes"}`}}}}, "incomplete_details": map[string]any{"reason": tc.reason}}
			server := newMockServer()
			defer server.close()
			server.response = response
			model := newResponsesProvider(t, server.server.URL)
			generated, err := model.Generate(t.Context(), fantasy.Call{Prompt: testPrompt})
			require.NoError(t, err)
			require.Equal(t, tc.finish, generated.FinishReason)
			checkResponsesRawFinish(t, generated.ProviderMetadata, tc.status, tc.reason)
			schema := fantasy.Schema{Type: "object", Properties: map[string]*fantasy.Schema{"answer": {Type: "string"}}, Required: []string{"answer"}}
			object, err := model.GenerateObject(t.Context(), fantasy.ObjectCall{Prompt: testPrompt, Schema: schema})
			require.NoError(t, err)
			checkResponsesRawFinish(t, object.ProviderMetadata, tc.status, tc.reason)
			streamServer := newStreamingMockServer()
			defer streamServer.close()
			event := "response." + tc.status
			raw, err := json.Marshal(map[string]any{"type": event, "response": response})
			require.NoError(t, err)
			streamServer.chunks = []string{
				responsesSSEEvent("response.output_text.delta", `{"type":"response.output_text.delta","item_id":"msg_1","delta":"{\"answer\":\"yes\"}"}`),
				responsesSSEEvent(event, string(raw)),
			}
			streamModel := newResponsesProvider(t, streamServer.server.URL)
			stream, err := streamModel.Stream(t.Context(), fantasy.Call{Prompt: testPrompt})
			require.NoError(t, err)
			seen := false
			for part := range stream {
				require.NoError(t, part.Error)
				if part.Type == fantasy.StreamPartTypeFinish {
					seen = true
					require.Equal(t, tc.finish, part.FinishReason)
					checkResponsesRawFinish(t, part.ProviderMetadata, tc.status, tc.reason)
				}
			}
			require.True(t, seen)
			objectStream, err := streamModel.StreamObject(t.Context(), fantasy.ObjectCall{Prompt: testPrompt, Schema: schema})
			require.NoError(t, err)
			seen = false
			for part := range objectStream {
				require.NoError(t, part.Error)
				if part.Type == fantasy.ObjectStreamPartTypeFinish {
					seen = true
					checkResponsesRawFinish(t, part.ProviderMetadata, tc.status, tc.reason)
				}
			}
			require.True(t, seen)
		})
	}
}

func checkResponsesRawFinish(t *testing.T, metadata fantasy.ProviderMetadata, status, reason string) {
	t.Helper()
	data, err := json.Marshal(metadata[Name])
	require.NoError(t, err)
	var wrapper struct {
		Data map[string]any `json:"data"`
	}
	require.NoError(t, json.Unmarshal(data, &wrapper))
	require.Equal(t, status, wrapper.Data["response_status"])
	if reason != "" {
		require.Equal(t, reason, wrapper.Data["raw_finish_reason"])
	}
}
