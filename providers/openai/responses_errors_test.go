package openai

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestResponsesTypedStreamErrors(t *testing.T) {
	t.Parallel()
	for _, event := range []struct{ name, data, code, kind, message string }{
		{"response.failed", `{"type":"response.failed","response":{"id":"resp_1","status":"failed","error":{"code":"server_error","type":"api_error","message":"boom"}}}`, "server_error", "api_error", "boom"},
		{"error", `{"type":"error","code":"invalid_prompt","message":"bad prompt"}`, "invalid_prompt", "error", "bad prompt"},
		{"error", `{"error":{"type":"invalid_request_error","code":"bad_argument","message":"bad argument"}}`, "bad_argument", "invalid_request_error", "bad argument"},
		{"error", `{"error":{"type":"invalid_request_error","code":"bad_argument","message":"bad argument","status_code":429}}`, "bad_argument", "invalid_request_error", "bad argument"},
	} {
		t.Run(event.data, func(t *testing.T) {
			t.Parallel()
			server := newStreamingMockServer()
			defer server.close()
			server.chunks = []string{
				responsesSSEEvent("response.output_text.delta", `{"type":"response.output_text.delta","item_id":"msg_1","delta":"{\"answer\":\"yes\"}"}`),
				responsesSSEEvent(event.name, event.data),
			}
			model := newResponsesProvider(t, server.server.URL)
			stream, err := model.Stream(t.Context(), fantasy.Call{Prompt: testPrompt})
			require.NoError(t, err)
			var failure error
			for part := range stream {
				if part.Error != nil {
					failure = part.Error
				}
			}
			require.Error(t, failure)
			var provider *fantasy.ProviderError
			require.True(t, errors.As(failure, &provider))
			require.Equal(t, 200, provider.StatusCode)
			var typed *ResponsesError
			require.ErrorAs(t, failure, &typed)
			require.Equal(t, event.code, typed.Code)
			require.Equal(t, event.kind, typed.Type)
			require.Equal(t, event.message, typed.Message)
			schema := fantasy.Schema{Type: "object", Properties: map[string]*fantasy.Schema{"answer": {Type: "string"}}, Required: []string{"answer"}}
			objects, err := model.StreamObject(t.Context(), fantasy.ObjectCall{Prompt: testPrompt, Schema: schema})
			require.NoError(t, err)
			failure = nil
			for part := range objects {
				if part.Error != nil {
					failure = part.Error
				}
			}
			require.True(t, errors.As(failure, &provider))
			require.Equal(t, 200, provider.StatusCode)
			require.NotContains(t, failure.Error(), "test-api-key")
			require.NotContains(t, failure.Error(), "Authorization")
		})
	}
}

func TestResponsesHTTPErrorFields(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"code":"invalid_prompt","type":"invalid_request_error","message":"bad prompt"}}`))
	}))
	defer server.Close()
	model := newResponsesProvider(t, server.URL)
	schema := fantasy.Schema{Type: "object", Properties: map[string]*fantasy.Schema{"answer": {Type: "string"}}}
	_, err := model.Generate(t.Context(), fantasy.Call{Prompt: testPrompt})
	checkResponsesHTTPError(t, err)
	_, err = model.GenerateObject(t.Context(), fantasy.ObjectCall{Prompt: testPrompt, Schema: schema})
	checkResponsesHTTPError(t, err)
	stream, err := model.Stream(t.Context(), fantasy.Call{Prompt: testPrompt})
	require.NoError(t, err)
	seen := false
	for part := range stream {
		if part.Error != nil {
			seen = true
			checkResponsesHTTPError(t, part.Error)
		}
	}
	require.True(t, seen)
	objects, err := model.StreamObject(t.Context(), fantasy.ObjectCall{Prompt: testPrompt, Schema: schema})
	require.NoError(t, err)
	seen = false
	for part := range objects {
		if part.Error != nil {
			seen = true
			checkResponsesHTTPError(t, part.Error)
		}
	}
	require.True(t, seen)
}

func checkResponsesHTTPError(t *testing.T, err error) {
	t.Helper()
	var typed *ResponsesError
	require.ErrorAs(t, err, &typed)
	require.Equal(t, "invalid_prompt", typed.Code)
	require.Equal(t, "invalid_request_error", typed.Type)
	require.Equal(t, "bad prompt", typed.Message)
	require.Equal(t, http.StatusBadRequest, typed.StatusCode)
	require.NotContains(t, err.Error(), "test-api-key")
	require.NotContains(t, err.Error(), "Authorization")
	// The existing SDK error keeps a request dump. Do not serialize it.
	require.NotEmpty(t, typed.RequestBody)
	data, marshalErr := json.Marshal(typed)
	require.NoError(t, marshalErr)
	require.NotContains(t, string(data), "test-api-key")
	require.NotContains(t, string(data), "RequestBody")
}

func TestResponsesGenerateBodyErrorFields(t *testing.T) {
	t.Parallel()
	server := newMockServer()
	defer server.close()
	server.response = map[string]any{"id": "resp_1", "status": "failed", "error": map[string]any{"code": "server_error", "type": "api_error", "message": "boom"}}
	model := newResponsesProvider(t, server.server.URL)
	_, err := model.Generate(t.Context(), fantasy.Call{Prompt: testPrompt})
	var typed *ResponsesError
	require.ErrorAs(t, err, &typed)
	require.Equal(t, "server_error", typed.Code)
	require.Equal(t, "api_error", typed.Type)
	require.Equal(t, "boom", typed.Message)
	require.Equal(t, http.StatusOK, typed.StatusCode)
}
