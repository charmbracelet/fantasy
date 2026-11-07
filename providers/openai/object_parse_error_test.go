package openai

import (
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const invalidObjectJSON = `{"person":"not the schema"}`

func objectParseFailureCall() fantasy.ObjectCall {
	return fantasy.ObjectCall{
		Prompt: fantasy.Prompt{fantasy.NewUserMessage("Generate a person")},
		Schema: fantasy.Schema{
			Type: "object",
			Properties: map[string]*fantasy.Schema{
				"name": {Type: "string"},
			},
			Required: []string{"name"},
		},
		SchemaName: "Person",
	}
}

// An object generation whose payload fails parsing or validation must surface
// as *fantasy.NoObjectGeneratedError carrying the turn's usage and finish
// reason. Callers switch on that type (fantasy.IsNoObjectGeneratedError) and
// read Usage off it; a bare *schema.ParseError carries neither.
func TestGenerateObjectParseFailureIsNoObjectGeneratedError(t *testing.T) {
	t.Parallel()

	server := newMockServer()
	defer server.close()
	server.prepareJSONResponse(map[string]any{"content": invalidObjectJSON})

	provider, err := New(WithAPIKey("test-api-key"), WithBaseURL(server.server.URL))
	require.NoError(t, err)
	model, err := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")
	require.NoError(t, err)

	_, err = model.GenerateObject(t.Context(), objectParseFailureCall())

	var nogErr *fantasy.NoObjectGeneratedError
	require.ErrorAs(t, err, &nogErr, "want *fantasy.NoObjectGeneratedError, got %T", err)
	assert.Equal(t, invalidObjectJSON, nogErr.RawText)
	assert.Equal(t, int64(34), nogErr.Usage.TotalTokens)
	assert.Equal(t, fantasy.FinishReasonStop, nogErr.FinishReason)
}

// The Responses API path shares the same contract.
func TestResponsesGenerateObjectParseFailureIsNoObjectGeneratedError(t *testing.T) {
	t.Parallel()

	server := newMockServer()
	defer server.close()
	server.response = map[string]any{
		"id":     "resp_parse_failure",
		"object": "response",
		"status": "completed",
		"output": []map[string]any{{
			"type": "message",
			"role": "assistant",
			"content": []map[string]any{
				{"type": "output_text", "text": invalidObjectJSON},
			},
		}},
		"usage": map[string]any{
			"input_tokens":  4,
			"output_tokens": 30,
			"total_tokens":  34,
		},
	}

	_, err := newResponsesProvider(t, server.server.URL).GenerateObject(t.Context(), objectParseFailureCall())

	var nogErr *fantasy.NoObjectGeneratedError
	require.ErrorAs(t, err, &nogErr, "want *fantasy.NoObjectGeneratedError, got %T", err)
	assert.Equal(t, invalidObjectJSON, nogErr.RawText)
	assert.Equal(t, int64(34), nogErr.Usage.TotalTokens)
}
