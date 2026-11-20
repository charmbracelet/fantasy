package google

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// newInvalidJSONModeServer serves a JSON-mode GenerateContent response whose
// payload does not satisfy the requested schema, so parsing cannot produce an
// object.
func newInvalidJSONModeServer(t *testing.T) *httptest.Server {
	t.Helper()

	body, err := json.Marshal(map[string]any{
		"candidates": []map[string]any{{
			"content": map[string]any{"role": "model", "parts": []map[string]any{
				{"text": `{"person":"not the schema"}`},
			}},
			"finishReason": "STOP",
		}},
		"usageMetadata": map[string]any{
			"promptTokenCount":     4,
			"candidatesTokenCount": 30,
			"totalTokenCount":      34,
		},
	})
	require.NoError(t, err)

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	}))
}

// An object generation whose payload fails parsing or validation must surface
// as *fantasy.NoObjectGeneratedError carrying the turn's usage and finish
// reason. Callers switch on that type (fantasy.IsNoObjectGeneratedError) and
// read Usage off it; a bare *schema.ParseError carries neither.
func TestGenerateObjectParseFailureIsNoObjectGeneratedError(t *testing.T) {
	t.Parallel()

	server := newInvalidJSONModeServer(t)
	defer server.Close()

	_, err := objectStreamModel(t, server.URL).GenerateObject(t.Context(), objectStreamCall())

	var nogErr *fantasy.NoObjectGeneratedError
	require.ErrorAs(t, err, &nogErr, "want *fantasy.NoObjectGeneratedError, got %T", err)
	assert.Equal(t, `{"person":"not the schema"}`, nogErr.RawText)
	assert.Equal(t, int64(4), nogErr.Usage.InputTokens)
	assert.Equal(t, int64(30), nogErr.Usage.OutputTokens)
	assert.Equal(t, int64(34), nogErr.Usage.TotalTokens)
	assert.Equal(t, fantasy.FinishReasonStop, nogErr.FinishReason)
}
