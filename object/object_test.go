package object

import (
	"context"
	"errors"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// stubLanguageModel returns a fixed Generate response and fails every other
// method, so tests can drive GenerateWithTool without a provider.
type stubLanguageModel struct {
	response *fantasy.Response
}

func (m *stubLanguageModel) Generate(context.Context, fantasy.Call) (*fantasy.Response, error) {
	return m.response, nil
}

func (*stubLanguageModel) Stream(context.Context, fantasy.Call) (fantasy.StreamResponse, error) {
	return nil, errors.New("not implemented")
}

func (*stubLanguageModel) GenerateObject(context.Context, fantasy.ObjectCall) (*fantasy.ObjectResponse, error) {
	return nil, errors.New("not implemented")
}

func (*stubLanguageModel) StreamObject(context.Context, fantasy.ObjectCall) (fantasy.ObjectStreamResponse, error) {
	return nil, errors.New("not implemented")
}

func (*stubLanguageModel) Provider() string { return "stub" }

func (*stubLanguageModel) Model() string { return "stub" }

// A tool call whose arguments do not satisfy the requested schema must surface
// as *fantasy.NoObjectGeneratedError carrying the turn's usage and finish
// reason, matching what GenerateWithText does for text mode.
func TestGenerateWithToolParseFailureIsNoObjectGeneratedError(t *testing.T) {
	t.Parallel()

	model := &stubLanguageModel{response: &fantasy.Response{
		Content: fantasy.ResponseContent{
			fantasy.ToolCallContent{
				ToolCallID: "call-1",
				ToolName:   "Person",
				Input:      `{"person":"not the schema"}`,
			},
		},
		FinishReason: fantasy.FinishReasonToolCalls,
		Usage:        fantasy.Usage{InputTokens: 4, OutputTokens: 30, TotalTokens: 34},
	}}

	_, err := GenerateWithTool(t.Context(), model, fantasy.ObjectCall{
		Prompt:     fantasy.Prompt{fantasy.NewUserMessage("Generate a person")},
		SchemaName: "Person",
		Schema: fantasy.Schema{
			Type: "object",
			Properties: map[string]*fantasy.Schema{
				"name": {Type: "string"},
			},
			Required: []string{"name"},
		},
	})

	var nogErr *fantasy.NoObjectGeneratedError
	require.ErrorAs(t, err, &nogErr, "want *fantasy.NoObjectGeneratedError, got %T", err)
	assert.Equal(t, `{"person":"not the schema"}`, nogErr.RawText)
	assert.Equal(t, int64(34), nogErr.Usage.TotalTokens)
	assert.Equal(t, fantasy.FinishReasonToolCalls, nogErr.FinishReason)
}
