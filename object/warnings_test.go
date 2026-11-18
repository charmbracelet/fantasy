package object

import (
	"context"
	"fmt"
	"reflect"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/schema"
	"github.com/stretchr/testify/require"
)

type warningsStreamModel struct {
	parts []fantasy.StreamPart
}

func (m *warningsStreamModel) Generate(context.Context, fantasy.Call) (*fantasy.Response, error) {
	return nil, fmt.Errorf("not implemented")
}

func (m *warningsStreamModel) Stream(context.Context, fantasy.Call) (fantasy.StreamResponse, error) {
	return func(yield func(fantasy.StreamPart) bool) {
		for _, part := range m.parts {
			if !yield(part) {
				return
			}
		}
	}, nil
}

func (m *warningsStreamModel) GenerateObject(context.Context, fantasy.ObjectCall) (*fantasy.ObjectResponse, error) {
	return nil, fmt.Errorf("not implemented")
}

func (m *warningsStreamModel) StreamObject(context.Context, fantasy.ObjectCall) (fantasy.ObjectStreamResponse, error) {
	return nil, fmt.Errorf("not implemented")
}

func (m *warningsStreamModel) Provider() string { return "mock" }
func (m *warningsStreamModel) Model() string    { return "mock" }

type testRecipe struct {
	Name string `json:"name"`
}

func TestStreamWithToolAccumulatesWarnings(t *testing.T) {
	t.Parallel()

	model := &warningsStreamModel{parts: []fantasy.StreamPart{
		{Type: fantasy.StreamPartTypeTextDelta, Delta: `{"name":"lasagna"}`},
		{Type: fantasy.StreamPartTypeWarnings, Warnings: []fantasy.CallWarning{{Type: fantasy.CallWarningTypeOther, Message: "first"}}},
		{Type: fantasy.StreamPartTypeWarnings, Warnings: []fantasy.CallWarning{{Type: fantasy.CallWarningTypeOther, Message: "second"}}},
		{Type: fantasy.StreamPartTypeFinish, FinishReason: fantasy.FinishReasonStop},
	}}

	stream, err := StreamWithTool(t.Context(), model, fantasy.ObjectCall{
		Prompt: fantasy.Prompt{fantasy.NewUserMessage("generate")},
		Schema: schema.Generate(reflect.TypeOf(testRecipe{})),
	})
	require.NoError(t, err)

	var finish *fantasy.ObjectStreamPart
	for part := range stream {
		if part.Type == fantasy.ObjectStreamPartTypeFinish {
			p := part
			finish = &p
		}
	}
	require.NotNil(t, finish)
	require.Len(t, finish.Warnings, 2)
	require.Equal(t, "first", finish.Warnings[0].Message)
	require.Equal(t, "second", finish.Warnings[1].Message)
}

func TestStreamWithTextAccumulatesWarnings(t *testing.T) {
	t.Parallel()

	model := &warningsStreamModel{parts: []fantasy.StreamPart{
		{Type: fantasy.StreamPartTypeTextDelta, Delta: `{"name":"lasagna"}`},
		{Type: fantasy.StreamPartTypeWarnings, Warnings: []fantasy.CallWarning{{Type: fantasy.CallWarningTypeOther, Message: "first"}}},
		{Type: fantasy.StreamPartTypeWarnings, Warnings: []fantasy.CallWarning{{Type: fantasy.CallWarningTypeOther, Message: "second"}}},
		{Type: fantasy.StreamPartTypeFinish, FinishReason: fantasy.FinishReasonStop},
	}}

	stream, err := StreamWithText(t.Context(), model, fantasy.ObjectCall{
		Prompt: fantasy.Prompt{fantasy.NewUserMessage("generate")},
		Schema: schema.Generate(reflect.TypeOf(testRecipe{})),
	})
	require.NoError(t, err)

	var finish *fantasy.ObjectStreamPart
	for part := range stream {
		if part.Type == fantasy.ObjectStreamPartTypeFinish {
			p := part
			finish = &p
		}
	}
	require.NotNil(t, finish)
	require.Len(t, finish.Warnings, 2)
	require.Equal(t, "first", finish.Warnings[0].Message)
	require.Equal(t, "second", finish.Warnings[1].Message)
}
