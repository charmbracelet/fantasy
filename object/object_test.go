package object

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/schema"
	"github.com/stretchr/testify/require"
)

type scriptedStreamModel struct {
	parts []fantasy.StreamPart
	fed   *int
}

func (scriptedStreamModel) Generate(context.Context, fantasy.Call) (*fantasy.Response, error) {
	return nil, nil
}

func (m scriptedStreamModel) Stream(context.Context, fantasy.Call) (fantasy.StreamResponse, error) {
	return func(yield func(fantasy.StreamPart) bool) {
		for _, part := range m.parts {
			if m.fed != nil {
				(*m.fed)++
			}
			if !yield(part) {
				return
			}
		}
	}, nil
}

func (scriptedStreamModel) GenerateObject(context.Context, fantasy.ObjectCall) (*fantasy.ObjectResponse, error) {
	return nil, nil
}

func (scriptedStreamModel) StreamObject(context.Context, fantasy.ObjectCall) (fantasy.ObjectStreamResponse, error) {
	return nil, nil
}

func (scriptedStreamModel) Provider() string { return "scripted" }
func (scriptedStreamModel) Model() string    { return "scripted" }

type person struct {
	Name string `json:"name"`
	Age  int    `json:"age"`
}

type objectStreamer struct {
	name   string
	stream func(context.Context, fantasy.LanguageModel, fantasy.ObjectCall) (fantasy.ObjectStreamResponse, error)
	delta  fantasy.StreamPartType
}

func streamers() []objectStreamer {
	return []objectStreamer{
		{name: "StreamWithText", stream: StreamWithText, delta: fantasy.StreamPartTypeTextDelta},
		{name: "StreamWithTool", stream: StreamWithTool, delta: fantasy.StreamPartTypeToolInputDelta},
	}
}

func personCall(t *testing.T, repair schema.ObjectRepairFunc) fantasy.ObjectCall {
	t.Helper()
	return fantasy.ObjectCall{
		Prompt:     fantasy.Prompt{fantasy.NewUserMessage("give me a person")},
		Schema:     schema.Generate(reflect.TypeOf(person{})),
		RepairText: repair,
	}
}

func repairWithText(text string) schema.ObjectRepairFunc {
	return func(context.Context, string, error) (string, error) { return text, nil }
}

func deltaParts(deltaType fantasy.StreamPartType, deltas ...string) []fantasy.StreamPart {
	parts := make([]fantasy.StreamPart, 0, len(deltas))
	for _, delta := range deltas {
		parts = append(parts, fantasy.StreamPart{Type: deltaType, Delta: delta})
	}
	return parts
}

func streamParts(t *testing.T, stream fantasy.ObjectStreamResponse, stopAfter int) []fantasy.ObjectStreamPart {
	t.Helper()
	var parts []fantasy.ObjectStreamPart
	for part := range stream {
		parts = append(parts, part)
		if stopAfter > 0 && len(parts) >= stopAfter {
			break
		}
	}
	return parts
}

func emittedObjects(parts []fantasy.ObjectStreamPart) []any {
	var objects []any
	for _, part := range parts {
		if part.Type == fantasy.ObjectStreamPartTypeObject {
			objects = append(objects, part.Object)
		}
	}
	return objects
}

func TestStreamingEmitsObjectsAsTheyValidate(t *testing.T) {
	t.Parallel()

	ada := map[string]any{"name": "Ada", "age": float64(36)}

	tests := []struct {
		name        string
		deltas      []string
		repair      schema.ObjectRepairFunc
		wantObjects []any
	}{
		{
			name:        "partial deltas that end in a valid object",
			deltas:      []string{`{"name": "Ada", `, `"age": 36}`},
			wantObjects: []any{ada},
		},
		{
			name:        "truncated json is repaired by the parser",
			deltas:      []string{`{"name": "Ada", "age": 36`},
			wantObjects: []any{ada},
		},
		{
			name:        "unparseable deltas go through the repair callback",
			deltas:      []string{`not json`},
			repair:      repairWithText(`{"name": "Ada", "age": 36}`),
			wantObjects: []any{ada},
		},
		{
			name:        "an object that parses the same twice is emitted once",
			deltas:      []string{`{"name": "Ada", "age": 36}`, ` `},
			wantObjects: []any{ada},
		},
		{
			name:        "the repair callback result is emitted once",
			deltas:      []string{`not json`, `not json`},
			repair:      repairWithText(`{"name": "Ada", "age": 36}`),
			wantObjects: []any{ada},
		},
		{
			name:        "an object that never validates is not emitted",
			deltas:      []string{`{"other": 1}`},
			wantObjects: nil,
		},
		{
			name:        "an empty delta leaves the stream undefined",
			deltas:      []string{""},
			wantObjects: nil,
		},
	}

	for _, streamer := range streamers() {
		for _, tt := range tests {
			t.Run(streamer.name+"/"+tt.name, func(t *testing.T) {
				t.Parallel()

				parts := append(deltaParts(streamer.delta, tt.deltas...),
					fantasy.StreamPart{Type: fantasy.StreamPartTypeFinish, FinishReason: fantasy.FinishReasonStop})
				stream, err := streamer.stream(context.Background(), scriptedStreamModel{parts: parts}, personCall(t, tt.repair))
				require.NoError(t, err)

				require.Equal(t, tt.wantObjects, emittedObjects(streamParts(t, stream, 0)))
			})
		}
	}
}

func TestStreamWithToolEmitsCompletedToolCalls(t *testing.T) {
	t.Parallel()

	parts := append(deltaParts(fantasy.StreamPartTypeToolInputDelta, `{"name": "Ada", `, `"age": 36}`),
		fantasy.StreamPart{Type: fantasy.StreamPartTypeToolCall, ToolCallName: "generate_object", ToolCallInput: `{"name": "Ada", "age": 36}`},
		fantasy.StreamPart{Type: fantasy.StreamPartTypeFinish, FinishReason: fantasy.FinishReasonStop},
	)

	stream, err := StreamWithTool(context.Background(), scriptedStreamModel{parts: parts}, personCall(t, nil))
	require.NoError(t, err)

	require.Equal(t, []any{map[string]any{"name": "Ada", "age": float64(36)}}, emittedObjects(streamParts(t, stream, 0)))
}

func TestStreamingReportsNoObjectGenerated(t *testing.T) {
	t.Parallel()

	for _, streamer := range streamers() {
		t.Run(streamer.name, func(t *testing.T) {
			t.Parallel()

			parts := append(deltaParts(streamer.delta, `{"other": 1}`),
				fantasy.StreamPart{Type: fantasy.StreamPartTypeFinish, Usage: fantasy.Usage{OutputTokens: 5}, FinishReason: fantasy.FinishReasonStop})
			stream, err := streamer.stream(context.Background(), scriptedStreamModel{parts: parts}, personCall(t, nil))
			require.NoError(t, err)

			got := streamParts(t, stream, 0)
			require.Len(t, got, 1)
			require.Equal(t, fantasy.ObjectStreamPartTypeError, got[0].Type)

			var noObject *fantasy.NoObjectGeneratedError
			require.ErrorAs(t, got[0].Error, &noObject)
			require.Equal(t, `{"other": 1}`, noObject.RawText)
			require.Equal(t, int64(5), noObject.Usage.OutputTokens)
			require.Equal(t, fantasy.FinishReasonStop, noObject.FinishReason)
		})
	}
}

func TestStreamingForwardsFinishMetadata(t *testing.T) {
	t.Parallel()

	warnings := []fantasy.CallWarning{{Type: fantasy.CallWarningTypeOther, Message: "watch out"}}

	for _, streamer := range streamers() {
		t.Run(streamer.name, func(t *testing.T) {
			t.Parallel()

			parts := append(deltaParts(streamer.delta, `{"name": "Ada", "age": 36}`),
				fantasy.StreamPart{Type: fantasy.StreamPartTypeWarnings, Warnings: warnings},
				fantasy.StreamPart{Type: fantasy.StreamPartTypeFinish, Usage: fantasy.Usage{OutputTokens: 4}, FinishReason: fantasy.FinishReasonStop},
			)
			stream, err := streamer.stream(context.Background(), scriptedStreamModel{parts: parts}, personCall(t, nil))
			require.NoError(t, err)

			got := streamParts(t, stream, 0)
			require.Len(t, got, 2)
			require.Equal(t, fantasy.ObjectStreamPartTypeObject, got[0].Type)
			require.Equal(t, fantasy.ObjectStreamPartTypeFinish, got[1].Type)
			require.Equal(t, warnings, got[1].Warnings)
			require.Equal(t, int64(4), got[1].Usage.OutputTokens)
			require.Equal(t, fantasy.FinishReasonStop, got[1].FinishReason)
		})
	}
}

func TestStreamingForwardsStreamErrors(t *testing.T) {
	t.Parallel()

	errBoom := errors.New("boom")

	for _, streamer := range streamers() {
		t.Run(streamer.name, func(t *testing.T) {
			t.Parallel()

			parts := append(deltaParts(streamer.delta, `{"name": "Ada", "age": 36}`),
				fantasy.StreamPart{Type: fantasy.StreamPartTypeError, Error: errBoom})
			stream, err := streamer.stream(context.Background(), scriptedStreamModel{parts: parts}, personCall(t, nil))
			require.NoError(t, err)

			got := streamParts(t, stream, 0)
			require.Len(t, got, 2)
			require.Equal(t, fantasy.ObjectStreamPartTypeObject, got[0].Type)
			require.Equal(t, fantasy.ObjectStreamPartTypeError, got[1].Type)
			require.ErrorIs(t, got[1].Error, errBoom)
		})
	}
}

func TestStreamingStopsWhenTheConsumerStops(t *testing.T) {
	t.Parallel()

	for _, streamer := range streamers() {
		t.Run(streamer.name, func(t *testing.T) {
			t.Parallel()

			parts := append(deltaParts(streamer.delta, `{"name": "Ada", "age": 36}`, `{"name": "Bob", "age": 40}`),
				fantasy.StreamPart{Type: fantasy.StreamPartTypeFinish, FinishReason: fantasy.FinishReasonStop})
			fed := 0
			stream, err := streamer.stream(context.Background(), scriptedStreamModel{parts: parts, fed: &fed}, personCall(t, nil))
			require.NoError(t, err)

			got := streamParts(t, stream, 1)
			require.Len(t, got, 1)
			require.Equal(t, fantasy.ObjectStreamPartTypeObject, got[0].Type)
			require.Equal(t, 1, fed, "the stream kept pulling from the model after the consumer stopped")
		})
	}
}
