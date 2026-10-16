package kronk

import (
	"testing"

	"charm.land/fantasy"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
)

// When a kronk run hits the token limit mid tool call, the started call's
// deltas have already been streamed to the client, the partial call is dropped
// server-side, and the run terminates with finish_reason=length (the kronk
// server replaces the truncated tool output with "Response truncated before
// completion."). The adapter must report that terminal reason verbatim: the
// openai adapter pins the same contract — "Terminal reasons that can cut
// output mid-call — length, content_filter, provider errors — must never be
// rewritten into a tool-call turn: dispatching their partial calls executes
// truncated input (CHARM-2020)."

func TestStreamLengthFinishWithStartedToolCalls(t *testing.T) {
	lengthFinish := model.FinishReasonLength
	client := fakeChatModel{
		modelInfo: model.ModelInfo{ID: "loaded-model", Type: model.ModelTypeDense},
		streamResponse: []model.ChatResponse{
			{Choices: []model.Choice{{Delta: &model.ResponseMessage{ToolCallDeltas: []model.ResponseToolCallDelta{
				{ID: "call-1", Index: 0, Function: model.ResponseToolCallDeltaFunction{Name: "weather"}},
			}}}}},
			{Choices: []model.Choice{{Delta: &model.ResponseMessage{ToolCallDeltas: []model.ResponseToolCallDelta{
				{Index: 0, Function: model.ResponseToolCallDeltaFunction{Arguments: `{"city":"Pa`}},
			}}}}},
			{Choices: []model.Choice{{
				Delta:           &model.ResponseMessage{Content: "Response truncated before completion."},
				FinishReasonPtr: &lengthFinish,
			}}},
		},
	}
	lm := newLanguageModel("model", Name, &client)

	stream, err := lm.Stream(t.Context(), fantasy.Call{})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	var parts []fantasy.StreamPart
	for part := range stream {
		parts = append(parts, part)
	}

	var finish *fantasy.StreamPart
	sawToolInputStart := false
	for i := range parts {
		switch parts[i].Type {
		case fantasy.StreamPartTypeToolCall:
			t.Errorf("stream emitted a ToolCall part for a truncated call: %#v", parts[i])
		case fantasy.StreamPartTypeToolInputStart:
			sawToolInputStart = true
		case fantasy.StreamPartTypeFinish:
			finish = &parts[i]
		}
	}
	if !sawToolInputStart {
		t.Fatal("test setup: expected a ToolInputStart part from the announced call")
	}
	if finish == nil {
		t.Fatalf("parts: got %#v, want a finish part", parts)
	}
	// The turn delivered no tool call: reporting tool-calls here hides the
	// truncation from callers that branch on the finish reason.
	if got, want := finish.FinishReason, fantasy.FinishReasonLength; got != want {
		t.Errorf("FinishReason: got %q, want %q", got, want)
	}
}

func TestGenerateTerminalFinishWithToolCalls(t *testing.T) {
	tests := []struct {
		name         string
		finishReason string
		want         fantasy.FinishReason
	}{
		{name: "length", finishReason: model.FinishReasonLength, want: fantasy.FinishReasonLength},
		{name: "error", finishReason: model.FinishReasonError, want: fantasy.FinishReasonError},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			finishReason := tt.finishReason
			client := fakeChatModel{
				modelInfo: model.ModelInfo{ID: "loaded-model", Type: model.ModelTypeDense},
				chatResponse: model.ChatResponse{
					Choices: []model.Choice{
						{
							FinishReasonPtr: &finishReason,
							Message: &model.ResponseMessage{
								Content: "Response truncated before completion.",
								ToolCalls: []model.ResponseToolCall{
									{
										ID: "call-1",
										Function: model.ResponseToolCallFunction{
											Name:      "weather",
											Arguments: model.ToolCallArguments{"city": "Paris"},
										},
									},
								},
							},
						},
					},
				},
			}
			lm := newLanguageModel("model", Name, &client)

			response, err := lm.Generate(t.Context(), fantasy.Call{})
			if err != nil {
				t.Fatalf("Generate: %v", err)
			}
			if got, want := response.FinishReason, tt.want; got != want {
				t.Errorf("FinishReason: got %q, want %q", got, want)
			}
			// A call from an abnormally terminated turn must not be handed
			// to the agent for dispatch.
			if got := response.Content.ToolCalls(); len(got) != 0 {
				t.Errorf("Content.ToolCalls: got %d, want 0", len(got))
			}
			if len(response.Warnings) == 0 {
				t.Error("Warnings: got none, want a suppression warning")
			}
		})
	}
}
