package replaytest

import (
	"context"
	"errors"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

type dispatchModel struct {
	fakeStreamModel
	calls         int
	followupError bool
	reuseID       bool
}

func (model *dispatchModel) Stream(context.Context, fantasy.Call) (fantasy.StreamResponse, error) {
	model.calls++
	if model.calls == 2 && model.followupError {
		return nil, errors.New("follow-up failed")
	}
	name := "lookup"
	if model.calls > 1 {
		if !model.reuseID {
			return model.fakeStreamModel.Stream(context.Background(), fantasy.Call{})
		}
		name = "summarize"
	}
	return func(yield func(fantasy.StreamPart) bool) {
		if !yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeToolCall, ID: "shared", ToolCallName: name, ToolCallInput: "{}"}) {
			return
		}
		yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeFinish, FinishReason: fantasy.FinishReasonToolCalls})
	}, nil
}

func TestDispatchSurvivesErrors(t *testing.T) {
	for _, scenario := range []string{"tool", "follow-up"} {
		t.Run(scenario, func(t *testing.T) {
			model := &dispatchModel{followupError: scenario == "follow-up"}
			executions := 0
			tool := fantasy.NewAgentTool("lookup", "lookup", func(context.Context, struct{}, fantasy.ToolCall) (fantasy.ToolResponse, error) {
				executions++
				if scenario == "tool" {
					return fantasy.ToolResponse{}, errors.New("tool failed")
				}
				return fantasy.NewTextResponse("ok"), nil
			})
			record := RunAgentStep(t, model, &Fixture{}, tool)
			require.Equal(t, 1, executions)
			require.NotEmpty(t, record.Error)
			require.Equal(t, []string{"lookup"}, record.ToolsDispatched)
		})
	}
}

func TestDispatchIsolatedFromLaterSteps(t *testing.T) {
	model := &dispatchModel{reuseID: true}
	var executions []string
	lookup := fantasy.NewAgentTool("lookup", "lookup", func(context.Context, struct{}, fantasy.ToolCall) (fantasy.ToolResponse, error) {
		executions = append(executions, "lookup")
		return fantasy.NewTextResponse("ok"), nil
	})
	summarize := fantasy.NewAgentTool("summarize", "summarize", func(context.Context, struct{}, fantasy.ToolCall) (fantasy.ToolResponse, error) {
		executions = append(executions, "summarize")
		response := fantasy.NewTextResponse("done")
		response.StopTurn = true
		return response, nil
	})
	record := RunAgentStep(t, model, &Fixture{}, lookup, summarize)
	require.Empty(t, record.Error)
	require.Equal(t, []string{"lookup", "summarize"}, executions)
	require.Equal(t, []string{"lookup"}, record.ToolsDispatched)
}
