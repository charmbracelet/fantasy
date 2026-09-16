package replaytest

import (
	"context"
	"encoding/json"
	"path/filepath"
	"sync"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openaicompat"
	"github.com/stretchr/testify/require"
)

func TestWithCounterIDs(t *testing.T) {
	previous := fantasy.NewID
	fantasy.NewID = func() string { return "original" }
	defer func() { fantasy.NewID = previous }()
	for range 2 {
		WithCounterIDs(func() {
			require.Equal(t, "id-1", fantasy.NewID())
			require.Equal(t, "id-2", fantasy.NewID())
		})
		require.Equal(t, "original", fantasy.NewID())
	}
	require.Panics(t, func() {
		WithCounterIDs(func() { panic("test") })
	})
	require.Equal(t, "original", fantasy.NewID())
}

func TestFixtureCall(t *testing.T) {
	fixture := &Fixture{Request: json.RawMessage(`{"prompt":[{"role":"user","content":[{"type":"text","data":{"text":"fixture prompt"}}]}],"temperature":0.25,"tools":[{"type":"function","data":{"name":"lookup","description":"lookup","input_schema":{"type":"object","properties":{}}}}]}`)}
	call, err := fixture.Call()
	require.NoError(t, err)
	require.Len(t, call.Prompt, 1)
	require.Equal(t, 0.25, *call.Temperature)
	require.Len(t, call.Tools, 1)
	server := Serve(t, &Fixture{Events: SplitSSE(cannedSSE(false))})
	provider, err := openaicompat.New(openaicompat.WithBaseURL(server.URL()), openaicompat.WithAPIKey("test-key"))
	require.NoError(t, err)
	model, err := provider.LanguageModel(t.Context(), "test-model")
	require.NoError(t, err)
	stream, err := model.Stream(t.Context(), call)
	require.NoError(t, err)
	Collect(stream)
	requests := server.Requests()
	require.Len(t, requests, 1)
	require.Contains(t, string(requests[0]), "fixture prompt")
	require.Contains(t, string(requests[0]), `"temperature":0.25`)
	require.Contains(t, string(requests[0]), `"name":"lookup"`)
	for _, request := range []string{`{`, `null`, `{}`, `{"messages":[]}`, `{"prompt":[],"stream":true}`, `{"prompt":"wrong"}`} {
		fixture.Request = json.RawMessage(request)
		_, err := fixture.Call()
		require.ErrorContains(t, err, "request.json")
	}
}

func TestCitationReplayIDs(t *testing.T) {
	var runs [][]PartRecord
	for range 2 {
		WithCounterIDs(func() {
			fixture := &Fixture{Events: SplitSSE("data: {\"choices\":[{\"index\":0,\"delta\":{\"annotations\":[{\"type\":\"url_citation\",\"url_citation\":{\"url\":\"\",\"title\":\"test\"}}]},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")}
			server := Serve(t, fixture)
			provider, err := openaicompat.New(openaicompat.WithBaseURL(server.URL()), openaicompat.WithAPIKey("test-key"))
			require.NoError(t, err)
			model, err := provider.LanguageModel(t.Context(), "test-model")
			require.NoError(t, err)
			stream, err := model.Stream(t.Context(), fantasy.Call{})
			require.NoError(t, err)
			records := Collect(stream)
			require.Equal(t, "source", records[0].Type)
			require.Equal(t, "id-1", records[0].ID)
			runs = append(runs, records)
		})
	}
	require.Equal(t, runs[0], runs[1])
}

func TestDispatchRecordsUseCallOrder(t *testing.T) {
	var mu sync.Mutex
	dispatched := map[string]string{}
	tool := fantasy.NewAgentTool("lookup", "lookup", func(context.Context, struct{}, fantasy.ToolCall) (fantasy.ToolResponse, error) {
		return fantasy.NewTextResponse("ok"), nil
	})
	wrapped := recordingTool{AgentTool: tool, dispatched: dispatched, mu: &mu}
	for _, id := range []string{"second", "first"} {
		_, err := wrapped.Run(t.Context(), fantasy.ToolCall{ID: id, Name: "lookup", Input: "{}"})
		require.NoError(t, err)
	}
	content := fantasy.ResponseContent{
		fantasy.ToolCallContent{ToolCallID: "first", ToolName: "lookup"},
		fantasy.ToolCallContent{ToolCallID: "suppressed", ToolName: "ignored"},
		fantasy.ToolCallContent{ToolCallID: "second", ToolName: "lookup"},
	}
	require.Equal(t, []string{"lookup", "lookup"}, orderedDispatches(content, dispatched))
	dispatched["first"] = "first-tool"
	dispatched["second"] = "second-tool"
	require.Equal(t, []string{"first-tool", "second-tool"}, orderedDispatches(content, dispatched))
}

func TestParallelDispatchRecords(t *testing.T) {
	for range 30 {
		fixture, err := Load(filepath.Join("..", "providertests", "testdata", "shapes", "toolcall", "parallel_two"))
		require.NoError(t, err)
		server := Serve(t, fixture)
		provider, err := openaicompat.New(openaicompat.WithBaseURL(server.URL()), openaicompat.WithAPIKey("test-key"))
		require.NoError(t, err)
		model, err := provider.LanguageModel(t.Context(), "test-model")
		require.NoError(t, err)
		run := func(context.Context, map[string]any, fantasy.ToolCall) (fantasy.ToolResponse, error) {
			return fantasy.NewTextResponse("ok"), nil
		}
		record := RunAgentStep(t, model, fixture, fantasy.NewParallelAgentTool("grep", "grep", run), fantasy.NewParallelAgentTool("glob", "glob", run))
		require.Empty(t, record.Error)
		require.Equal(t, []string{"grep", "glob"}, record.ToolsDispatched)
	}
}
