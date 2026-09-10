package replaytest

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestSplitSSE(t *testing.T) {
	events := SplitSSE("data: {\"a\":1}\n\ndata: [DONE]\n")
	require.Equal(t, []string{`data: {"a":1}`, "data: [DONE]"}, events)

	events = SplitSSE("event: ping\ndata: {\"type\":\"ping\"}\n\n")
	require.Equal(t, []string{"event: ping\ndata: {\"type\":\"ping\"}"}, events)

	events = SplitSSE("data: {\"a\":1}\r\n\r\ndata: [DONE]\r\n")
	require.Equal(t, []string{`data: {"a":1}`, "data: [DONE]"}, events)

	events = SplitSSE("data: {\"a\":1}\r\n\r\n<connection closed>\r\n")
	require.Equal(t, []string{`data: {"a":1}`, connectionClosedMarker}, events)
}

func TestLoad(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(dir, "request.json"), []byte(`{}`), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "response.sse"), []byte("data: {\"x\":1}\n\ndata: [DONE]\n"), 0o644))

	fixture, err := Load(dir)
	require.NoError(t, err)
	require.Equal(t, dir, fixture.Dir)
	require.Equal(t, []string{`data: {"x":1}`, "data: [DONE]"}, fixture.Events)
	require.Empty(t, fixture.Body)

	require.NoError(t, os.WriteFile(filepath.Join(dir, "response.json"), []byte(`{}`), 0o644))
	_, err = Load(dir)
	require.Error(t, err, "a fixture cannot have both response files")

	require.NoError(t, os.Remove(filepath.Join(dir, "response.sse")))
	fixture, err = Load(dir)
	require.NoError(t, err)
	require.Equal(t, `{}`, string(fixture.Body))
	require.Empty(t, fixture.Events)
}

func TestServerRecordsRequests(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(dir, "request.json"), []byte(`{"stream":true}`), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "response.sse"), []byte("data: {\"a\":1}\n\ndata: [DONE]\n"), 0o644))
	fixture, err := Load(dir)
	require.NoError(t, err)

	server := Serve(t, fixture)
	require.Len(t, server.Requests(), 0)
	res, err := server.server.Client().Post(server.URL(), "application/json", strings.NewReader(`{"stream":true}`))
	require.NoError(t, err)
	require.NoError(t, res.Body.Close())

	requests := server.Requests()
	require.Len(t, requests, 1)
	require.Equal(t, `{"stream":true}`, string(requests[0]))
}

type fakeStreamModel struct{}

func (fakeStreamModel) Generate(context.Context, fantasy.Call) (*fantasy.Response, error) {
	return nil, nil
}

func (fakeStreamModel) Stream(context.Context, fantasy.Call) (fantasy.StreamResponse, error) {
	return func(yield func(fantasy.StreamPart) bool) {
		if !yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeTextStart, ID: "t1"}) {
			return
		}
		if !yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeTextDelta, ID: "t1", Delta: "hello"}) {
			return
		}
		if !yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeTextEnd, ID: "t1"}) {
			return
		}
		yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeFinish,
			FinishReason: fantasy.FinishReasonStop,
			Usage:        fantasy.Usage{InputTokens: 1, OutputTokens: 2, TotalTokens: 3},
		})
	}, nil
}

func (fakeStreamModel) GenerateObject(context.Context, fantasy.ObjectCall) (*fantasy.ObjectResponse, error) {
	return nil, nil
}

func (fakeStreamModel) StreamObject(context.Context, fantasy.ObjectCall) (fantasy.ObjectStreamResponse, error) {
	return nil, nil
}

func (fakeStreamModel) Provider() string { return "fake" }

func (fakeStreamModel) Model() string { return "test-model" }

func TestCollect(t *testing.T) {
	stream, err := fakeStreamModel{}.Stream(context.Background(), fantasy.Call{})
	require.NoError(t, err)
	records := Collect(stream)
	require.Equal(t, []PartRecord{
		{Type: "text_start", ID: "t1"},
		{Type: "text_delta", ID: "t1", Delta: "hello"},
		{Type: "text_end", ID: "t1"},
		{
			Type:   "finish",
			Reason: "stop",
			Usage:  &UsageRecord{InputTokens: 1, OutputTokens: 2, TotalTokens: 3},
		},
	}, records)
}

func TestRunAgentStep(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(dir, "request.json"), []byte(`{}`), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "response.sse"), []byte("data: {\"x\":1}\n\ndata: [DONE]\n"), 0o644))
	fixture, err := Load(dir)
	require.NoError(t, err)

	record := RunAgentStep(t, fakeStreamModel{}, fixture)
	require.Empty(t, record.Error)
	require.Equal(t, []PartRecord{
		{Type: "text", Delta: "hello"},
	}, record.Content)
	require.Equal(t, "stop", record.FinishReason)
	require.Empty(t, record.ToolsDispatched)
	require.Nil(t, record.NextRequest)
}

func TestUnifiedDiff(t *testing.T) {
	before := "a\nb\nc\n"
	after := "a\nb\nx\n"
	diff := UnifiedDiff(before, after)
	require.Contains(t, diff, "@@ -1,3 +1,3 @@\n")
	require.Contains(t, diff, "-c\n")
	require.Contains(t, diff, "+x\n")
	require.Contains(t, diff, " b\n")

	data, err := json.Marshal([]PartRecord{{Type: "finish", Reason: "stop"}})
	require.NoError(t, err)
	require.JSONEq(t, `[{"type":"finish","reason":"stop"}]`, string(data))
}

func TestAssertGoldenToleratesCRLF(t *testing.T) {
	path := filepath.Join(t.TempDir(), "golden.json")
	require.NoError(t, os.WriteFile(path, []byte("[\r\n  {\r\n    \"type\": \"finish\",\r\n    \"reason\": \"stop\"\r\n  }\r\n]\r\n"), 0o600))
	AssertGolden(t, path, []PartRecord{{Type: "finish", Reason: "stop"}})
}
