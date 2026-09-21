package replaytest

import (
	"context"
	"encoding/json"
	"strconv"
	"sync"
	"testing"

	"charm.land/fantasy"
)

// StepRecord captures one agent step driven by RunAgentStep: the step
// content as PartRecords, the finish reason, which tools were dispatched,
// and the body of the next request the agent sent (from Server.Requests,
// index 1), or empty when the step made no further request.
//
// next_request is the raw request body, so step goldens keep byte-for-byte
// assertions on what the agent sends upstream.
type StepRecord struct {
	Content         []PartRecord    `json:"content"`
	FinishReason    string          `json:"finish_reason"`
	ToolsDispatched []string        `json:"tools_dispatched,omitempty"`
	NextRequest     json.RawMessage `json:"next_request,omitempty"`
	Error           string          `json:"error,omitempty"`
}

// RunAgentStep runs Agent.Stream for one step against the fixture with the
// given stub tools, which are wrapped so dispatches are recorded. Retries
// are disabled so the step makes at most the requests the fixture implies,
// and generated IDs are replaced by a deterministic counter for the duration
// of the call.
//
// Serve the fixture first and build the model against that server; the step
// then reads the next request body from that server. When the fixture has no
// server bound yet, RunAgentStep serves it itself — useful when the model
// does not need the fixture's response, such as error-path fixtures.
//
// The agent prompt is fixed ("hi") so step goldens are stable.
func RunAgentStep(t testing.TB, model fantasy.LanguageModel, f *Fixture, tools ...fantasy.AgentTool) StepRecord {
	t.Helper()

	server := f.server
	if server == nil {
		server = Serve(t, f)
	}
	before := len(server.Requests())

	var (
		dispatched = make(map[string]string)
		mu         sync.Mutex
		stepNumber int
		callOrder  fantasy.ResponseContent
	)
	wrapped := make([]fantasy.AgentTool, 0, len(tools))
	for _, tool := range tools {
		wrapped = append(wrapped, &recordingTool{AgentTool: tool, dispatched: dispatched, mu: &mu, stepNumber: &stepNumber})
	}

	agent := fantasy.NewAgent(model, fantasy.WithMaxRetries(0), fantasy.WithTools(wrapped...))
	var result *fantasy.AgentResult
	var err error
	WithCounterIDs(func() {
		result, err = agent.Stream(t.Context(), fantasy.AgentStreamCall{
			Prompt: "hi",
			OnStepStart: func(number int) error {
				mu.Lock()
				stepNumber = number
				mu.Unlock()
				return nil
			},
			OnToolCall: func(call fantasy.ToolCallContent) error {
				mu.Lock()
				if stepNumber == 0 {
					callOrder = append(callOrder, call)
				}
				mu.Unlock()
				return nil
			},
		})
	})

	record := StepRecord{Content: []PartRecord{}}
	if err != nil {
		record.Error = localAddrPattern.ReplaceAllString(err.Error(), "127.0.0.1:0")
	} else if len(result.Steps) > 0 {
		record.Content = recordsFromContent(result.Steps[0].Content)
		record.FinishReason = string(result.Steps[0].FinishReason)
	}

	mu.Lock()
	record.ToolsDispatched = orderedDispatches(callOrder, dispatched)
	mu.Unlock()

	requests := server.Requests()
	if len(requests) > before+1 {
		record.NextRequest = requests[before+1]
	}
	return record
}

// recordingTool wraps an AgentTool and records every dispatch by tool name.
type recordingTool struct {
	fantasy.AgentTool
	dispatched map[string]string
	mu         *sync.Mutex
	stepNumber *int
}

func (r *recordingTool) Run(ctx context.Context, call fantasy.ToolCall) (fantasy.ToolResponse, error) {
	r.mu.Lock()
	if r.stepNumber == nil || *r.stepNumber == 0 {
		r.dispatched[call.ID] = call.Name
	}
	r.mu.Unlock()
	return r.AgentTool.Run(ctx, call)
}

func orderedDispatches(content fantasy.ResponseContent, dispatched map[string]string) []string {
	var names []string
	for _, call := range content.ToolCalls() {
		if name, ok := dispatched[call.ToolCallID]; ok {
			names = append(names, name)
		}
	}
	return names
}

// WithCounterIDs makes generated IDs reproducible within run and restores the
// previous generator on return or panic. The scope must not overlap parallel
// tests or unrelated provider operations because fantasy.NewID is global.
func WithCounterIDs(run func()) {
	previous := fantasy.NewID
	var (
		mu      sync.Mutex
		counter int
	)
	fantasy.NewID = func() string {
		mu.Lock()
		defer mu.Unlock()
		counter++
		return "id-" + strconv.Itoa(counter)
	}
	defer func() {
		fantasy.NewID = previous
	}()
	run()
}
