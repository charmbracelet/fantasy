package openaicompat

import (
	"context"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/replaytest"
	"github.com/stretchr/testify/require"
)

// cannedStopJSON answers the non-streaming second request of the manual
// round-trip test with a complete stop turn.
const cannedStopJSON = `{"id":"test-response","created":0,"model":"test-model","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"The date is 2026-08-28."},"finish_reason":"stop"}],"usage":{"prompt_tokens":200,"completion_tokens":10,"total_tokens":210}}`

// shapesDir locates the shared shape fixtures: the same fixtures the
// providertests golden harness replays, so every case here also has a
// reviewed parts.golden.json.
func shapeFixture(t *testing.T, shape, name string) *replaytest.Fixture {
	t.Helper()
	dir := filepath.Join("..", "..", "providertests", "testdata", "shapes", shape, name)
	fixture, err := replaytest.Load(dir)
	require.NoError(t, err)
	return fixture
}

func providerOn(t testing.TB, server *replaytest.Server) fantasy.LanguageModel {
	t.Helper()
	provider, err := New(WithBaseURL(server.URL()), WithAPIKey("x"))
	require.NoError(t, err)
	lm, err := provider.LanguageModel(context.Background(), "test-model")
	require.NoError(t, err)
	return lm
}

func streamParts(t *testing.T, lm fantasy.LanguageModel, prompt fantasy.Prompt) []fantasy.StreamPart {
	t.Helper()
	stream, err := lm.Stream(context.Background(), fantasy.Call{Prompt: prompt})
	require.NoError(t, err)
	var parts []fantasy.StreamPart
	for part := range stream {
		parts = append(parts, part)
	}
	return parts
}

func partTypes(parts []fantasy.StreamPart) []fantasy.StreamPartType {
	types := make([]fantasy.StreamPartType, 0, len(parts))
	for _, p := range parts {
		types = append(types, p.Type)
	}
	return types
}

// reasoningText concatenates the reasoning deltas; for reasoning deltas it
// also detects empty deltas.
func reasoningText(parts []fantasy.StreamPart) string {
	var sb strings.Builder
	for _, p := range parts {
		if p.Type == fantasy.StreamPartTypeReasoningDelta {
			sb.WriteString(p.Delta)
		}
	}
	return sb.String()
}

func countType(parts []fantasy.StreamPart, typ fantasy.StreamPartType) int {
	n := 0
	for _, p := range parts {
		if p.Type == typ {
			n++
		}
	}
	return n
}

func finishPart(parts []fantasy.StreamPart) *fantasy.StreamPart {
	for i, p := range parts {
		if p.Type == fantasy.StreamPartTypeFinish {
			return &parts[i]
		}
	}
	return nil
}

// TestReplay_DeepSeekThinkingToolCall drives the reasoning-then-toolcall
// fixture through a real openaicompat model and asserts the reasoning block
// survives end-to-end: exactly one ReasoningStart, the concatenated deltas,
// exactly one ReasoningEnd before Finish, and no empty ReasoningDeltas.
func TestReplay_DeepSeekThinkingToolCall(t *testing.T) {
	lm := providerOn(t, replaytest.Serve(t, shapeFixture(t, "reasoning_then_toolcall", "interleaved_with_nulls")))

	parts := streamParts(t, lm, fantasy.Prompt{
		{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "date?"}}},
	})

	require.Equal(t, 1, countType(parts, fantasy.StreamPartTypeReasoningStart), "types: %v", partTypes(parts))
	require.Equal(t, 1, countType(parts, fantasy.StreamPartTypeReasoningEnd), "types: %v", partTypes(parts))
	require.Equal(t, "The user wants the date. I'll call get_date.", reasoningText(parts))
	for _, p := range parts {
		if p.Type == fantasy.StreamPartTypeReasoningDelta {
			require.NotEmpty(t, p.Delta, "empty ReasoningDelta emitted")
		}
	}
	// ReasoningEnd must come before Finish.
	endIdx, finishIdx := -1, -1
	for i, p := range parts {
		if p.Type == fantasy.StreamPartTypeReasoningEnd {
			endIdx = i
		}
		if p.Type == fantasy.StreamPartTypeFinish {
			finishIdx = i
		}
	}
	require.Greater(t, endIdx, -1)
	require.Greater(t, finishIdx, endIdx, "ReasoningEnd must precede Finish: %v", partTypes(parts))
	require.Equal(t, 1, countType(parts, fantasy.StreamPartTypeToolCall), "types: %v", partTypes(parts))
}

// TestReplay_DeepSeekReasoningRoundTripsToNextRequest is the regression test
// for crush#2696 / CHARM-2020: the reasoning the model produced must be sent
// back, byte-equal, on the assistant message carrying tool_calls in the next
// request. We drive two steps manually (no agent) to keep the harness small.
func TestReplay_DeepSeekReasoningRoundTripsToNextRequest(t *testing.T) {
	fixture := shapeFixture(t, "reasoning_then_toolcall", "interleaved_with_nulls")
	server := replaytest.Serve(t, fixture, replaytest.WithSubsequentJSON([]byte(cannedStopJSON)))
	lm := providerOn(t, server)

	// Step 1: collect reasoning + tool call from the stream.
	parts := streamParts(t, lm, fantasy.Prompt{
		{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "date?"}}},
	})
	reasoning := reasoningText(parts)
	require.NotEmpty(t, reasoning)
	var toolCall *fantasy.StreamPart
	for i, p := range parts {
		if p.Type == fantasy.StreamPartTypeToolCall {
			toolCall = &parts[i]
		}
	}
	require.NotNil(t, toolCall, "types: %v", partTypes(parts))

	// Step 2: replay the assistant message the way a faithful client would
	// and assert on the request body the provider sends upstream. Non-stream
	// so the server answers with JSON.
	resp, err := lm.Generate(context.Background(), fantasy.Call{Prompt: fantasy.Prompt{
		{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "date?"}}},
		{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{
			fantasy.ReasoningPart{Text: reasoning},
			fantasy.TextPart{Text: "Let me check."},
			fantasy.ToolCallPart{ToolCallID: toolCall.ID, ToolName: toolCall.ToolCallName, Input: toolCall.ToolCallInput},
		}},
		{Role: fantasy.MessageRoleTool, Content: []fantasy.MessagePart{
			fantasy.ToolResultPart{ToolCallID: toolCall.ID, Output: fantasy.ToolResultOutputContentText{Text: "2026-08-28"}},
		}},
	}})
	require.NoError(t, err)
	require.NotNil(t, resp)

	requests := server.Requests()
	require.Len(t, requests, 2)
	assertReasoningRoundTrip(t, requests[1], reasoning)
}

// TestReplay_AgentReasoningRoundTrip is the regression test for crush#2696
// and the client-side half of CHARM-2020: drive the reasoning-then-toolcall
// fixture through a real agent (the loop Crush runs), let it dispatch the
// tool, and assert on the SECOND request body — the assistant message
// carrying tool_calls must carry reasoning_content byte-equal to the
// concatenated reasoning deltas.
func TestReplay_AgentReasoningRoundTrip(t *testing.T) {
	fixture := shapeFixture(t, "reasoning_then_toolcall", "interleaved_with_nulls")
	lm := providerOn(t, replaytest.Serve(t, fixture))

	getDate := fantasy.NewAgentTool(
		"get_date",
		"Get the current date.",
		func(_ context.Context, _ struct{}, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			return fantasy.NewTextResponse("2026-08-28"), nil
		},
	)

	record := replaytest.RunAgentStep(t, lm, fixture, getDate)
	require.Equal(t, []string{"get_date"}, record.ToolsDispatched)
	require.Equal(t, fantasy.FinishReasonToolCalls, fantasy.FinishReason(record.FinishReason))
	require.NotEmpty(t, record.NextRequest)
	assertReasoningRoundTrip(t, []byte(record.NextRequest), "The user wants the date. I'll call get_date.")
	replaytest.AssertGolden(t, filepath.Join(shapeFixture(t, "reasoning_then_toolcall", "interleaved_with_nulls").Dir, "step.golden.json"), record)
}

// assertReasoningRoundTrip asserts that the request body carries the
// assistant message with reasoning_content byte-equal to the streamed
// reasoning and the complete tool call.
func assertReasoningRoundTrip(t *testing.T, body []byte, reasoning string) {
	t.Helper()
	var request struct {
		Messages []struct {
			Role             string  `json:"role"`
			ReasoningContent *string `json:"reasoning_content"`
			ToolCalls        []struct {
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(body, &request))
	var assistant *struct {
		Role             string  `json:"role"`
		ReasoningContent *string `json:"reasoning_content"`
		ToolCalls        []struct {
			ID       string `json:"id"`
			Function struct {
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			} `json:"function"`
		} `json:"tool_calls"`
	}
	for i := range request.Messages {
		if request.Messages[i].Role == "assistant" {
			assistant = &request.Messages[i]
		}
	}
	require.NotNil(t, assistant, "no assistant message in request: %s", body)
	require.NotNil(t, assistant.ReasoningContent,
		"reasoning_content missing on the replayed assistant message — DeepSeek 400s this and other hosts loop: %s", body)
	require.Equal(t, reasoning, *assistant.ReasoningContent, "reasoning_content must round-trip byte-for-byte")
	require.Len(t, assistant.ToolCalls, 1)
	require.Equal(t, "call_test_1", assistant.ToolCalls[0].ID)
	require.Equal(t, "get_date", assistant.ToolCalls[0].Function.Name)
	require.Equal(t, "{}", assistant.ToolCalls[0].Function.Arguments)
}

// TestReplay_DeepSeekReasoningOnlyTruncated: a reasoning-only response
// truncated by length. ReasoningEnd must still be emitted (on the finish
// chunk) and the reasoning kept.
func TestReplay_DeepSeekReasoningOnlyTruncated(t *testing.T) {
	lm := providerOn(t, replaytest.Serve(t, shapeFixture(t, "reasoning_only", "finish_length")))

	parts := streamParts(t, lm, fantasy.Prompt{
		{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "hi"}}},
	})
	require.Equal(t, "Thinking about it at length", reasoningText(parts))
	require.Equal(t, 1, countType(parts, fantasy.StreamPartTypeReasoningEnd), "types: %v", partTypes(parts))
	finish := finishPart(parts)
	require.NotNil(t, finish)
	require.Equal(t, fantasy.FinishReasonLength, finish.FinishReason)
}

// TestReplay_DeepSeekBatchedBoundaryChunk: a batching host puts the
// reasoning tail and the whole tool call in one delta. The reasoning must be
// kept and the tool call intact.
func TestReplay_DeepSeekBatchedBoundaryChunk(t *testing.T) {
	lm := providerOn(t, replaytest.Serve(t, shapeFixture(t, "reasoning_tail_batched_with_toolcall", "basic")))

	parts := streamParts(t, lm, fantasy.Prompt{
		{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "date?"}}},
	})
	require.Equal(t, "Need the date. Calling.", reasoningText(parts))
	require.Equal(t, 1, countType(parts, fantasy.StreamPartTypeToolCall), "types: %v", partTypes(parts))
}

// TestReplay_KimiNullFinishDoesNotReopen: reasoning chunks carry
// reasoning_content, content chunks omit the key, the finish chunk has
// "reasoning_content": null — which must not reopen a reasoning block.
func TestReplay_KimiNullFinishDoesNotReopen(t *testing.T) {
	lm := providerOn(t, replaytest.Serve(t, shapeFixture(t, "reasoning_then_text", "null_finish")))

	parts := streamParts(t, lm, fantasy.Prompt{
		{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "hi"}}},
	})
	require.Equal(t, "first second", reasoningText(parts))
	require.Equal(t, 1, countType(parts, fantasy.StreamPartTypeReasoningStart), "types: %v", partTypes(parts))
	require.Equal(t, 1, countType(parts, fantasy.StreamPartTypeReasoningEnd), "types: %v", partTypes(parts))
	for _, p := range parts {
		if p.Type == fantasy.StreamPartTypeReasoningDelta {
			require.NotEmpty(t, p.Delta, "empty ReasoningDelta emitted (phantom block reopened)")
		}
	}
}

// TestReplay_KimiPresentButEmpty: a chunk with "reasoning_content": ""
// before any content starts an (empty) reasoning block that closes on the
// first tool-call chunk and replays as "reasoning_content": "".
func TestReplay_KimiPresentButEmpty(t *testing.T) {
	lm := providerOn(t, replaytest.Serve(t, shapeFixture(t, "reasoning_empty_then_toolcall", "basic")))

	parts := streamParts(t, lm, fantasy.Prompt{
		{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "hi"}}},
	})
	require.Equal(t, 1, countType(parts, fantasy.StreamPartTypeReasoningStart), "types: %v", partTypes(parts))
	require.Equal(t, 1, countType(parts, fantasy.StreamPartTypeReasoningEnd), "types: %v", partTypes(parts))
	for _, p := range parts {
		if p.Type == fantasy.StreamPartTypeReasoningDelta {
			require.NotEmpty(t, p.Delta, "empty ReasoningDelta emitted")
		}
	}
}

// TestReplay_CutStreamMidArguments_NotDispatched: a stream cut mid-arguments
// with no finish_reason and no [DONE] must not surface a complete ToolCall:
// truncated arguments must never be dispatched.
func TestReplay_CutStreamMidArguments_NotDispatched(t *testing.T) {
	lm := providerOn(t, replaytest.Serve(t, shapeFixture(t, "toolcall", "connection_closed")))

	parts := streamParts(t, lm, fantasy.Prompt{
		{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "write main.go"}}},
	})
	require.Equal(t, 0, countType(parts, fantasy.StreamPartTypeToolCall),
		"truncated tool call must not be dispatched; types: %v", partTypes(parts))
	var sawIncomplete bool
	for _, p := range parts {
		if p.Type == fantasy.StreamPartTypeError {
			sawIncomplete = true
		}
	}
	require.True(t, sawIncomplete, "expected an error part (IncompleteStreamError), types: %v", partTypes(parts))
}

// TestReplay_AgentReasoningOnlyTruncated: a reasoning-only response truncated
// by length at agent level. The agent must surface the reasoning content in
// the step even though the provider can only close the block on the finish
// chunk.
func TestReplay_AgentReasoningOnlyTruncated(t *testing.T) {
	fixture := shapeFixture(t, "reasoning_only", "finish_length")
	lm := providerOn(t, replaytest.Serve(t, fixture))

	record := replaytest.RunAgentStep(t, lm, fixture)
	require.Equal(t, fantasy.FinishReasonLength, fantasy.FinishReason(record.FinishReason))
	require.Len(t, record.Content, 1)
	require.Equal(t, "reasoning", record.Content[0].Type)
	require.Equal(t, "Thinking about it at length", record.Content[0].Delta)
	replaytest.AssertGolden(t, filepath.Join(shapeFixture(t, "reasoning_only", "finish_length").Dir, "step.golden.json"), record)
}

// TestReplay_MultiChoiceReasoningNotDuplicated: a chunk carrying two choices
// must not duplicate reasoning events: StreamExtraFunc is invoked per choice
// by the openai language model but iterates all choices itself, so without
// care each choice's reasoning is emitted once per choice.
func TestReplay_MultiChoiceReasoningNotDuplicated(t *testing.T) {
	lm := providerOn(t, replaytest.Serve(t, shapeFixture(t, "multi_choice", "reasoning_two_choices")))

	parts := streamParts(t, lm, fantasy.Prompt{
		{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "hi"}}},
	})
	deltasByID := map[string]string{}
	startsByID := map[string]int{}
	for _, p := range parts {
		switch p.Type {
		case fantasy.StreamPartTypeReasoningStart:
			startsByID[p.ID]++
		case fantasy.StreamPartTypeReasoningDelta:
			deltasByID[p.ID] += p.Delta
		}
	}
	require.Equal(t, map[string]int{"0": 1, "1": 1}, startsByID, "types: %v", partTypes(parts))
	require.Equal(t, map[string]string{"0": "a", "1": "b"}, deltasByID, "types: %v", partTypes(parts))
}

// TestReplay_DoneWithoutFinishReason_TruncatedArgsSuppressed: a stream that
// ends cleanly ([DONE]) but never sent a finish_reason must not dispatch tool
// calls whose arguments are incomplete: "tool calls were seen" is not proof
// of a complete turn.
func TestReplay_DoneWithoutFinishReason_TruncatedArgsSuppressed(t *testing.T) {
	lm := providerOn(t, replaytest.Serve(t, shapeFixture(t, "toolcall", "finish_missing_truncated")))

	parts := streamParts(t, lm, fantasy.Prompt{
		{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "x"}}},
	})
	require.Equal(t, 0, countType(parts, fantasy.StreamPartTypeToolCall),
		"truncated tool call must not be dispatched; types: %v", partTypes(parts))
	var sawError bool
	for _, p := range parts {
		if p.Type == fantasy.StreamPartTypeError {
			sawError = true
		}
	}
	require.True(t, sawError, "expected an IncompleteStreamError part, types: %v", partTypes(parts))
	require.Equal(t, 0, countType(parts, fantasy.StreamPartTypeFinish), "types: %v", partTypes(parts))
}

// TestReplay_DoneWithoutFinishReason_ValidArgsKeptWithWarning: valid
// arguments keep today's inferred tool-call turn, with a warning.
func TestReplay_DoneWithoutFinishReason_ValidArgsKeptWithWarning(t *testing.T) {
	lm := providerOn(t, replaytest.Serve(t, shapeFixture(t, "toolcall", "finish_missing_valid")))

	parts := streamParts(t, lm, fantasy.Prompt{
		{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "x"}}},
	})
	require.Equal(t, 1, countType(parts, fantasy.StreamPartTypeToolCall), "types: %v", partTypes(parts))
	var sawWarning bool
	for _, p := range parts {
		if p.Type == fantasy.StreamPartTypeWarnings {
			sawWarning = true
		}
	}
	require.True(t, sawWarning, "expected a CallWarning about the missing finish_reason, types: %v", partTypes(parts))
	finish := finishPart(parts)
	require.NotNil(t, finish)
	require.Equal(t, fantasy.FinishReasonToolCalls, finish.FinishReason)
}

// TestReplay_InsufficientSystemResource_SuppressesToolCalls:
// insufficient_system_resource is a provider-side failure, not a completed
// turn: it must map to an error finish and suppress any open tool calls.
func TestReplay_InsufficientSystemResource_SuppressesToolCalls(t *testing.T) {
	lm := providerOn(t, replaytest.Serve(t, shapeFixture(t, "toolcall", "finish_insufficient_resource")))

	parts := streamParts(t, lm, fantasy.Prompt{
		{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "x"}}},
	})
	require.Equal(t, 0, countType(parts, fantasy.StreamPartTypeToolCall),
		"tool calls from a failed upstream must not be dispatched; types: %v", partTypes(parts))
	finish := finishPart(parts)
	require.NotNil(t, finish, "types: %v", partTypes(parts))
	require.NotEqual(t, fantasy.FinishReasonToolCalls, finish.FinishReason)
}

// TestReplay_ReorderedChoices: choices may not arrive in slice order;
// reasoning state and part IDs must follow choice.Index, not the slice
// position within a chunk.
func TestReplay_ReorderedChoices(t *testing.T) {
	lm := providerOn(t, replaytest.Serve(t, shapeFixture(t, "multi_choice", "reordered_choices")))

	parts := streamParts(t, lm, fantasy.Prompt{
		{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "hi"}}},
	})
	deltasByID := map[string]string{}
	endsByID := map[string]int{}
	startsByID := map[string]int{}
	for _, p := range parts {
		switch p.Type {
		case fantasy.StreamPartTypeReasoningStart:
			startsByID[p.ID]++
		case fantasy.StreamPartTypeReasoningDelta:
			deltasByID[p.ID] += p.Delta
		case fantasy.StreamPartTypeReasoningEnd:
			endsByID[p.ID]++
		}
	}
	require.Equal(t, map[string]int{"0": 1, "1": 1}, startsByID, "types: %v", partTypes(parts))
	require.Equal(t, map[string]string{"0": "A1A2", "1": "B1B2"}, deltasByID, "types: %v", partTypes(parts))
	require.Equal(t, map[string]int{"0": 1, "1": 1}, endsByID, "types: %v", partTypes(parts))
}

// TestReplay_ContentFilter_SuppressesToolCalls: a content_filter finish can
// cut a tool call mid-arguments; like length, it must not be rewritten to
// tool_calls and its calls must not dispatch.
func TestReplay_ContentFilter_SuppressesToolCalls(t *testing.T) {
	lm := providerOn(t, replaytest.Serve(t, shapeFixture(t, "toolcall", "finish_content_filter")))

	parts := streamParts(t, lm, fantasy.Prompt{
		{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "x"}}},
	})
	require.Equal(t, 0, countType(parts, fantasy.StreamPartTypeToolCall),
		"tool calls from a filtered response must not be dispatched; types: %v", partTypes(parts))
	finish := finishPart(parts)
	require.NotNil(t, finish, "types: %v", partTypes(parts))
	require.Equal(t, fantasy.FinishReasonContentFilter, finish.FinishReason)
}

// TestReplay_DoneWithoutFinishReason_NoArgsSuppressed: a stream with no
// finish_reason whose only tool call is a bare declaration (no arguments at
// all) was cut before any argument arrived. Inferring a complete "{}" call
// invents arguments the model never sent.
func TestReplay_DoneWithoutFinishReason_NoArgsSuppressed(t *testing.T) {
	lm := providerOn(t, replaytest.Serve(t, shapeFixture(t, "toolcall", "finish_missing_no_args")))

	parts := streamParts(t, lm, fantasy.Prompt{
		{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "x"}}},
	})
	require.Equal(t, 0, countType(parts, fantasy.StreamPartTypeToolCall),
		"a bare tool declaration is not a complete call; types: %v", partTypes(parts))
	var sawError bool
	for _, p := range parts {
		if p.Type == fantasy.StreamPartTypeError {
			sawError = true
		}
	}
	require.True(t, sawError, "expected IncompleteStreamError, types: %v", partTypes(parts))
}

// TestReplay_InterleavedParallelToolCalls: interleaved parallel tool calls:
// deltas for index 0 continue after index 1 has started. Arguments for both
// must accumulate fully — closing index 0 when 1 first appears would drop
// its trailing deltas.
func TestReplay_InterleavedParallelToolCalls(t *testing.T) {
	lm := providerOn(t, replaytest.Serve(t, shapeFixture(t, "toolcall", "parallel_two")))

	parts := streamParts(t, lm, fantasy.Prompt{
		{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "x"}}},
	})
	calls := map[string]string{}
	for _, p := range parts {
		if p.Type == fantasy.StreamPartTypeToolCall {
			calls[p.ToolCallName] = p.ToolCallInput
		}
	}
	require.Equal(t, map[string]string{
		"grep": `{"pattern":"foo"}`,
		"glob": `{"path":"x"}`,
	}, calls)
}

// TestReplay_DoneWithoutFinishReason_NoFabricatedEnd: on a cut stream,
// consumers must not see a ToolInputEnd that presents fabricated "{}" input
// for a call that never sent arguments.
func TestReplay_DoneWithoutFinishReason_NoFabricatedEnd(t *testing.T) {
	lm := providerOn(t, replaytest.Serve(t, shapeFixture(t, "toolcall", "finish_missing_no_args")))

	parts := streamParts(t, lm, fantasy.Prompt{
		{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "x"}}},
	})
	for _, p := range parts {
		require.NotEqual(t, fantasy.StreamPartTypeToolInputEnd, p.Type,
			"no ToolInputEnd may be emitted on a cut stream; types: %v", partTypes(parts))
		require.NotEqual(t, fantasy.StreamPartTypeToolCall, p.Type, "types: %v", partTypes(parts))
	}
	var sawError bool
	for _, p := range parts {
		if p.Type == fantasy.StreamPartTypeError {
			sawError = true
		}
	}
	require.True(t, sawError, "types: %v", partTypes(parts))
}
