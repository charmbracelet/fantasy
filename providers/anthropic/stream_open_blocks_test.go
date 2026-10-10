package anthropic

import (
	"context"
	"testing"

	"charm.land/fantasy"
	"github.com/anthropics/anthropic-sdk-go"
	"github.com/stretchr/testify/require"
)

// A tool call must be reported even when its block is never ended:
// max_tokens truncation omits content_block_stop, and the accumulator drops
// a block whose start it could not index.
func TestStream_ReportsToolCallsLeftOpenByTheProvider(t *testing.T) {
	t.Parallel()

	messageStart := anthropicSSEEvent("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}`)
	toolStart := anthropicSSEEvent("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"write","input":{}}}`)
	partialArgs := anthropicSSEEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"a.go\",\"content\":\"pack"}}`)
	toolStop := anthropicSSEEvent("content_block_stop", `{"type":"content_block_stop","index":0}`)
	maxTokens := anthropicSSEEvent("message_delta", `{"type":"message_delta","delta":{"stop_reason":"max_tokens","stop_sequence":null},"usage":{"output_tokens":64000}}`)
	toolUse := anthropicSSEEvent("message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":20}}`)
	messageStop := anthropicSSEEvent("message_stop", `{"type":"message_stop"}`)

	// A start event at an unexpected index is dropped by the accumulator.
	skewedStart := anthropicSSEEvent("content_block_start", `{"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"toolu_2","name":"write","input":{}}}`)
	skewedArgs := anthropicSSEEvent("content_block_delta", `{"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"b.go\"}"}}`)
	skewedStop := anthropicSSEEvent("content_block_stop", `{"type":"content_block_stop","index":3}`)

	tests := []struct {
		name         string
		chunks       []string
		wantInput    string
		wantFinish   fantasy.FinishReason
		wantToolName string
	}{
		{
			name:         "complete tool call",
			chunks:       []string{messageStart, toolStart, partialArgs, toolStop, toolUse, messageStop},
			wantInput:    `{"path":"a.go","content":"pack`,
			wantFinish:   fantasy.FinishReasonToolCalls,
			wantToolName: "write",
		},
		{
			name:         "tool call truncated at max_tokens",
			chunks:       []string{messageStart, toolStart, partialArgs, maxTokens, messageStop},
			wantInput:    `{"path":"a.go","content":"pack`,
			wantFinish:   fantasy.FinishReasonLength,
			wantToolName: "write",
		},
		{
			name:         "tool call the accumulator could not index",
			chunks:       []string{messageStart, skewedStart, skewedArgs, skewedStop, toolUse, messageStop},
			wantInput:    `{"path":"b.go"}`,
			wantFinish:   fantasy.FinishReasonToolCalls,
			wantToolName: "write",
		},
		{
			// No parameters means no deltas, so the arguments come from the
			// fallback.
			name:         "tool call with no arguments",
			chunks:       []string{messageStart, toolStart, toolStop, toolUse, messageStop},
			wantInput:    `{}`,
			wantFinish:   fantasy.FinishReasonToolCalls,
			wantToolName: "write",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			server, _ := newAnthropicStreamingServer(tt.chunks)
			defer server.Close()

			provider, err := New(WithAPIKey("test-api-key"), WithBaseURL(server.URL))
			require.NoError(t, err)
			model, err := provider.LanguageModel(context.Background(), "claude-sonnet-4-20250514")
			require.NoError(t, err)

			stream, err := model.Stream(context.Background(), fantasy.Call{Prompt: testPrompt()})
			require.NoError(t, err)

			var (
				toolCalls    []fantasy.StreamPart
				sawInputEnd  bool
				finishReason fantasy.FinishReason
				streamErr    error
			)
			for part := range stream {
				switch part.Type {
				case fantasy.StreamPartTypeToolCall:
					toolCalls = append(toolCalls, part)
				case fantasy.StreamPartTypeToolInputEnd:
					sawInputEnd = true
				case fantasy.StreamPartTypeFinish:
					finishReason = part.FinishReason
				case fantasy.StreamPartTypeError:
					streamErr = part.Error
				}
			}

			require.NoError(t, streamErr)
			require.True(t, sawInputEnd, "a started tool call must report the end of its input")
			require.Len(t, toolCalls, 1, "the tool call must be reported exactly once")
			require.Equal(t, tt.wantToolName, toolCalls[0].ToolCallName)
			require.Equal(t, tt.wantInput, toolCalls[0].ToolCallInput)
			require.Equal(t, tt.wantFinish, finishReason)
		})
	}
}

// Abandoning iteration while blocks are still open must not panic or emit
// anything further.
func TestStream_StopsCleanlyWithOpenToolCall(t *testing.T) {
	t.Parallel()

	chunks := []string{
		anthropicSSEEvent("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}`),
		anthropicSSEEvent("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"write","input":{}}}`),
		anthropicSSEEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":"}}`),
		anthropicSSEEvent("message_delta", `{"type":"message_delta","delta":{"stop_reason":"max_tokens","stop_sequence":null},"usage":{"output_tokens":9}}`),
		anthropicSSEEvent("message_stop", `{"type":"message_stop"}`),
	}

	server, _ := newAnthropicStreamingServer(chunks)
	defer server.Close()

	provider, err := New(WithAPIKey("test-api-key"), WithBaseURL(server.URL))
	require.NoError(t, err)
	model, err := provider.LanguageModel(context.Background(), "claude-sonnet-4-20250514")
	require.NoError(t, err)

	stream, err := model.Stream(context.Background(), fantasy.Call{Prompt: testPrompt()})
	require.NoError(t, err)

	var seen int
	for range stream {
		seen++
		break
	}
	require.Equal(t, 1, seen)
}

// The accumulator is indexed by position, and a block that could not be
// indexed leaves the positions of everything after it shifted. The fallback
// therefore has to confirm identity rather than trust position alone.
//
// This exercises arguments directly. Reaching the identity check through a
// stream is not currently possible: a dropped block only ever makes
// acc.Content shorter, so a drifted index fails the bounds check first. The
// check guards the contract rather than a reachable bug, and testing it at
// this level says so honestly instead of dressing it up as a stream case.
func TestOpenToolBlockArguments(t *testing.T) {
	t.Parallel()

	mine := anthropic.ContentBlockUnion{Type: "tool_use", ID: "toolu_1", Input: []byte(`{"path":"a.go"}`)}
	theirs := anthropic.ContentBlockUnion{Type: "tool_use", ID: "toolu_other", Input: []byte(`{"path":"other.go"}`)}

	tests := []struct {
		name    string
		deltas  string
		content []anthropic.ContentBlockUnion
		index   int64
		want    string
	}{
		{
			name:   "deltas win over the accumulator",
			deltas: `{"path":"from-delta.go"}`,
			// Even a matching block does not override what was streamed.
			content: []anthropic.ContentBlockUnion{mine},
			index:   0,
			want:    `{"path":"from-delta.go"}`,
		},
		{
			name:    "falls back to the block with this call's ID",
			content: []anthropic.ContentBlockUnion{mine},
			index:   0,
			want:    `{"path":"a.go"}`,
		},
		{
			name: "refuses a block belonging to another call",
			// Position says yes, identity says no. Handing over another
			// call's arguments would tell the model it asked for something
			// it never named, which is worse than reporting none.
			content: []anthropic.ContentBlockUnion{theirs},
			index:   0,
			want:    `{}`,
		},
		{
			name:    "index past the end",
			content: []anthropic.ContentBlockUnion{mine},
			index:   3,
			want:    `{}`,
		},
		{
			name:    "negative index",
			content: nil,
			index:   -1,
			want:    `{}`,
		},
		{
			name:    "empty accumulator",
			content: nil,
			index:   0,
			want:    `{}`,
		},
		{
			name:    "matching block with no input",
			content: []anthropic.ContentBlockUnion{{Type: "tool_use", ID: "toolu_1"}},
			index:   0,
			want:    `{}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			block := &openToolBlock{id: "toolu_1", name: "write"}
			if tt.deltas != "" {
				block.input.WriteString(tt.deltas)
			}
			got := block.arguments(anthropic.Message{Content: tt.content}, tt.index)
			require.Equal(t, tt.want, got)
		})
	}
}
