package openai

import (
	"context"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestResponsesStream_ReasoningSummaryPartsStartWithBlankLine(t *testing.T) {
	t.Parallel()

	partAdded := func(itemID, index string) string {
		return responsesSSEEvent("response.reasoning_summary_part.added", `{"type":"response.reasoning_summary_part.added","item_id":"`+itemID+`","output_index":0,"summary_index":`+index+`,"part":{"type":"summary_text","text":""}}`)
	}
	textDelta := func(itemID, index, delta string) string {
		return responsesSSEEvent("response.reasoning_summary_text.delta", `{"type":"response.reasoning_summary_text.delta","item_id":"`+itemID+`","output_index":0,"summary_index":`+index+`,"delta":"`+delta+`"}`)
	}
	sms := newStreamingMockServer()
	defer sms.close()
	sms.chunks = []string{
		responsesSSEEvent("response.output_item.added", `{"type":"response.output_item.added","output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[]}}`),
		partAdded("rs_1", "0"),
		textDelta("rs_1", "0", `**A**\n\na`),
		partAdded("rs_1", "1"),
		textDelta("rs_1", "1", `**B**\n\nb`),
		responsesSSEEvent("response.output_item.done", `{"type":"response.output_item.done","output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"**A**\n\na"},{"type":"summary_text","text":"**B**\n\nb"}]}}`),
		responsesSSEEvent("response.output_item.added", `{"type":"response.output_item.added","output_index":1,"item":{"id":"rs_2","type":"reasoning","summary":[]}}`),
		partAdded("rs_2", "0"),
		textDelta("rs_2", "0", `**C**\n\nc`),
		responsesSSEEvent("response.output_item.done", `{"type":"response.output_item.done","output_index":1,"item":{"id":"rs_2","type":"reasoning","summary":[{"type":"summary_text","text":"**C**\n\nc"}]}}`),
		responsesSSEEvent("response.completed", `{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}`),
	}

	stream, err := newResponsesProvider(t, sms.server.URL).Stream(context.Background(), fantasy.Call{Prompt: testPrompt})
	require.NoError(t, err)
	parts, err := collectStreamParts(stream)
	require.NoError(t, err)

	text := map[string]string{}
	for _, part := range parts {
		if part.Type == fantasy.StreamPartTypeReasoningDelta {
			text[part.ID] += part.Delta
		}
	}
	require.Equal(t, map[string]string{
		"rs_1": "\n\n**A**\n\na\n\n**B**\n\nb",
		"rs_2": "\n\n**C**\n\nc",
	}, text)
}

func TestResponsesGenerate_ReasoningSummaryPartsJoinWithBlankLine(t *testing.T) {
	t.Parallel()

	server := newMockServer()
	defer server.close()
	server.response = map[string]any{
		"id":     "resp_1",
		"object": "response",
		"model":  "gpt-4.1",
		"status": "completed",
		"output": []any{
			map[string]any{"type": "reasoning", "id": "rs_1", "summary": []any{
				map[string]any{"type": "summary_text", "text": "**A**\n\na"},
				map[string]any{"type": "summary_text", "text": "**B**\n\nb"},
			}},
		},
		"usage": map[string]any{"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
	}

	resp, err := newResponsesProvider(t, server.server.URL).Generate(context.Background(), fantasy.Call{Prompt: testPrompt})
	require.NoError(t, err)

	reasoning := resp.Content.Reasoning()
	require.Len(t, reasoning, 1)
	require.Equal(t, "**A**\n\na\n\n**B**\n\nb", reasoning[0].Text)
}
