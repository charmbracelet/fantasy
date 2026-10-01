package openai

import (
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestResponsesEncryptedReasoningReplay(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name      string
		repeat    bool
		tool      bool
		encrypted bool
	}{
		{name: "reasoning only", encrypted: true},
		{name: "before tool call", encrypted: true, tool: true},
		{name: "duplicate fragments", encrypted: true, repeat: true},
		{name: "no encrypted content", tool: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			metadata := &ResponsesReasoningMetadata{ItemID: "rs_1", Summary: []string{"first", "second"}}
			if tc.encrypted {
				metadata.EncryptedContent = new("encrypted-final")
			}
			part := fantasy.ReasoningPart{Text: "first\nsecond", ProviderOptions: fantasy.ProviderOptions{Name: metadata}}
			msg := fantasy.Message{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{part}}
			if tc.repeat {
				msg.Content = append(msg.Content, part)
			}
			if tc.tool {
				msg.Content = append(msg.Content, fantasy.ToolCallPart{ToolCallID: "call_1", ToolName: "echo", Input: `{}`})
			}
			input, warnings := toResponsesPrompt(fantasy.Prompt{msg}, "system", false)
			require.Empty(t, warnings)
			want := 0
			if tc.encrypted {
				want++
			}
			if tc.tool {
				want++
			}
			require.Len(t, input, want)
			if tc.encrypted {
				reasoning := input[0].OfReasoning
				require.NotNil(t, reasoning)
				require.Equal(t, "rs_1", reasoning.ID)
				require.Equal(t, "encrypted-final", reasoning.EncryptedContent.Value)
				require.Len(t, reasoning.Summary, 2)
				require.Equal(t, "first", reasoning.Summary[0].Text)
				require.Equal(t, "second", reasoning.Summary[1].Text)
			}
		})
	}
}

func TestResponsesStreamFinalReasoningMetadata(t *testing.T) {
	t.Parallel()
	for _, initial := range []string{"", "partial-encrypted"} {
		t.Run(initial, func(t *testing.T) {
			t.Parallel()
			sms := newStreamingMockServer()
			defer sms.close()
			sms.chunks = []string{
				responsesSSEEvent("response.output_item.added", `{"type":"response.output_item.added","output_index":0,"item":{"id":"rs_1","type":"reasoning","encrypted_content":"`+initial+`","summary":[]}}`),
				responsesSSEEvent("response.output_item.done", `{"type":"response.output_item.done","output_index":0,"item":{"id":"rs_1","type":"reasoning","encrypted_content":"encrypted-final","summary":[{"type":"summary_text","text":"complete summary"}]}}`),
				responsesSSEEvent("response.completed", `{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}`),
			}
			m := newResponsesProvider(t, sms.server.URL)
			stream, err := m.Stream(t.Context(), fantasy.Call{Prompt: testPrompt})
			require.NoError(t, err)
			var metadata *ResponsesReasoningMetadata
			for part := range stream {
				require.NotEqual(t, fantasy.StreamPartTypeError, part.Type)
				if part.Type == fantasy.StreamPartTypeReasoningEnd {
					metadata = GetReasoningMetadata(fantasy.ProviderOptions(part.ProviderMetadata))
				}
			}
			require.NotNil(t, metadata)
			require.NotNil(t, metadata.EncryptedContent)
			require.Equal(t, "encrypted-final", *metadata.EncryptedContent)
			require.Equal(t, []string{"complete summary"}, metadata.Summary)
		})
	}
}
