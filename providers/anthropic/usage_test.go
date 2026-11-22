package anthropic

import (
	"context"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestGenerate_TotalTokensIncludesCache(t *testing.T) {
	t.Parallel()

	resp := mockAnthropicGenerateResponse()
	usage := resp["usage"].(map[string]any)
	usage["input_tokens"] = 5
	usage["output_tokens"] = 2
	usage["cache_creation_input_tokens"] = 100
	usage["cache_read_input_tokens"] = 1000

	server, _ := newAnthropicJSONServer(resp)
	defer server.Close()

	provider, err := New(WithAPIKey("test-api-key"), WithBaseURL(server.URL))
	require.NoError(t, err)
	model, err := provider.LanguageModel(context.Background(), "claude-sonnet-4-20250514")
	require.NoError(t, err)

	got, err := model.Generate(context.Background(), fantasy.Call{Prompt: testPrompt()})
	require.NoError(t, err)
	require.Equal(t, int64(5), got.Usage.InputTokens)
	require.Equal(t, int64(100), got.Usage.CacheCreationTokens)
	require.Equal(t, int64(1000), got.Usage.CacheReadTokens)
	require.Equal(t, int64(1107), got.Usage.TotalTokens)
}

func TestStream_TotalTokensIncludesCache(t *testing.T) {
	t.Parallel()

	server, _ := newAnthropicStreamingServer([]string{
		anthropicSSEEvent("message_start", `{"type":"message_start","message":{"id":"msg_cache","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[],"stop_reason":null,"usage":{"input_tokens":5,"output_tokens":0,"cache_creation_input_tokens":100,"cache_read_input_tokens":1000}}}`),
		anthropicSSEEvent("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
		anthropicSSEEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}`),
		anthropicSSEEvent("content_block_stop", `{"type":"content_block_stop","index":0}`),
		anthropicSSEEvent("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}`),
		anthropicSSEEvent("message_stop", `{"type":"message_stop"}`),
	})
	defer server.Close()

	provider, err := New(WithAPIKey("test-api-key"), WithBaseURL(server.URL))
	require.NoError(t, err)
	model, err := provider.LanguageModel(context.Background(), "claude-sonnet-4-20250514")
	require.NoError(t, err)

	stream, err := model.Stream(context.Background(), fantasy.Call{Prompt: testPrompt()})
	require.NoError(t, err)

	var finish *fantasy.StreamPart
	stream(func(p fantasy.StreamPart) bool {
		if p.Type == fantasy.StreamPartTypeFinish {
			p := p
			finish = &p
		}
		return true
	})
	require.NotNil(t, finish)
	require.Equal(t, int64(1000), finish.Usage.CacheReadTokens)
	require.Equal(t, int64(1107), finish.Usage.TotalTokens)
}
