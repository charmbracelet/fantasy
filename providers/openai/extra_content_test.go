package openai

import (
	"encoding/json"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestToolCallExtraContent(t *testing.T) {
	t.Parallel()

	t.Run("extracts extra_content", func(t *testing.T) {
		t.Parallel()
		raw := `{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{}"},"extra_content":{"google":{"thought_signature":"sig-123"}}}`
		require.JSONEq(t, `{"google":{"thought_signature":"sig-123"}}`, string(toolCallExtraContent(raw)))
	})

	t.Run("missing extra_content", func(t *testing.T) {
		t.Parallel()
		raw := `{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{}"}}`
		require.Nil(t, toolCallExtraContent(raw))
	})

	t.Run("null extra_content", func(t *testing.T) {
		t.Parallel()
		require.Nil(t, toolCallExtraContent(`{"extra_content":null}`))
	})

	t.Run("malformed json", func(t *testing.T) {
		t.Parallel()
		require.Nil(t, toolCallExtraContent(`{not-json`))
	})
}

func TestToolCallProviderMetadata(t *testing.T) {
	t.Parallel()

	raw := `{"extra_content":{"google":{"thought_signature":"sig-123"}}}`
	meta := toolCallProviderMetadata(raw)
	require.Len(t, meta, 1)
	extra, ok := meta[Name].(*ToolCallExtraContent)
	require.True(t, ok)
	require.JSONEq(t, `{"google":{"thought_signature":"sig-123"}}`, string(extra.ExtraContent))

	require.Nil(t, toolCallProviderMetadata(`{"id":"call_1"}`))
}

func TestToolCallExtraContentJSONRoundTrip(t *testing.T) {
	t.Parallel()

	meta := fantasy.ProviderMetadata{Name: &ToolCallExtraContent{
		ExtraContent: json.RawMessage(`{"google":{"thought_signature":"sig-123"}}`),
	}}
	data, err := json.Marshal(meta)
	require.NoError(t, err)

	// Persistence layers decode maps through the provider type registry, which
	// routes each wrapper to its registered constructor.
	var raw map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(data, &raw))
	decoded, err := fantasy.UnmarshalProviderMetadata(raw)
	require.NoError(t, err)
	extra, ok := decoded[Name].(*ToolCallExtraContent)
	require.True(t, ok)
	require.JSONEq(t, `{"google":{"thought_signature":"sig-123"}}`, string(extra.ExtraContent))
}

func TestDefaultToPromptReplaysToolCallExtraContent(t *testing.T) {
	t.Parallel()

	prompt := fantasy.Prompt{
		{
			Role:    fantasy.MessageRoleUser,
			Content: []fantasy.MessagePart{fantasy.TextPart{Text: "Weather in Paris?"}},
		},
		{
			Role: fantasy.MessageRoleAssistant,
			Content: []fantasy.MessagePart{
				fantasy.ToolCallPart{
					ToolCallID: "call_1",
					ToolName:   "get_weather",
					Input:      `{"city":"Paris"}`,
					ProviderOptions: fantasy.ProviderOptions{Name: &ToolCallExtraContent{
						ExtraContent: json.RawMessage(`{"google":{"thought_signature":"sig-123"}}`),
					}},
				},
			},
		},
	}

	messages, warnings := DefaultToPrompt(prompt, Name, "gemini-3.8-flash")
	require.Empty(t, warnings)
	require.Len(t, messages, 2)

	data, err := json.Marshal(messages[1])
	require.NoError(t, err)
	require.Contains(t, string(data), `"extra_content":{"google":{"thought_signature":"sig-123"}}`)
	require.Contains(t, string(data), `"name":"get_weather"`)
}

func TestDefaultToPromptOmitsAbsentExtraContent(t *testing.T) {
	t.Parallel()

	prompt := fantasy.Prompt{
		{
			Role: fantasy.MessageRoleAssistant,
			Content: []fantasy.MessagePart{
				fantasy.ToolCallPart{ToolCallID: "call_1", ToolName: "get_weather", Input: `{}`},
			},
		},
	}

	messages, warnings := DefaultToPrompt(prompt, Name, "gpt-5")
	require.Empty(t, warnings)
	require.Len(t, messages, 1)

	data, err := json.Marshal(messages[0])
	require.NoError(t, err)
	require.NotContains(t, string(data), "extra_content")
}
