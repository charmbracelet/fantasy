package providertests

import (
	"encoding/json"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openai"
	"github.com/stretchr/testify/require"
)

func TestProviderRegistry_Serialization_OpenAIOptions(t *testing.T) {
	msg := fantasy.Message{
		Role: fantasy.MessageRoleUser,
		Content: []fantasy.MessagePart{
			fantasy.TextPart{Text: "hi"},
		},
		ProviderOptions: fantasy.ProviderOptions{
			openai.Name: &openai.ProviderOptions{User: fantasy.Opt("tester")},
		},
	}

	data, err := json.Marshal(msg)
	require.NoError(t, err)

	var raw struct {
		ProviderOptions map[string]map[string]any `json:"provider_options"`
	}
	require.NoError(t, json.Unmarshal(data, &raw))

	po, ok := raw.ProviderOptions[openai.Name]
	require.True(t, ok)
	require.Equal(t, openai.TypeProviderOptions, po["type"]) // no magic strings
	// ensure inner data has the field we set
	inner, ok := po["data"].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "tester", inner["user"])

	var decoded fantasy.Message
	require.NoError(t, json.Unmarshal(data, &decoded))

	got, ok := decoded.ProviderOptions[openai.Name]
	require.True(t, ok)
	opt, ok := got.(*openai.ProviderOptions)
	require.True(t, ok)
	require.NotNil(t, opt.User)
	require.Equal(t, "tester", *opt.User)
}

func TestProviderRegistry_Serialization_OpenAIResponses(t *testing.T) {
	// Use ResponsesProviderOptions in provider options
	msg := fantasy.Message{
		Role: fantasy.MessageRoleUser,
		Content: []fantasy.MessagePart{
			fantasy.TextPart{Text: "hello"},
		},
		ProviderOptions: fantasy.ProviderOptions{
			openai.Name: &openai.ResponsesProviderOptions{
				PromptCacheKey:    fantasy.Opt("cache-key-1"),
				ParallelToolCalls: fantasy.Opt(true),
			},
		},
	}

	data, err := json.Marshal(msg)
	require.NoError(t, err)

	// JSON should include the typed wrapper with constant TypeResponsesProviderOptions
	var raw struct {
		ProviderOptions map[string]map[string]any `json:"provider_options"`
	}
	require.NoError(t, json.Unmarshal(data, &raw))

	po := raw.ProviderOptions[openai.Name]
	require.Equal(t, openai.TypeResponsesProviderOptions, po["type"]) // no magic strings
	inner, ok := po["data"].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "cache-key-1", inner["prompt_cache_key"])
	require.Equal(t, true, inner["parallel_tool_calls"])

	// Unmarshal back and assert concrete type
	var decoded fantasy.Message
	require.NoError(t, json.Unmarshal(data, &decoded))
	got := decoded.ProviderOptions[openai.Name]
	reqOpts, ok := got.(*openai.ResponsesProviderOptions)
	require.True(t, ok)
	require.NotNil(t, reqOpts.PromptCacheKey)
	require.Equal(t, "cache-key-1", *reqOpts.PromptCacheKey)
	require.NotNil(t, reqOpts.ParallelToolCalls)
	require.Equal(t, true, *reqOpts.ParallelToolCalls)
}

func TestProviderRegistry_Serialization_OpenAIResponsesReasoningMetadata(t *testing.T) {
	resp := fantasy.Response{
		Content: []fantasy.Content{
			fantasy.TextContent{
				Text: "",
				ProviderMetadata: fantasy.ProviderMetadata{
					openai.Name: &openai.ResponsesReasoningMetadata{
						ItemID:  "item-123",
						Summary: []string{"part1", "part2"},
					},
				},
			},
		},
	}

	data, err := json.Marshal(resp)
	require.NoError(t, err)

	// Ensure the provider metadata is wrapped with type using constant
	var raw struct {
		Content []struct {
			Type string         `json:"type"`
			Data map[string]any `json:"data"`
		} `json:"content"`
	}
	require.NoError(t, json.Unmarshal(data, &raw))
	require.Greater(t, len(raw.Content), 0)
	tc := raw.Content[0]
	pm, ok := tc.Data["provider_metadata"].(map[string]any)
	require.True(t, ok)
	om, ok := pm[openai.Name].(map[string]any)
	require.True(t, ok)
	require.Equal(t, openai.TypeResponsesReasoningMetadata, om["type"]) // no magic strings
	inner, ok := om["data"].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "item-123", inner["item_id"])

	// Unmarshal back
	var decoded fantasy.Response
	require.NoError(t, json.Unmarshal(data, &decoded))
	pmDecoded := decoded.Content[0].(fantasy.TextContent).ProviderMetadata
	val, ok := pmDecoded[openai.Name]
	require.True(t, ok)
	meta, ok := val.(*openai.ResponsesReasoningMetadata)
	require.True(t, ok)
	require.Equal(t, "item-123", meta.ItemID)
	require.Equal(t, []string{"part1", "part2"}, meta.Summary)
}
