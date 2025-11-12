package providertests

import (
	"encoding/json"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/anthropic"
	"charm.land/fantasy/providers/google"
	"charm.land/fantasy/providers/openai"
	"charm.land/fantasy/providers/openaicompat"
	"charm.land/fantasy/providers/openrouter"
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

func TestProviderRegistry_Serialization_AnthropicOptions(t *testing.T) {
	sendReasoning := true
	msg := fantasy.Message{
		Role: fantasy.MessageRoleUser,
		Content: []fantasy.MessagePart{
			fantasy.TextPart{Text: "test message"},
		},
		ProviderOptions: fantasy.ProviderOptions{
			anthropic.Name: &anthropic.ProviderOptions{
				SendReasoning: &sendReasoning,
			},
		},
	}

	data, err := json.Marshal(msg)
	require.NoError(t, err)

	var decoded fantasy.Message
	require.NoError(t, json.Unmarshal(data, &decoded))

	got, ok := decoded.ProviderOptions[anthropic.Name]
	require.True(t, ok)
	opt, ok := got.(*anthropic.ProviderOptions)
	require.True(t, ok)
	require.NotNil(t, opt.SendReasoning)
	require.Equal(t, true, *opt.SendReasoning)
}

func TestProviderRegistry_Serialization_GoogleOptions(t *testing.T) {
	msg := fantasy.Message{
		Role: fantasy.MessageRoleUser,
		Content: []fantasy.MessagePart{
			fantasy.TextPart{Text: "test message"},
		},
		ProviderOptions: fantasy.ProviderOptions{
			google.Name: &google.ProviderOptions{
				CachedContent: "cached-123",
				Threshold:     "BLOCK_ONLY_HIGH",
			},
		},
	}

	data, err := json.Marshal(msg)
	require.NoError(t, err)

	var decoded fantasy.Message
	require.NoError(t, json.Unmarshal(data, &decoded))

	got, ok := decoded.ProviderOptions[google.Name]
	require.True(t, ok)
	opt, ok := got.(*google.ProviderOptions)
	require.True(t, ok)
	require.Equal(t, "cached-123", opt.CachedContent)
	require.Equal(t, "BLOCK_ONLY_HIGH", opt.Threshold)
}

func TestProviderRegistry_Serialization_OpenRouterOptions(t *testing.T) {
	includeUsage := true
	msg := fantasy.Message{
		Role: fantasy.MessageRoleUser,
		Content: []fantasy.MessagePart{
			fantasy.TextPart{Text: "test message"},
		},
		ProviderOptions: fantasy.ProviderOptions{
			openrouter.Name: &openrouter.ProviderOptions{
				IncludeUsage: &includeUsage,
				User:         fantasy.Opt("test-user"),
			},
		},
	}

	data, err := json.Marshal(msg)
	require.NoError(t, err)

	var decoded fantasy.Message
	require.NoError(t, json.Unmarshal(data, &decoded))

	got, ok := decoded.ProviderOptions[openrouter.Name]
	require.True(t, ok)
	opt, ok := got.(*openrouter.ProviderOptions)
	require.True(t, ok)
	require.NotNil(t, opt.IncludeUsage)
	require.Equal(t, true, *opt.IncludeUsage)
	require.NotNil(t, opt.User)
	require.Equal(t, "test-user", *opt.User)
}

func TestProviderRegistry_Serialization_OpenAICompatOptions(t *testing.T) {
	effort := openai.ReasoningEffortHigh
	msg := fantasy.Message{
		Role: fantasy.MessageRoleUser,
		Content: []fantasy.MessagePart{
			fantasy.TextPart{Text: "test message"},
		},
		ProviderOptions: fantasy.ProviderOptions{
			openaicompat.Name: &openaicompat.ProviderOptions{
				User:            fantasy.Opt("test-user"),
				ReasoningEffort: &effort,
			},
		},
	}

	data, err := json.Marshal(msg)
	require.NoError(t, err)

	var decoded fantasy.Message
	require.NoError(t, json.Unmarshal(data, &decoded))

	got, ok := decoded.ProviderOptions[openaicompat.Name]
	require.True(t, ok)
	opt, ok := got.(*openaicompat.ProviderOptions)
	require.True(t, ok)
	require.NotNil(t, opt.User)
	require.Equal(t, "test-user", *opt.User)
	require.NotNil(t, opt.ReasoningEffort)
	require.Equal(t, openai.ReasoningEffortHigh, *opt.ReasoningEffort)
}

func TestProviderRegistry_MultiProvider(t *testing.T) {
	// Test with multiple providers in one message
	sendReasoning := true
	msg := fantasy.Message{
		Role: fantasy.MessageRoleUser,
		Content: []fantasy.MessagePart{
			fantasy.TextPart{Text: "test"},
		},
		ProviderOptions: fantasy.ProviderOptions{
			openai.Name: &openai.ProviderOptions{User: fantasy.Opt("user1")},
			anthropic.Name: &anthropic.ProviderOptions{
				SendReasoning: &sendReasoning,
			},
		},
	}

	data, err := json.Marshal(msg)
	require.NoError(t, err)

	var decoded fantasy.Message
	require.NoError(t, json.Unmarshal(data, &decoded))

	// Check OpenAI options
	openaiOpt, ok := decoded.ProviderOptions[openai.Name]
	require.True(t, ok)
	openaiData, ok := openaiOpt.(*openai.ProviderOptions)
	require.True(t, ok)
	require.Equal(t, "user1", *openaiData.User)

	// Check Anthropic options
	anthropicOpt, ok := decoded.ProviderOptions[anthropic.Name]
	require.True(t, ok)
	anthropicData, ok := anthropicOpt.(*anthropic.ProviderOptions)
	require.True(t, ok)
	require.Equal(t, true, *anthropicData.SendReasoning)
}

func TestProviderRegistry_ErrorHandling(t *testing.T) {
	t.Run("unknown provider type", func(t *testing.T) {
		invalidJSON := `{
			"role": "user",
			"content": [{"type": "text", "data": {"text": "hi"}}],
			"provider_options": {
				"unknown": {
					"type": "unknown.provider.type",
					"data": {}
				}
			}
		}`

		var msg fantasy.Message
		err := json.Unmarshal([]byte(invalidJSON), &msg)
		require.Error(t, err)
		require.Contains(t, err.Error(), "unknown provider data type")
	})

	t.Run("malformed provider data", func(t *testing.T) {
		invalidJSON := `{
			"role": "user",
			"content": [{"type": "text", "data": {"text": "hi"}}],
			"provider_options": {
				"openai": "not-an-object"
			}
		}`

		var msg fantasy.Message
		err := json.Unmarshal([]byte(invalidJSON), &msg)
		require.Error(t, err)
	})
}

func TestProviderRegistry_AllTypesRegistered(t *testing.T) {
	// Verify all expected provider types are registered
	// We test that unmarshaling with proper type IDs doesn't fail with "unknown provider data type"
	tests := []struct {
		name         string
		providerName string
		data         fantasy.ProviderOptionsData
	}{
		{"OpenAI Options", openai.Name, &openai.ProviderOptions{}},
		{"OpenAI File Options", openai.Name, &openai.ProviderFileOptions{}},
		{"OpenAI Metadata", openai.Name, &openai.ProviderMetadata{}},
		{"OpenAI Responses Options", openai.Name, &openai.ResponsesProviderOptions{}},
		{"Anthropic Options", anthropic.Name, &anthropic.ProviderOptions{}},
		{"Google Options", google.Name, &google.ProviderOptions{}},
		{"OpenRouter Options", openrouter.Name, &openrouter.ProviderOptions{}},
		{"OpenAICompat Options", openaicompat.Name, &openaicompat.ProviderOptions{}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Create a message with the provider options
			msg := fantasy.Message{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "test"},
				},
				ProviderOptions: fantasy.ProviderOptions{
					tc.providerName: tc.data,
				},
			}

			// Marshal and unmarshal
			data, err := json.Marshal(msg)
			require.NoError(t, err)

			var decoded fantasy.Message
			err = json.Unmarshal(data, &decoded)
			require.NoError(t, err)

			// Verify the provider options exist
			_, ok := decoded.ProviderOptions[tc.providerName]
			require.True(t, ok, "Provider options should be present after round-trip")
		})
	}

	// Test metadata types separately as they go in different field
	metadataTests := []struct {
		name         string
		providerName string
		data         fantasy.ProviderOptionsData
	}{
		{"OpenAI Responses Reasoning Metadata", openai.Name, &openai.ResponsesReasoningMetadata{}},
		{"Anthropic Reasoning Metadata", anthropic.Name, &anthropic.ReasoningOptionMetadata{}},
		{"Google Reasoning Metadata", google.Name, &google.ReasoningMetadata{}},
		{"OpenRouter Metadata", openrouter.Name, &openrouter.ProviderMetadata{}},
	}

	for _, tc := range metadataTests {
		t.Run(tc.name, func(t *testing.T) {
			// Create a response with provider metadata
			resp := fantasy.Response{
				Content: []fantasy.Content{
					fantasy.TextContent{
						Text: "test",
						ProviderMetadata: fantasy.ProviderMetadata{
							tc.providerName: tc.data,
						},
					},
				},
			}

			// Marshal and unmarshal
			data, err := json.Marshal(resp)
			require.NoError(t, err)

			var decoded fantasy.Response
			err = json.Unmarshal(data, &decoded)
			require.NoError(t, err)

			// Verify the provider metadata exists
			textContent, ok := decoded.Content[0].(fantasy.TextContent)
			require.True(t, ok)
			_, ok = textContent.ProviderMetadata[tc.providerName]
			require.True(t, ok, "Provider metadata should be present after round-trip")
		})
	}
}
