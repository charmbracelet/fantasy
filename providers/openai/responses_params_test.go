package openai

import (
	"encoding/json"
	"testing"

	"charm.land/fantasy"
	"github.com/charmbracelet/openai-go/responses"
	"github.com/stretchr/testify/require"
)

func TestPrepareParams_Store(t *testing.T) {
	t.Parallel()

	lm := testResponsesLM()
	prompt := fantasy.Prompt{testTextMessage(fantasy.MessageRoleUser, "hello")}

	tests := []struct {
		name      string
		opts      *ResponsesProviderOptions
		wantStore bool
	}{
		{
			name:      "store true",
			opts:      &ResponsesProviderOptions{Store: new(true)},
			wantStore: true,
		},
		{
			name:      "store false",
			opts:      &ResponsesProviderOptions{Store: new(false)},
			wantStore: false,
		},
		{
			name:      "store default",
			opts:      &ResponsesProviderOptions{},
			wantStore: false,
		},
		{
			name:      "no provider options",
			opts:      nil,
			wantStore: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			params, warnings, err := lm.prepareParams(testCall(prompt, tt.opts))
			require.NoError(t, err)
			require.Empty(t, warnings)
			require.True(t, params.Store.Valid())
			require.Equal(t, tt.wantStore, params.Store.Value)
		})
	}
}

func TestPrepareParams_PreviousResponseID(t *testing.T) {
	t.Parallel()

	lm := testResponsesLM()
	prompt := fantasy.Prompt{testTextMessage(fantasy.MessageRoleUser, "hello")}

	t.Run("forwarded", func(t *testing.T) {
		t.Parallel()

		params, warnings, err := lm.prepareParams(testCall(prompt, &ResponsesProviderOptions{
			PreviousResponseID: new("resp_abc123"),
			Store:              new(true),
		}))
		require.NoError(t, err)
		require.Empty(t, warnings)
		require.True(t, params.PreviousResponseID.Valid())
		require.Equal(t, "resp_abc123", params.PreviousResponseID.Value)
	})

	t.Run("not set", func(t *testing.T) {
		t.Parallel()

		params, warnings, err := lm.prepareParams(testCall(prompt, &ResponsesProviderOptions{}))
		require.NoError(t, err)
		require.Empty(t, warnings)
		require.False(t, params.PreviousResponseID.Valid())
	})

	t.Run("empty string ignored", func(t *testing.T) {
		t.Parallel()

		params, warnings, err := lm.prepareParams(testCall(prompt, &ResponsesProviderOptions{
			PreviousResponseID: new(""),
		}))
		require.NoError(t, err)
		require.Empty(t, warnings)
		require.False(t, params.PreviousResponseID.Valid())
	})
}

func TestPrepareParams_PreviousResponseID_Validation(t *testing.T) {
	t.Parallel()

	lm := testResponsesLM()
	opts := &ResponsesProviderOptions{
		PreviousResponseID: new("resp_abc123"),
		Store:              new(true),
	}

	t.Run("rejects with assistant messages", func(t *testing.T) {
		t.Parallel()

		_, _, err := lm.prepareParams(testCall(fantasy.Prompt{
			testTextMessage(fantasy.MessageRoleUser, "hello"),
			testTextMessage(fantasy.MessageRoleAssistant, "hi there"),
		}, opts))
		require.EqualError(t, err, previousResponseIDHistoryError)
	})

	t.Run("allows user-only prompt", func(t *testing.T) {
		t.Parallel()

		_, warnings, err := lm.prepareParams(testCall(fantasy.Prompt{
			testTextMessage(fantasy.MessageRoleUser, "hello"),
			testTextMessage(fantasy.MessageRoleUser, "follow up"),
		}, opts))
		require.NoError(t, err)
		require.Empty(t, warnings)
	})

	t.Run("allows system + user prompt", func(t *testing.T) {
		t.Parallel()

		_, warnings, err := lm.prepareParams(testCall(fantasy.Prompt{
			testTextMessage(fantasy.MessageRoleSystem, "be concise"),
			testTextMessage(fantasy.MessageRoleUser, "hello"),
		}, opts))
		require.NoError(t, err)
		require.Empty(t, warnings)
	})

	t.Run("rejects tool messages", func(t *testing.T) {
		t.Parallel()

		_, _, err := lm.prepareParams(testCall(fantasy.Prompt{
			testToolResultMessage("done"),
			testTextMessage(fantasy.MessageRoleUser, "hello"),
		}, opts))
		require.EqualError(t, err, previousResponseIDHistoryError)
	})

	t.Run("rejects without store", func(t *testing.T) {
		t.Parallel()

		_, _, err := lm.prepareParams(testCall(fantasy.Prompt{
			testTextMessage(fantasy.MessageRoleUser, "hello"),
		}, &ResponsesProviderOptions{
			PreviousResponseID: new("resp_abc123"),
		}))
		require.EqualError(t, err, previousResponseIDStoreError)
	})

	t.Run("rejects with store false", func(t *testing.T) {
		t.Parallel()

		_, _, err := lm.prepareParams(testCall(fantasy.Prompt{
			testTextMessage(fantasy.MessageRoleUser, "hello"),
		}, &ResponsesProviderOptions{
			PreviousResponseID: new("resp_abc123"),
			Store:              new(false),
		}))
		require.EqualError(t, err, previousResponseIDStoreError)
	})
}

func TestValidatePreviousResponseIDPrompt(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		prompt  fantasy.Prompt
		wantErr bool
	}{
		{
			name:   "empty prompt",
			prompt: nil,
		},
		{
			name: "user-only messages",
			prompt: fantasy.Prompt{
				testTextMessage(fantasy.MessageRoleUser, "hello"),
				testTextMessage(fantasy.MessageRoleUser, "follow up"),
			},
		},
		{
			name: "system + user messages",
			prompt: fantasy.Prompt{
				testTextMessage(fantasy.MessageRoleSystem, "be concise"),
				testTextMessage(fantasy.MessageRoleUser, "hello"),
			},
		},
		{
			name: "contains assistant message",
			prompt: fantasy.Prompt{
				testTextMessage(fantasy.MessageRoleAssistant, "hi there"),
			},
			wantErr: true,
		},
		{
			name: "assistant in the middle",
			prompt: fantasy.Prompt{
				testTextMessage(fantasy.MessageRoleUser, "hello"),
				testTextMessage(fantasy.MessageRoleAssistant, "hi there"),
				testTextMessage(fantasy.MessageRoleUser, "follow up"),
			},
			wantErr: true,
		},
		{
			name: "contains tool message",
			prompt: fantasy.Prompt{
				testToolResultMessage("done"),
				testTextMessage(fantasy.MessageRoleUser, "follow up"),
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			err := validatePreviousResponseIDPrompt(tt.prompt)
			if tt.wantErr {
				require.EqualError(t, err, previousResponseIDHistoryError)
				return
			}

			require.NoError(t, err)
		})
	}
}

func TestResponsesProviderMetadata_Helper(t *testing.T) {
	t.Parallel()

	t.Run("non-empty id", func(t *testing.T) {
		t.Parallel()

		metadata := responsesProviderMetadata(responses.Response{ID: "resp_123"})
		require.Len(t, metadata, 1)

		providerMetadata, ok := metadata[Name].(*ResponsesProviderMetadata)
		require.True(t, ok)
		require.Equal(t, "resp_123", providerMetadata.ResponseID)
	})

	t.Run("empty id", func(t *testing.T) {
		t.Parallel()

		metadata := responsesProviderMetadata(responses.Response{})
		require.Empty(t, metadata)
	})
}

func TestResponsesProviderMetadata_JSON(t *testing.T) {
	t.Parallel()

	encoded, err := json.Marshal(ResponsesProviderMetadata{ResponseID: "resp_123"})
	require.NoError(t, err)
	require.Contains(t, string(encoded), `"response_id":"resp_123"`)

	decoded, err := fantasy.UnmarshalProviderMetadata(map[string]json.RawMessage{
		Name: encoded,
	})
	require.NoError(t, err)

	providerMetadata, ok := decoded[Name].(*ResponsesProviderMetadata)
	require.True(t, ok)
	require.Equal(t, "resp_123", providerMetadata.ResponseID)
}

func testCall(prompt fantasy.Prompt, opts *ResponsesProviderOptions) fantasy.Call {
	call := fantasy.Call{
		Prompt: prompt,
	}
	if opts != nil {
		call.ProviderOptions = fantasy.ProviderOptions{
			Name: opts,
		}
	}
	return call
}

func testResponsesLM() responsesLanguageModel {
	return responsesLanguageModel{
		provider: Name,
		modelID:  "gpt-4o",
	}
}

func testTextMessage(role fantasy.MessageRole, text string) fantasy.Message {
	return fantasy.Message{
		Role: role,
		Content: []fantasy.MessagePart{
			fantasy.TextPart{Text: text},
		},
	}
}

func testToolResultMessage(text string) fantasy.Message {
	return fantasy.Message{
		Role: fantasy.MessageRoleTool,
		Content: []fantasy.MessagePart{
			fantasy.ToolResultPart{
				ToolCallID: "call_123",
				Output: fantasy.ToolResultOutputContentText{
					Text: text,
				},
			},
		},
	}
}

func TestGetResponsesModelConfig_ReasoningOptions(t *testing.T) {
	t.Parallel()

	tests := []struct {
		modelID         string
		wantReasoning   bool
		wantAcceptsOpts bool
		wantSystemMode  string
	}{
		{"gpt-5", true, true, "developer"},
		{"gpt-5.4-mini", true, true, "developer"},
		{"o3", true, true, "developer"},
		{"o1-mini", true, true, "remove"},
		{"gpt-6-sol", true, true, "developer"},
		{"gpt-6.1-sol", true, true, "developer"},
		{"grok-4.7", false, true, "system"},
		{"grok-4.3", false, true, "system"},
		{"some-new-model", false, true, "system"},
		{"gpt-4o", false, false, "system"},
		{"gpt-4.1-mini", false, false, "system"},
		{"gpt-5-chat-latest", false, false, "system"},
		{"gpt-3.5-turbo", false, false, "system"},
		{"chatgpt-4o-latest", false, false, "system"},
	}

	for _, tt := range tests {
		t.Run(tt.modelID, func(t *testing.T) {
			t.Parallel()

			cfg := getResponsesModelConfig(tt.modelID)
			require.Equal(t, tt.wantReasoning, cfg.isReasoningModel)
			require.Equal(t, tt.wantAcceptsOpts, cfg.acceptsReasoningOptions)
			require.Equal(t, tt.wantSystemMode, cfg.systemMessageMode)
		})
	}
}

func TestPrepareParams_ReasoningEffortByModelID(t *testing.T) {
	t.Parallel()

	prompt := fantasy.Prompt{testTextMessage(fantasy.MessageRoleUser, "hello")}
	opts := &ResponsesProviderOptions{
		ReasoningEffort:  new(ReasoningEffortHigh),
		ReasoningSummary: new("auto"),
	}

	for _, modelID := range []string{"gpt-6-sol", "gpt-6.1-sol", "grok-4.7", "grok-4.3", "gpt-5.4", "o3"} {
		t.Run(modelID, func(t *testing.T) {
			t.Parallel()

			lm := responsesLanguageModel{provider: Name, modelID: modelID}
			params, warnings, err := lm.prepareParams(testCall(prompt, opts))
			require.NoError(t, err)
			require.Empty(t, warnings)

			body, err := json.Marshal(params)
			require.NoError(t, err)
			var got struct {
				Reasoning struct {
					Effort  string `json:"effort"`
					Summary string `json:"summary"`
				} `json:"reasoning"`
			}
			require.NoError(t, json.Unmarshal(body, &got))
			require.Equal(t, "high", got.Reasoning.Effort)
			require.Equal(t, "auto", got.Reasoning.Summary)
		})
	}

	t.Run("gpt-4o warns and omits", func(t *testing.T) {
		t.Parallel()

		lm := testResponsesLM()
		params, warnings, err := lm.prepareParams(testCall(prompt, opts))
		require.NoError(t, err)
		require.Len(t, warnings, 2)
		require.Equal(t, "reasoningEffort", warnings[0].Setting)
		require.Equal(t, "reasoningSummary", warnings[1].Setting)

		body, err := json.Marshal(params)
		require.NoError(t, err)
		require.NotContains(t, string(body), `"reasoning"`)
	})

	t.Run("grok keeps temperature", func(t *testing.T) {
		t.Parallel()

		lm := responsesLanguageModel{provider: Name, modelID: "grok-4.7"}
		call := testCall(prompt, opts)
		call.Temperature = new(0.5)
		params, warnings, err := lm.prepareParams(call)
		require.NoError(t, err)
		require.Empty(t, warnings)
		require.True(t, params.Temperature.Valid())
	})
}
