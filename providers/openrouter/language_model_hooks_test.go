package openrouter

import (
	"encoding/json"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openai"
	"github.com/stretchr/testify/require"
)

// The openai/ and xai/ routes share the Responses API reasoning format and
// differ only in the format label they report, so the label has to keep
// following the model prefix.
func TestReasoningDetailsFormatFollowsModelPrefix(t *testing.T) {
	t.Parallel()

	encrypted := "ciphertext"
	prompt := fantasy.Prompt{
		fantasy.NewUserMessage("hello"),
		{
			Role: fantasy.MessageRoleAssistant,
			Content: []fantasy.MessagePart{
				fantasy.TextPart{Text: "answer"},
				fantasy.ReasoningPart{
					Text: "thinking",
					ProviderOptions: fantasy.ProviderOptions{
						openai.Name: &openai.ResponsesReasoningMetadata{
							ItemID:           "item-1",
							EncryptedContent: &encrypted,
							Summary:          []string{"first step", "second step"},
						},
					},
				},
			},
		},
	}

	tests := []struct {
		model  string
		format string
	}{
		{model: "openai/gpt-5.1", format: "openai-responses-v1"},
		{model: "xai/grok-4", format: "xai-responses-v1"},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			t.Parallel()

			messages, warnings := languageModelToPrompt(prompt, "", tt.model)
			require.Empty(t, warnings)

			encoded, err := json.Marshal(messages)
			require.NoError(t, err)
			require.Contains(t, string(encoded), `"format":"`+tt.format+`"`)
			require.Contains(t, string(encoded), `"type":"reasoning.summary"`)
			require.Contains(t, string(encoded), `"type":"reasoning.encrypted"`)
			require.Contains(t, string(encoded), `"data":"ciphertext"`)
		})
	}
}
