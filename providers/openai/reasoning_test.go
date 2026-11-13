package openai

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestReasoningModelDetection(t *testing.T) {
	tests := []struct {
		modelID   string
		chat      bool
		responses bool
	}{
		{"gpt-5", true, true},
		{"gpt-5.6-sol", true, true},
		{"gpt-6-astra", true, true},
		{"GPT-6-ASTRA", true, true},
		{"gpt-6.1-sol", true, true},
		{"gpt-6.1-terra", true, true},
		{"us.openai.gpt-6-sol", true, true}, // Amazon Bedrock inference profile
		{"openai.gpt-6-luna", true, true},   // Amazon Bedrock Mantle
		{"o1", true, true},
		{"o3-mini", true, true},
		{"o4-mini", true, true},
		{"us.openai.o3", true, true},
		{"gpt-oss-120b", true, true},
		{"openai.gpt-oss-120b-1:0", true, true},
		{"codex-mini-latest", true, true},
		{"computer-use-preview", true, true},

		{"gpt-5-chat-latest", false, false},
		{"gpt-5.1-chat", false, false},
		{"chatgpt-4o-latest", false, false},
		{"gpt-4o", false, false},
		{"gpt-4.1-mini", false, false},
		{"gpt-4.5-preview", false, false},
		{"gpt-3.5-turbo", false, false},
		{"gpt-35-turbo", false, false}, // Azure's name for gpt-3.5-turbo

		// A newer OpenAI generation: Chat Completions only trusts known
		// families, Responses treats it as a current OpenAI model.
		{"gpt-10-turbo", false, true},

		// Ids outside the OpenAI families, such as xAI grok models on the
		// Responses API, do not get the OpenAI reasoning-model rules.
		{"grok-4.7", false, false},
		{"some-custom-model", false, false},
		{"llama-3.3-70b", false, false},
	}

	for _, tt := range tests {
		assert.Equal(t, tt.chat, isReasoningModel(tt.modelID), "chat completions: "+tt.modelID)
		assert.Equal(t, tt.responses, getResponsesModelConfig(tt.modelID).isReasoningModel, "responses: "+tt.modelID)
	}
}

func TestResponsesSystemMessageMode(t *testing.T) {
	tests := map[string]string{
		"gpt-4o":      "system",
		"gpt-6-astra": "developer",
		"o3":          "developer",
		"o1-mini":     "remove",
		"o1-preview":  "remove",
		"grok-4.7":    "system",
	}

	for modelID, want := range tests {
		assert.Equal(t, want, getResponsesModelConfig(modelID).systemMessageMode, modelID)
	}
}
