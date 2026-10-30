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

		// Unknown models: Chat Completions serves any compatible model so it
		// only trusts known families, Responses assumes a current OpenAI model.
		{"gpt-10-turbo", false, true},
		{"some-custom-model", false, true},
		{"llama-3.3-70b", false, true},
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
	}

	for modelID, want := range tests {
		assert.Equal(t, want, getResponsesModelConfig(modelID).systemMessageMode, modelID)
	}
}
