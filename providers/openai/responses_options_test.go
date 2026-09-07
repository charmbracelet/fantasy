package openai

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIsResponsesModel(t *testing.T) {
	tests := []struct {
		modelID string
		want    bool
	}{
		// Explicitly listed models.
		{"gpt-4.1", true},
		{"gpt-4o-mini", true},
		{"chatgpt-4o-latest", true},
		{"o3", true},
		{"gpt-oss-120b", true},

		// Generations caught by the pattern, listed or not.
		{"gpt-5", true},
		{"gpt-5.1-codex", true},
		{"gpt-6-astra", true},
		{"GPT-6-ASTRA", true},
		{"gpt-10-turbo", true},

		// Everything predating the Responses API stays on chat
		// completions.
		{"gpt-3.5-turbo-1106", true}, // in the explicit list
		{"gpt-3-turbo-instruct", false},
		{"babbage-002", false},
		{"davinci-002", false},
		{"some-custom-model", false},
	}

	for _, tt := range tests {
		assert.Equal(t, tt.want, IsResponsesModel(tt.modelID), tt.modelID)
	}
}

func TestIsResponsesReasoningModel(t *testing.T) {
	tests := []struct {
		modelID string
		want    bool
	}{
		{"gpt-5.1-codex", true},
		{"gpt-6-astra", true},
		{"o4-mini", true},
		{"gpt-oss-120b", true},

		{"gpt-4.1-mini", true}, // gpt-4 matches, as before
		{"gpt-3-turbo-instruct", false},
		{"some-custom-model", false},
	}

	for _, tt := range tests {
		assert.Equal(t, tt.want, IsResponsesReasoningModel(tt.modelID), tt.modelID)
	}
}
