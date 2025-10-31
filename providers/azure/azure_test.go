package azure

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParseAzureURL(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "full https openai azure url",
			input:    "https://my-resource.openai.azure.com",
			expected: "https://my-resource.openai.azure.com/openai/v1",
		},
		{
			name:     "full https openai azure url with trailing slash",
			input:    "https://my-resource.openai.azure.com/",
			expected: "https://my-resource.openai.azure.com/openai/v1",
		},
		{
			name:     "full https cognitiveservices azure url",
			input:    "https://my-resource.cognitiveservices.azure.com",
			expected: "https://my-resource.openai.azure.com/openai/v1",
		},
		{
			name:     "full https services.ai azure url with path",
			input:    "https://fantasy-playground-resource.services.ai.azure.com/api/projects/fantasy-playground",
			expected: "https://fantasy-playground-resource.openai.azure.com/openai/v1",
		},
		{
			name:     "openai azure url without protocol",
			input:    "my-resource.openai.azure.com",
			expected: "https://my-resource.openai.azure.com/openai/v1",
		},
		{
			name:     "cognitiveservices azure url without protocol",
			input:    "my-resource.cognitiveservices.azure.com",
			expected: "https://my-resource.openai.azure.com/openai/v1",
		},
		{
			name:     "services.ai azure url without protocol",
			input:    "fantasy-playground-resource.services.ai.azure.com/api/projects/fantasy-playground",
			expected: "https://fantasy-playground-resource.openai.azure.com/openai/v1",
		},
		{
			name:     "resource with hyphens",
			input:    "https://my-complex-resource-123.openai.azure.com",
			expected: "https://my-complex-resource-123.openai.azure.com/openai/v1",
		},
		{
			name:     "openai azure url with trailing slash",
			input:    "https://fantasy-playground-resource.openai.azure.com/",
			expected: "https://fantasy-playground-resource.openai.azure.com/openai/v1",
		},
		{
			name:     "cognitiveservices azure url with trailing slash",
			input:    "https://fantasy-playground-resource.cognitiveservices.azure.com/",
			expected: "https://fantasy-playground-resource.openai.azure.com/openai/v1",
		},
		{
			name:     "malformed url - non azure domain",
			input:    "https://non.sense.com",
			expected: "https://non.sense.com",
		},
		{
			name:     "malformed url - simple domain",
			input:    "example.com",
			expected: "https://example.com",
		},
		{
			name:     "custom endpoint with protocol",
			input:    "https://custom-endpoint.example.com",
			expected: "https://custom-endpoint.example.com",
		},
		{
			name:     "custom endpoint without protocol",
			input:    "custom-endpoint.example.com",
			expected: "https://custom-endpoint.example.com",
		},
		{
			name:     "localhost",
			input:    "http://localhost:8080",
			expected: "http://localhost:8080",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := parseAzureURL(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}
