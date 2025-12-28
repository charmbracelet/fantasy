package openaicompat

import (
	"testing"
)

func TestNormalizeToolCallID(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "valid 9-char alphanumeric",
			input:    "abc123XYZ",
			expected: "abc123XYZ",
		},
		{
			name:     "UUID-style with underscore",
			input:    "call_1d9af98d68f24568a1aefd62",
			expected: "call1d9af",
		},
		{
			name:     "UUID-style with hyphens",
			input:    "c7f85f48-1ca1-4b3a-90aa-3580551a814f",
			expected: "c7f85f481",
		},
		{
			name:     "long string",
			input:    "thisisaverylongtoolcallidthatexceeds9characters",
			expected: "thisisave",
		},
		{
			name:     "short string",
			input:    "abc",
			expected: "abcghijkl",
		},
		{
			name:     "mixed special characters",
			input:    "call_123-456_ABC",
			expected: "call12345",
		},
		{
			name:     "exactly 9 chars with special",
			input:    "call_12_3",
			expected: "call123op",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := normalizeToolCallID(tt.input)
			if result != tt.expected {
				t.Errorf("normalizeToolCallID(%q) = %q, want %q", tt.input, result, tt.expected)
			}
			// Verify the result is always 9 characters
			if len(result) != 9 {
				t.Errorf("normalizeToolCallID(%q) returned length %d, want 9", tt.input, len(result))
			}
			// Verify the result only contains alphanumeric characters
			for _, c := range result {
				if !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) {
					t.Errorf("normalizeToolCallID(%q) contains invalid character: %c", tt.input, c)
				}
			}
		})
	}
}

func TestNormalizeToolCallIDConsistency(t *testing.T) {
	// Test that the same input always produces the same output
	input := "call_1d9af98d68f24568a1aefd62"
	result1 := normalizeToolCallID(input)
	result2 := normalizeToolCallID(input)
	if result1 != result2 {
		t.Errorf("normalizeToolCallID(%q) is not consistent: %q vs %q", input, result1, result2)
	}
}

func TestNormalizeToolCallIDEdgeCases(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{
			name:  "empty string",
			input: "",
		},
		{
			name:  "only special characters",
			input: "---___",
		},
		{
			name:  "only underscores",
			input: "_________",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := normalizeToolCallID(tt.input)
			// Should still return a 9-character alphanumeric string
			if len(result) != 9 {
				t.Errorf("normalizeToolCallID(%q) returned length %d, want 9", tt.input, len(result))
			}
			// Verify the result only contains alphanumeric characters
			for _, c := range result {
				if !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) {
					t.Errorf("normalizeToolCallID(%q) contains invalid character: %c", tt.input, c)
				}
			}
		})
	}
}