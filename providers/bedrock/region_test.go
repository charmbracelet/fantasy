package bedrock

import (
	"os"
	"testing"

	"github.com/stretchr/testify/require"
	"pgregory.net/rapid"
)

// Feature: amazon-nova-bedrock-support, Property 4: Region Prefix Idempotence
// Validates: Requirements 1.6, 5.1, 5.2
func TestProperty_RegionPrefixIdempotence(t *testing.T) {
	t.Parallel()

	rapid.Check(t, func(t *rapid.T) {
		// Generate model IDs matching the Nova pattern
		modelID := rapid.StringMatching(`amazon\.nova-(pro|lite|micro|premier)-v[0-9]+:[0-9]+`).Draw(t, "modelID")

		// Generate region strings matching AWS region pattern
		region := rapid.StringMatching(`[a-z]{2}-[a-z]+-[0-9]+`).Draw(t, "region")

		// Apply region prefix once
		once := applyRegionPrefix(modelID, region)

		// Apply region prefix twice
		twice := applyRegionPrefix(once, region)

		// Property: applying region prefix twice should equal applying once (idempotence)
		require.Equal(t, once, twice, "Region prefix should be idempotent")

		// Additional check: the result should start with the region prefix
		expectedPrefix := region[:2] + "."
		require.True(t, len(once) >= 3, "Result should have at least 3 characters")
		require.Equal(t, expectedPrefix, once[:3], "Result should start with region prefix")
	})
}

// Unit tests for region prefix edge cases

func TestApplyRegionPrefix_EmptyRegion(t *testing.T) {
	t.Parallel()

	modelID := "amazon.nova-pro-v1:0"
	result := applyRegionPrefix(modelID, "")

	// Should default to "us." prefix
	require.Equal(t, "us.amazon.nova-pro-v1:0", result)
}

func TestApplyRegionPrefix_RegionLessThan2Characters(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		modelID  string
		region   string
		expected string
	}{
		{
			name:     "single character region",
			modelID:  "amazon.nova-pro-v1:0",
			region:   "u",
			expected: "us.amazon.nova-pro-v1:0",
		},
		{
			name:     "empty region",
			modelID:  "amazon.nova-lite-v1:0",
			region:   "",
			expected: "us.amazon.nova-lite-v1:0",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := applyRegionPrefix(tc.modelID, tc.region)
			require.Equal(t, tc.expected, result)
		})
	}
}

func TestApplyRegionPrefix_ModelIDAlreadyWithPrefix(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		modelID  string
		region   string
		expected string
	}{
		{
			name:     "already has us prefix",
			modelID:  "us.amazon.nova-pro-v1:0",
			region:   "us-east-1",
			expected: "us.amazon.nova-pro-v1:0",
		},
		{
			name:     "already has eu prefix",
			modelID:  "eu.amazon.nova-lite-v1:0",
			region:   "eu-west-1",
			expected: "eu.amazon.nova-lite-v1:0",
		},
		{
			name:     "already has ap prefix",
			modelID:  "ap.amazon.nova-micro-v1:0",
			region:   "ap-southeast-1",
			expected: "ap.amazon.nova-micro-v1:0",
		},
		{
			name:     "different region prefix already exists",
			modelID:  "eu.amazon.nova-pro-v1:0",
			region:   "us-east-1",
			expected: "eu.amazon.nova-pro-v1:0",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := applyRegionPrefix(tc.modelID, tc.region)
			require.Equal(t, tc.expected, result)
		})
	}
}

func TestApplyRegionPrefix_AWSRegionEnvironmentVariable(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name      string
		envValue  string
		expected  string
		shouldSet bool
	}{
		{
			name:      "AWS_REGION set to us-west-2",
			envValue:  "us-west-2",
			expected:  "us-west-2",
			shouldSet: true,
		},
		{
			name:      "AWS_REGION set to eu-central-1",
			envValue:  "eu-central-1",
			expected:  "eu-central-1",
			shouldSet: true,
		},
		{
			name:      "AWS_REGION not set",
			envValue:  "",
			expected:  "us-east-1",
			shouldSet: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Save original value
			originalValue := os.Getenv("AWS_REGION")
			defer func() {
				if originalValue != "" {
					os.Setenv("AWS_REGION", originalValue)
				} else {
					os.Unsetenv("AWS_REGION")
				}
			}()

			// Set or unset AWS_REGION
			if tc.shouldSet {
				os.Setenv("AWS_REGION", tc.envValue)
			} else {
				os.Unsetenv("AWS_REGION")
			}

			// Test getRegionFromEnv
			result := getRegionFromEnv()
			require.Equal(t, tc.expected, result)
		})
	}
}

func TestApplyRegionPrefix_VariousRegions(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		modelID  string
		region   string
		expected string
	}{
		{
			name:     "us-east-1",
			modelID:  "amazon.nova-pro-v1:0",
			region:   "us-east-1",
			expected: "us.amazon.nova-pro-v1:0",
		},
		{
			name:     "eu-west-1",
			modelID:  "amazon.nova-lite-v1:0",
			region:   "eu-west-1",
			expected: "eu.amazon.nova-lite-v1:0",
		},
		{
			name:     "ap-southeast-1",
			modelID:  "amazon.nova-micro-v1:0",
			region:   "ap-southeast-1",
			expected: "ap.amazon.nova-micro-v1:0",
		},
		{
			name:     "ca-central-1",
			modelID:  "amazon.nova-premier-v1:0",
			region:   "ca-central-1",
			expected: "ca.amazon.nova-premier-v1:0",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := applyRegionPrefix(tc.modelID, tc.region)
			require.Equal(t, tc.expected, result)
		})
	}
}

func TestApplyRegionPrefix_NonRegionPrefixPattern(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		modelID  string
		region   string
		expected string
	}{
		{
			name:     "model ID starts with number",
			modelID:  "12.amazon.nova-pro-v1:0",
			region:   "us-east-1",
			expected: "us.12.amazon.nova-pro-v1:0",
		},
		{
			name:     "model ID starts with uppercase",
			modelID:  "US.amazon.nova-pro-v1:0",
			region:   "us-east-1",
			expected: "us.US.amazon.nova-pro-v1:0",
		},
		{
			name:     "model ID starts with mixed case",
			modelID:  "Ab.amazon.nova-pro-v1:0",
			region:   "us-east-1",
			expected: "us.Ab.amazon.nova-pro-v1:0",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := applyRegionPrefix(tc.modelID, tc.region)
			require.Equal(t, tc.expected, result)
		})
	}
}

func TestIsLowercaseLetters(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    string
		expected bool
	}{
		{"all lowercase", "us", true},
		{"all lowercase longer", "euwest", true},
		{"has uppercase", "Us", false},
		{"has number", "u1", false},
		{"has special char", "u-", false},
		{"empty string", "", true},
		{"single lowercase", "a", true},
		{"single uppercase", "A", false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := isLowercaseLetters(tc.input)
			require.Equal(t, tc.expected, result)
		})
	}
}
