package bedrock

import (
	"context"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"pgregory.net/rapid"
)

// Feature: amazon-nova-bedrock-support, Property 2: SDK Routing Correctness
// Validates: Requirements 1.7, 6.1, 6.2
func TestProperty_SDKRoutingCorrectness(t *testing.T) {
	t.Parallel()

	rapid.Check(t, func(t *rapid.T) {
		// Generate model IDs with different prefixes
		prefix := rapid.SampledFrom([]string{"anthropic.", "amazon.", "other.", ""}).Draw(t, "prefix")
		modelName := rapid.StringMatching(`[a-z0-9\-]+`).Draw(t, "modelName")
		modelID := prefix + modelName

		// Create provider
		provider, err := New()
		require.NoError(t, err)
		require.NotNil(t, provider)

		// Call LanguageModel
		ctx := context.Background()
		model, err := provider.LanguageModel(ctx, modelID)

		// Verify routing behavior based on prefix
		if strings.HasPrefix(modelID, "anthropic.") {
			// Should route to Anthropic SDK - will succeed or fail based on Anthropic SDK behavior
			// We just verify it doesn't return the "unsupported model prefix" error
			if err != nil {
				require.NotContains(t, err.Error(), "unsupported model prefix")
			}
		} else if strings.HasPrefix(modelID, "amazon.") {
			// Should route to Nova implementation
			// Should succeed in creating a model instance
			require.NoError(t, err, "Nova model creation should succeed for: %s", modelID)
			require.NotNil(t, model, "Model should not be nil for: %s", modelID)

			// Verify it's a valid language model
			require.Equal(t, Name, model.Provider())
			require.NotEmpty(t, model.Model())
		} else {
			// Should return unsupported prefix error
			require.Error(t, err)
			require.Contains(t, err.Error(), "unsupported model prefix")
			require.Nil(t, model)
		}
	})
}

// Unit tests for routing edge cases

func TestLanguageModel_EmptyModelID(t *testing.T) {
	t.Parallel()

	provider, err := New()
	require.NoError(t, err)
	require.NotNil(t, provider)

	ctx := context.Background()
	model, err := provider.LanguageModel(ctx, "")

	// Empty model ID should return unsupported prefix error
	require.Error(t, err)
	require.Contains(t, err.Error(), "unsupported model prefix")
	require.Nil(t, model)
}

func TestLanguageModel_ModelIDWithoutPrefix(t *testing.T) {
	t.Parallel()

	provider, err := New()
	require.NoError(t, err)
	require.NotNil(t, provider)

	ctx := context.Background()

	testCases := []struct {
		name    string
		modelID string
	}{
		{"no prefix", "claude-3-opus"},
		{"no prefix with version", "nova-pro-v1:0"},
		{"single word", "model"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			model, err := provider.LanguageModel(ctx, tc.modelID)

			// Model ID without proper prefix should return unsupported prefix error
			require.Error(t, err)
			require.Contains(t, err.Error(), "unsupported model prefix")
			require.Nil(t, model)
		})
	}
}

func TestLanguageModel_AnthropicModels_BackwardCompatibility(t *testing.T) {
	t.Parallel()

	provider, err := New()
	require.NoError(t, err)
	require.NotNil(t, provider)

	ctx := context.Background()

	// Test various Anthropic model IDs to ensure backward compatibility
	testCases := []struct {
		name    string
		modelID string
	}{
		{"claude-3-opus", "anthropic.claude-3-opus-20240229-v1:0"},
		{"claude-3-sonnet", "anthropic.claude-3-sonnet-20240229-v1:0"},
		{"claude-3-haiku", "anthropic.claude-3-haiku-20240307-v1:0"},
		{"claude-3-5-sonnet", "anthropic.claude-3-5-sonnet-20240620-v1:0"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			model, err := provider.LanguageModel(ctx, tc.modelID)

			// Should route to Anthropic SDK
			// The Anthropic SDK may return an error due to missing credentials,
			// but it should NOT be the "unsupported model prefix" error
			if err != nil {
				require.NotContains(t, err.Error(), "unsupported model prefix")
				require.NotContains(t, err.Error(), "Nova model support not yet implemented")
			}

			// If successful, verify it's a valid language model
			if model != nil {
				require.Equal(t, Name, model.Provider())
			}
		})
	}
}

func TestLanguageModel_AmazonModels_RoutesToNova(t *testing.T) {
	t.Parallel()

	provider, err := New()
	require.NoError(t, err)
	require.NotNil(t, provider)

	ctx := context.Background()

	// Test various Amazon Nova model IDs
	testCases := []struct {
		name    string
		modelID string
	}{
		{"nova-pro", "amazon.nova-pro-v1:0"},
		{"nova-lite", "amazon.nova-lite-v1:0"},
		{"nova-micro", "amazon.nova-micro-v1:0"},
		{"nova-premier", "amazon.nova-premier-v1:0"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			model, err := provider.LanguageModel(ctx, tc.modelID)

			// Should route to Nova implementation and succeed
			require.NoError(t, err, "Nova model creation should succeed for: %s", tc.modelID)
			require.NotNil(t, model, "Model should not be nil for: %s", tc.modelID)

			// Verify it's a valid language model
			require.Equal(t, Name, model.Provider())
			require.NotEmpty(t, model.Model())
		})
	}
}

func TestProvider_Name(t *testing.T) {
	t.Parallel()

	provider, err := New()
	require.NoError(t, err)
	require.NotNil(t, provider)

	// Verify provider name
	require.Equal(t, Name, provider.Name())
	require.Equal(t, "bedrock", provider.Name())
}

func TestNew_WithOptions(t *testing.T) {
	t.Parallel()

	// Test creating provider with various options
	t.Run("with skip auth", func(t *testing.T) {
		provider, err := New(WithSkipAuth(true))
		require.NoError(t, err)
		require.NotNil(t, provider)
	})

	t.Run("with headers", func(t *testing.T) {
		headers := map[string]string{
			"X-Custom-Header": "value",
		}
		provider, err := New(WithHeaders(headers))
		require.NoError(t, err)
		require.NotNil(t, provider)
	})

	t.Run("with multiple options", func(t *testing.T) {
		headers := map[string]string{
			"X-Custom-Header": "value",
		}
		provider, err := New(
			WithSkipAuth(true),
			WithHeaders(headers),
		)
		require.NoError(t, err)
		require.NotNil(t, provider)
	})
}

// Feature: amazon-nova-bedrock-support, Property 1: Model Instantiation Success
// Validates: Requirements 1.1
func TestProperty_ModelInstantiationSuccess(t *testing.T) {
	t.Parallel()

	rapid.Check(t, func(t *rapid.T) {
		// Generate valid Nova model identifiers
		modelVariant := rapid.SampledFrom([]string{
			"amazon.nova-pro-v1:0",
			"amazon.nova-lite-v1:0",
			"amazon.nova-micro-v1:0",
			"amazon.nova-premier-v1:0",
		}).Draw(t, "modelVariant")

		// Create provider
		provider, err := New()
		require.NoError(t, err)
		require.NotNil(t, provider)

		// Call LanguageModel with Nova model ID
		ctx := context.Background()
		model, err := provider.LanguageModel(ctx, modelVariant)

		// For any valid Nova model identifier, LanguageModel() should return
		// a non-nil language model instance without error
		require.NoError(t, err, "LanguageModel should succeed for valid Nova model: %s", modelVariant)
		require.NotNil(t, model, "LanguageModel should return non-nil model for: %s", modelVariant)

		// Verify the model implements the interface correctly
		require.Equal(t, Name, model.Provider(), "Provider should be 'bedrock'")
		require.NotEmpty(t, model.Model(), "Model ID should not be empty")

		// Verify the model ID has a region prefix applied
		modelID := model.Model()
		require.Contains(t, modelID, ".", "Model ID should contain region prefix")
		// The model ID should start with a 2-letter region code followed by a dot
		require.True(t, len(modelID) >= 3 && modelID[2] == '.',
			"Model ID should have region prefix format (e.g., 'us.amazon.nova-pro-v1:0')")
	})
}
