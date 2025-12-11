package yzma

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGetModelURL(t *testing.T) {
	t.Parallel()

	t.Run("returns URL for supported model", func(t *testing.T) {
		url, ok := getModelURL("Qwen2.5-VL-3B-Instruct-Q8_0.gguf")
		assert.True(t, ok)
		assert.Equal(t, "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-Q8_0.gguf", url)
	})

	t.Run("returns false for unsupported model", func(t *testing.T) {
		url, ok := getModelURL("nonexistent-model.gguf")
		assert.False(t, ok)
		assert.Empty(t, url)
	})

	t.Run("returns false for empty model name", func(t *testing.T) {
		url, ok := getModelURL("")
		assert.False(t, ok)
		assert.Empty(t, url)
	})

	t.Run("returns false for partial model name", func(t *testing.T) {
		url, ok := getModelURL("Qwen2.5-VL-3B")
		assert.False(t, ok)
		assert.Empty(t, url)
	})
}

func TestEnsureModelExists(t *testing.T) {
	t.Run("returns path when model file exists at given path", func(t *testing.T) {
		// Create a temporary model file
		tmpDir := t.TempDir()
		modelPath := filepath.Join(tmpDir, "test-model.gguf")
		err := os.WriteFile(modelPath, []byte("fake model content"), 0644)
		require.NoError(t, err)

		result, err := ensureModelExists(context.Background(), modelPath, "")
		require.NoError(t, err)
		assert.Equal(t, modelPath, result)
	})

	t.Run("returns path from YZMA_MODELS env when model exists there", func(t *testing.T) {
		// Create a temporary models directory
		tmpDir := t.TempDir()
		modelName := "env-test-model.gguf"
		modelPath := filepath.Join(tmpDir, modelName)
		err := os.WriteFile(modelPath, []byte("fake model content"), 0644)
		require.NoError(t, err)

		// Set YZMA_MODELS environment variable
		originalEnv := os.Getenv("YZMA_MODELS")
		t.Setenv("YZMA_MODELS", tmpDir)
		defer func() {
			if originalEnv != "" {
				os.Setenv("YZMA_MODELS", originalEnv)
			} else {
				os.Unsetenv("YZMA_MODELS")
			}
		}()

		// Request model by name (not full path)
		result, err := ensureModelExists(context.Background(), modelName, tmpDir)
		require.NoError(t, err)
		assert.Equal(t, modelPath, result)
	})

	t.Run("returns path from default models directory when model exists there", func(t *testing.T) {
		// Get home directory
		home, err := os.UserHomeDir()
		require.NoError(t, err)

		// Create default models directory if it doesn't exist
		defaultModelsDir := filepath.Join(home, "models")
		err = os.MkdirAll(defaultModelsDir, 0755)
		require.NoError(t, err)

		// Create a unique test model file
		modelName := "default-dir-test-model.gguf"
		modelPath := filepath.Join(defaultModelsDir, modelName)
		err = os.WriteFile(modelPath, []byte("fake model content"), 0644)
		require.NoError(t, err)
		defer os.Remove(modelPath)

		// Clear YZMA_MODELS to ensure we use default path
		originalEnv := os.Getenv("YZMA_MODELS")
		os.Unsetenv("YZMA_MODELS")
		defer func() {
			if originalEnv != "" {
				os.Setenv("YZMA_MODELS", originalEnv)
			}
		}()

		// Request model by name (not full path)
		result, err := ensureModelExists(context.Background(), modelName, "")
		require.NoError(t, err)
		assert.Equal(t, modelPath, result)
	})

	t.Run("returns error for unsupported model that doesn't exist", func(t *testing.T) {
		// Clear YZMA_MODELS to ensure consistent behavior
		originalEnv := os.Getenv("YZMA_MODELS")
		os.Unsetenv("YZMA_MODELS")
		defer func() {
			if originalEnv != "" {
				os.Setenv("YZMA_MODELS", originalEnv)
			}
		}()

		// Request a model that doesn't exist and isn't in supported list
		_, err := ensureModelExists(context.Background(), "nonexistent-unsupported-model.gguf", "")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "model file not found")
	})

	t.Run("extracts basename from full path for env lookup", func(t *testing.T) {
		// Create a temporary models directory
		tmpDir := t.TempDir()
		modelName := "basename-test-model.gguf"
		modelPath := filepath.Join(tmpDir, modelName)
		err := os.WriteFile(modelPath, []byte("fake model content"), 0644)
		require.NoError(t, err)

		// Set YZMA_MODELS environment variable
		t.Setenv("YZMA_MODELS", tmpDir)

		// Request model with a fake full path (file doesn't exist at this path)
		fakePath := filepath.Join("/nonexistent/path", modelName)
		result, err := ensureModelExists(context.Background(), fakePath, tmpDir)
		require.NoError(t, err)
		assert.Equal(t, modelPath, result)
	})
}

func TestSupportedModels(t *testing.T) {
	t.Parallel()

	t.Run("all supported models have valid URLs", func(t *testing.T) {
		for _, model := range supportedModels {
			assert.NotEmpty(t, model.name, "model name should not be empty")
			assert.NotEmpty(t, model.url, "model URL should not be empty")
			assert.Contains(t, model.url, "https://", "URL should be HTTPS")
			assert.Contains(t, model.url, model.name, "URL should contain model name")
		}
	})

	t.Run("supported models list is not empty", func(t *testing.T) {
		assert.NotEmpty(t, supportedModels)
	})
}
