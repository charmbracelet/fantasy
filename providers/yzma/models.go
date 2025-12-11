package yzma

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/hybridgroup/yzma/pkg/download"
)

type modelLocation struct {
	name string
	url  string
}

var (
	// supportedModels is a list of supported model IDs.
	supportedModels = []modelLocation{
		{"Qwen3VL-2B-Instruct-Q8_0.gguf", "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/Qwen3VL-2B-Instruct-Q8_0.gguf"},
		{"Qwen2.5-VL-3B-Instruct-Q8_0.gguf", "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-Q8_0.gguf"},
	}
)

func getModelURL(modelID string) (string, bool) {
	for _, m := range supportedModels {
		if m.name == modelID {
			return m.url, true
		}
	}
	return "", false
}

func ensureModelExists(ctx context.Context, modelPath string, modelsPath string) (string, error) {
	// Check if model file already exists
	defaultPath, err := checkModelExists(modelPath, modelsPath)
	if err != nil {
		return "", err
	}

	if defaultPath != "" {
		return defaultPath, nil
	}

	// is it a supported model we can download?
	url, ok := getModelURL(filepath.Base(modelPath))
	if !ok {
		return "", fmt.Errorf("model file not found: %s", modelPath)
	}

	if modelsPath == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		modelsPath = filepath.Join(home, "models")
	}
	if err := downloadModel(ctx, url, modelsPath); err != nil {
		return "", fmt.Errorf("failed to download model: %w", err)
	}

	return filepath.Join(modelsPath, filepath.Base(modelPath)), nil
}

func checkModelExists(modelPath, modelsPath string) (string, error) {
	// Check if model file already exists
	if _, err := os.Stat(modelPath); !os.IsNotExist(err) {
		return modelPath, nil
	}

	// check default models directory
	if modelsPath != "" {
		envPath := filepath.Join(modelsPath, filepath.Base(modelPath))
		if _, err := os.Stat(envPath); !os.IsNotExist(err) {
			return envPath, nil
		}
	}

	// check user home directory
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	modelsPath = filepath.Join(home, "models")
	defaultPath := filepath.Join(modelsPath, filepath.Base(modelPath))
	if _, err := os.Stat(defaultPath); !os.IsNotExist(err) {
		return defaultPath, nil
	}

	// could not find model
	return "", nil
}

func downloadModel(ctx context.Context, url string, destPath string) error {
	fmt.Printf("Downloading model %s to %s\n", url, destPath)
	return download.GetModelWithContext(ctx, url, destPath, download.DefaultProgressTracker())
}
