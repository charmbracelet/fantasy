package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/kronk"
)

const modelURL = "unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"

func main() {
	if err := run(); err != nil {
		fmt.Printf("\nERROR: %s\n", err)
		os.Exit(1)
	}
}

func run() error {
	if len(os.Args) != 2 {
		return fmt.Errorf("usage: go run ./kronk/vision <image-path>")
	}

	image, err := os.ReadFile(os.Args[1])
	if err != nil {
		return fmt.Errorf("reading image: %w", err)
	}
	mediaType := http.DetectContentType(image)
	if !strings.HasPrefix(mediaType, "image/") {
		return fmt.Errorf("unsupported image type %q", mediaType)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	provider, err := kronk.NewProvider(
		kronk.WithLogger(kronk.FmtLogger),
		kronk.WithAutoTune(true),
	)
	if err != nil {
		return fmt.Errorf("creating provider: %w", err)
	}
	defer func() {
		fmt.Println("\nUnloading Kronk")
		if err := provider.Close(context.Background()); err != nil {
			fmt.Printf("failed to close provider: %v\n", err)
		}
	}()

	languageModel, err := provider.LanguageModel(ctx, modelURL)
	if err != nil {
		return fmt.Errorf("loading language model: %w", err)
	}

	agent := fantasy.NewAgent(languageModel)
	result, err := agent.Generate(ctx, fantasy.AgentCall{
		Prompt: "Describe this image briefly.",
		Files: []fantasy.FilePart{
			{
				Filename:  filepath.Base(os.Args[1]),
				Data:      image,
				MediaType: mediaType,
			},
		},
	})
	if err != nil {
		return fmt.Errorf("generating description: %w", err)
	}

	fmt.Println(result.Response.Content.Text())
	fmt.Printf("\nUsage: %s\n", result.TotalUsage)

	return nil
}
