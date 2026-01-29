package main

// This is a basic example illustrating how to create an agent with a custom
// tool call.

import (
	"context"
	"fmt"
	"os"
	"time"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/kronk"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
)

const modelURL = "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q8_0.gguf"

func main() {
	if err := run(); err != nil {
		fmt.Printf("\nERROR: %s\n", err)
		os.Exit(1)
	}
}

func run() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	// Create the provider with optional logging.
	provider, err := kronk.New(
		kronk.WithName("kronk"),
		kronk.WithLogger(kronk.FmtLogger),
		kronk.WithModelConfig(model.Config{
			CacheTypeK: model.GGMLTypeQ8_0,
			CacheTypeV: model.GGMLTypeQ8_0,
			NSeqMax:    1,
		}),
	)
	if err != nil {
		return fmt.Errorf("unable to create provider: %w", err)
	}

	// Clean up when done.
	defer func() {
		fmt.Println("\nUnloading Kronk")
		if closer, ok := provider.(interface{ Close(context.Context) error }); ok {
			if err := closer.Close(context.Background()); err != nil {
				fmt.Printf("failed to close provider: %v\n", err)
			}
		}
	}()

	// Get a language model by providing the model URL.
	// The provider will download and initialize the model automatically.
	model, err := provider.LanguageModel(ctx, modelURL)
	if err != nil {
		return fmt.Errorf("unable to get language model: %w", err)
	}

	// -------------------------------------------------------------------------

	// Let's make a tool that fetches info about cute dogs. Here's a schema
	// for the tool's input.
	type cuteDogQuery struct {
		Location string `json:"location" description:"The location to search for cute dogs."`
	}

	// And here's the implementation of that tool.
	fetchCuteDogInfo := func(ctx context.Context, input cuteDogQuery, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
		if input.Location == "Silver Lake, Los Angeles" {
			return fantasy.NewTextResponse("Cute dogs are everywhere!"), nil
		}
		return fantasy.NewTextResponse("No cute dogs found."), nil
	}

	// Add the tool.
	cuteDogTool := fantasy.NewAgentTool("cute_dog_tool", "Provide up-to-date info on cute dogs.", fetchCuteDogInfo)

	// Equip your agent.
	agent := fantasy.NewAgent(model,
		fantasy.WithSystemPrompt("You are a moderately helpful, dog-centric assistant."),
		fantasy.WithTools(cuteDogTool),
		fantasy.WithMaxOutputTokens(2048),
		fantasy.WithTemperature(0.7),
		fantasy.WithTopP(0.8),
		fantasy.WithTopK(20),
	)

	// Put that agent to work!
	const prompt = "Find all the cute dogs in Silver Lake, Los Angeles. Use the cute dog tool."
	result, err := agent.Generate(ctx, fantasy.AgentCall{Prompt: prompt})
	if err != nil {
		return fmt.Errorf("agent generate failed: %w", err)
	}
	fmt.Println(result.Response.Content.Text())

	return nil
}
