package main

// This is a basic example illustrting how to create an agent with a custom
// tool call.

import (
	"context"
	"fmt"
	"os"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openrouter"
)

func main() {
	// Choose your fave provider.
	provider, err := openrouter.New(openrouter.WithAPIKey(os.Getenv("OPENROUTER_API_KEY")))
	if err != nil {
		fmt.Fprintln(os.Stderr, "Whoops:", err)
		os.Exit(1)
	}

	ctx := context.Background()

	// Pick your fave model.
	model, err := provider.LanguageModel(ctx, "moonshotai/kimi-k2")
	if err != nil {
		fmt.Fprintln(os.Stderr, "Dang:", err)
		os.Exit(1)
	}

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
	agent := fantasy.NewAgent(model, fantasy.WithTools(cuteDogTool))

	// Put that agent to work!
	const prompt = "Find all the cute dogs in Silver Lake, Los Angeles."
	result, err := agent.Generate(ctx, fantasy.AgentCall{Prompt: prompt})
	if err != nil {
		fmt.Fprintln(os.Stderr, "Oof:", err)
		os.Exit(1)
	}
	fmt.Println(result.Response.Content.Text())
}
