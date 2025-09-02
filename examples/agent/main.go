package main

import (
	"context"
	"fmt"
	"os"

	"github.com/charmbracelet/ai/ai"
	"github.com/charmbracelet/ai/providers/openai"
)

func main() {
	provider := openai.New(
		openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
	)
	model, err := provider.LanguageModel("gpt-4o")
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	// Create weather tool using the new type-safe API
	type WeatherInput struct {
		Location string `json:"location" description:"the city"`
	}

	weatherTool := ai.NewAgentTool(
		"weather",
		"Get weather information for a location",
		func(ctx context.Context, input WeatherInput, _ ai.ToolCall) (ai.ToolResponse, error) {
			return ai.NewTextResponse("40 C"), nil
		},
	)

	agent := ai.NewAgent(
		model,
		ai.WithSystemPrompt("You are a helpful assistant"),
		ai.WithTools(weatherTool),
	)

	result, err := agent.Generate(context.Background(), ai.AgentCall{
		Prompt: "What's the weather in pristina",
	})
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	fmt.Println("Steps: ", len(result.Steps))
	for _, s := range result.Steps {
		for _, c := range s.Content {
			if c.GetType() == ai.ContentTypeToolCall {
				tc, _ := ai.AsContentType[ai.ToolCallContent](c)
				fmt.Println("ToolCall: ", tc.ToolName)
			}
		}
	}

	fmt.Println("Final Response: ", result.Response.Content.Text())
	fmt.Println("Total Usage: ", result.TotalUsage)
}
