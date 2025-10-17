package main

import (
	"context"
	"fmt"
	"os"

	"charm.land/fantasy"
	"charm.land/fantasy/openrouter"
)

func main() {
	provider := openrouter.New(
		openrouter.WithAPIKey(os.Getenv("OPENROUTER_API_KEY")),
	)
	model, err := provider.LanguageModel("moonshotai/kimi-k2-0905")
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	// Create weather tool using the new type-safe API
	type WeatherInput struct {
		Location string `json:"location" description:"the city"`
	}

	weatherTool := fantasy.NewAgentTool(
		"weather",
		"Get weather information for a location",
		func(ctx context.Context, input WeatherInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			return fantasy.NewTextResponse("40 C"), nil
		},
	)

	agent := fantasy.NewAgent(
		model,
		fantasy.WithSystemPrompt("You are a helpful assistant"),
		fantasy.WithTools(weatherTool),
	)

	result, err := agent.Generate(context.Background(), fantasy.AgentCall{
		Prompt: "What's the weather in pristina",
	})
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	fmt.Println("Steps: ", len(result.Steps))
	for _, s := range result.Steps {
		for _, c := range s.Content {
			if c.GetType() == fantasy.ContentTypeToolCall {
				tc, _ := fantasy.AsContentType[fantasy.ToolCallContent](c)
				fmt.Println("ToolCall: ", tc.ToolName)
			}
		}
	}

	fmt.Println("Final Response: ", result.Response.Content.Text())
	fmt.Println("Total Usage: ", result.TotalUsage)
}
