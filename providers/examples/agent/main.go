package main

import (
	"context"
	"fmt"
	"os"

	"github.com/charmbracelet/crush/internal/ai"
	"github.com/charmbracelet/crush/internal/ai/providers"
	"github.com/charmbracelet/crush/internal/llm/tools"
)

type weatherTool struct{}

// Info implements tools.BaseTool.
func (w *weatherTool) Info() tools.ToolInfo {
	return tools.ToolInfo{
		Name: "weather",
		Parameters: map[string]any{
			"location": map[string]string{
				"type":        "string",
				"description": "the city",
			},
		},
		Required: []string{"location"},
	}
}

// Name implements tools.BaseTool.
func (w *weatherTool) Name() string {
	return "weather"
}

// Run implements tools.BaseTool.
func (w *weatherTool) Run(ctx context.Context, params tools.ToolCall) (tools.ToolResponse, error) {
	return tools.NewTextResponse("40 C"), nil
}

func newWeatherTool() tools.BaseTool {
	return &weatherTool{}
}

func main() {
	provider := providers.NewOpenAIProvider(
		providers.WithOpenAIApiKey(os.Getenv("OPENAI_API_KEY")),
	)
	model := provider.LanguageModel("gpt-4o")

	agent := ai.NewAgent(
		model,
		ai.WithSystemPrompt("You are a helpful assistant"),
		ai.WithTools(newWeatherTool()),
	)

	result, _ := agent.Generate(context.Background(), ai.AgentCall{
		Prompt: "What's the weather in pristina",
	})

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
