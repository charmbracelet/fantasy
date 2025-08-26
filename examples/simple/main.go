package main

import (
	"context"
	"fmt"
	"os"

	"github.com/charmbracelet/crush/internal/ai"
	"github.com/charmbracelet/crush/internal/ai/providers"
)

func main() {
	provider := providers.NewAnthropicProvider(providers.WithAnthropicAPIKey(os.Getenv("ANTHROPIC_API_KEY")))
	model, err := provider.LanguageModel("claude-sonnet-4-20250514")
	if err != nil {
		fmt.Println(err)
		return
	}

	response, err := model.Generate(context.Background(), ai.Call{
		Prompt: ai.Prompt{
			ai.NewUserMessage("Hello"),
		},
		Temperature: ai.FloatOption(0.7),
	})
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println("Assistant: ", response.Content.Text())
	fmt.Println("Usage:", response.Usage)
}
