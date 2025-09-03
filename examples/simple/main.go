package main

import (
	"context"
	"fmt"
	"os"

	"github.com/charmbracelet/ai/ai"
	"github.com/charmbracelet/ai/anthropic"
)

func main() {
	provider := anthropic.New(anthropic.WithAPIKey(os.Getenv("ANTHROPIC_API_KEY")))
	model, err := provider.LanguageModel("claude-sonnet-4-20250514")
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	response, err := model.Generate(context.Background(), ai.Call{
		Prompt: ai.Prompt{
			ai.NewUserMessage("Hello"),
		},
		Temperature: ai.FloatOption(0.7),
	})
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	fmt.Println("Assistant: ", response.Content.Text())
	fmt.Println("Usage:", response.Usage)
}
