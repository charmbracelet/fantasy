package main

import (
	"context"
	"fmt"

	"github.com/charmbracelet/crush/internal/ai"
	"github.com/charmbracelet/crush/internal/ai/providers"
)

func main() {
	provider := providers.NewOpenAIProvider(providers.WithOpenAIApiKey("$OPENAI_API_KEY"))
	model := provider.LanguageModel("gpt-4o")

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
