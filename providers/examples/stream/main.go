package main

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/charmbracelet/crush/internal/ai"
	"github.com/charmbracelet/crush/internal/ai/providers"
)

func main() {
	provider := providers.NewOpenAIProvider(providers.WithOpenAIApiKey("$OPENAI_API_KEY"))
	model := provider.LanguageModel("gpt-4o")

	stream, err := model.Stream(context.Background(), ai.Call{
		Prompt: ai.Prompt{
			ai.NewUserMessage("Whats the weather in pristina."),
		},
		Temperature: ai.FloatOption(0.7),
		Tools: []ai.Tool{
			ai.FunctionTool{
				Name:        "weather",
				Description: "Gets the weather for a location",
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"location": map[string]string{
							"type":        "string",
							"description": "the city",
						},
					},
					"required": []string{
						"location",
					},
				},
			},
		},
	})
	if err != nil {
		fmt.Println(err)
		return
	}

	for chunk := range stream {
		data, _ := json.Marshal(chunk)
		fmt.Println(string(data))
	}
}
