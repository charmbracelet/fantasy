package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openai"
)

func main() {
	provider := openai.New(openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")))
	model, err := provider.LanguageModel("gpt-4o")
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	stream, err := model.Stream(context.Background(), fantasy.Call{
		Prompt: fantasy.Prompt{
			fantasy.NewUserMessage("Whats the weather in pristina."),
		},
		Temperature: fantasy.Opt(0.7),
		Tools: []fantasy.Tool{
			fantasy.FunctionTool{
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
		os.Exit(1)
	}

	for chunk := range stream {
		data, err := json.Marshal(chunk)
		if err != nil {
			fmt.Println(err)
			continue
		}
		fmt.Println(string(data))
	}
}
