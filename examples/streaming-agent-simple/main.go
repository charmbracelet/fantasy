// Package main provides a simple streaming agent example of using the fantasy AI SDK.
package main

import (
	"context"
	"fmt"
	"os"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openai"
)

func main() {
	// Check for API key
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("Please set OPENAI_API_KEY environment variable")
		os.Exit(1)
	}

	// Create provider and model
	provider, err := openai.New(openai.WithAPIKey(apiKey))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating OpenAI provider: %v\n", err)
		os.Exit(1)
	}

	ctx := context.Background()

	model, err := provider.LanguageModel(ctx, "gpt-4o-mini")
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	// Create echo tool using the new type-safe API
	type EchoInput struct {
		Message string `json:"message" description:"The message to echo back"`
	}

	echoTool := fantasy.NewAgentTool(
		"echo",
		"Echo back the provided message",
		func(_ context.Context, input EchoInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			return fantasy.NewTextResponse("Echo: " + input.Message), nil
		},
	)

	// Create streaming agent
	agent := fantasy.NewAgent(
		model,
		fantasy.WithSystemPrompt("You are a helpful assistant."),
		fantasy.WithTools(echoTool),
	)

	fmt.Println("Simple Streaming Agent Example")
	fmt.Println("==============================")
	fmt.Println()

	// Basic streaming with key callbacks
	streamCall := fantasy.AgentStreamCall{
		Prompt: "Please echo back 'Hello, streaming world!'",

		// Show real-time text as it streams
		OnTextDelta: func(_ string, text string) error {
			fmt.Print(text)
			return nil
		},

		// Show when tools are called
		OnToolCall: func(toolCall fantasy.ToolCallContent) error {
			fmt.Printf("\n[Tool: %s called]\n", toolCall.ToolName)
			return nil
		},

		// Show tool results
		OnToolResult: func(_ fantasy.ToolResultContent) error {
			fmt.Printf("[Tool result received]\n")
			return nil
		},

		// Show when each step completes
		OnStepFinish: func(step fantasy.StepResult) error {
			fmt.Printf("\n[Step completed: %s]\n", step.FinishReason)
			return nil
		},
	}

	fmt.Println("Assistant response:")
	result, err := agent.Stream(ctx, streamCall)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\n\nFinal result: %s\n", result.Response.Content.Text())
	fmt.Printf("Steps: %d, Total tokens: %d\n", len(result.Steps), result.TotalUsage.TotalTokens)
}
