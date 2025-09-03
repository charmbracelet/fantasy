package main

import (
	"context"
	"fmt"
	"os"

	"github.com/charmbracelet/ai/ai"
	"github.com/charmbracelet/ai/openai"
)

func main() {
	// Check for API key
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("Please set OPENAI_API_KEY environment variable")
		os.Exit(1)
	}

	// Create provider and model
	provider := openai.New(
		openai.WithAPIKey(apiKey),
	)
	model, err := provider.LanguageModel("gpt-4o-mini")
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	// Create echo tool using the new type-safe API
	type EchoInput struct {
		Message string `json:"message" description:"The message to echo back"`
	}

	echoTool := ai.NewAgentTool(
		"echo",
		"Echo back the provided message",
		func(ctx context.Context, input EchoInput, _ ai.ToolCall) (ai.ToolResponse, error) {
			return ai.NewTextResponse("Echo: " + input.Message), nil
		},
	)

	// Create streaming agent
	agent := ai.NewAgent(
		model,
		ai.WithSystemPrompt("You are a helpful assistant."),
		ai.WithTools(echoTool),
	)

	ctx := context.Background()

	fmt.Println("Simple Streaming Agent Example")
	fmt.Println("==============================")
	fmt.Println()

	// Basic streaming with key callbacks
	streamCall := ai.AgentStreamCall{
		Prompt: "Please echo back 'Hello, streaming world!'",

		// Show real-time text as it streams
		OnTextDelta: func(id, text string) error {
			fmt.Print(text)
			return nil
		},

		// Show when tools are called
		OnToolCall: func(toolCall ai.ToolCallContent) error {
			fmt.Printf("\n[Tool: %s called]\n", toolCall.ToolName)
			return nil
		},

		// Show tool results
		OnToolResult: func(result ai.ToolResultContent) error {
			fmt.Printf("[Tool result received]\n")
			return nil
		},

		// Show when each step completes
		OnStepFinish: func(step ai.StepResult) error {
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
