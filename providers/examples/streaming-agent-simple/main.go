package main

import (
	"context"
	"fmt"
	"os"

	"github.com/charmbracelet/crush/internal/ai"
	"github.com/charmbracelet/crush/internal/ai/providers"
	"github.com/charmbracelet/crush/internal/llm/tools"
)

// Simple echo tool for demonstration
type EchoTool struct{}

func (e *EchoTool) Info() tools.ToolInfo {
	return tools.ToolInfo{
		Name:        "echo",
		Description: "Echo back the provided message",
		Parameters: map[string]any{
			"message": map[string]any{
				"type":        "string",
				"description": "The message to echo back",
			},
		},
		Required: []string{"message"},
	}
}

func (e *EchoTool) Run(ctx context.Context, params tools.ToolCall) (tools.ToolResponse, error) {
	return tools.NewTextResponse("Echo: " + params.Input), nil
}

func main() {
	// Check for API key
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("Please set OPENAI_API_KEY environment variable")
		os.Exit(1)
	}

	// Create provider and model
	provider := providers.NewOpenAIProvider(
		providers.WithOpenAIApiKey(apiKey),
	)
	model := provider.LanguageModel("gpt-4o-mini")

	// Create streaming agent
	agent := ai.NewAgent(
		model,
		ai.WithSystemPrompt("You are a helpful assistant."),
		ai.WithTools(&EchoTool{}),
	)

	ctx := context.Background()

	fmt.Println("Simple Streaming Agent Example")
	fmt.Println("==============================")
	fmt.Println()

	// Basic streaming with key callbacks
	streamCall := ai.AgentStreamCall{
		Prompt: "Please echo back 'Hello, streaming world!'",
		
		// Show real-time text as it streams
		OnTextDelta: func(id, text string) {
			fmt.Print(text)
		},
		
		// Show when tools are called
		OnToolCall: func(toolCall ai.ToolCallContent) {
			fmt.Printf("\n[Tool: %s called]\n", toolCall.ToolName)
		},
		
		// Show tool results
		OnToolResult: func(result ai.ToolResultContent) {
			fmt.Printf("[Tool result received]\n")
		},
		
		// Show when each step completes
		OnStepFinish: func(step ai.StepResult) {
			fmt.Printf("\n[Step completed: %s]\n", step.FinishReason)
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