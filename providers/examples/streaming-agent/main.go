package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/charmbracelet/crush/internal/ai"
	"github.com/charmbracelet/crush/internal/ai/providers"
	"github.com/charmbracelet/crush/internal/llm/tools"
)

// WeatherTool is a simple tool that simulates weather lookup
type WeatherTool struct{}

func (w *WeatherTool) Info() tools.ToolInfo {
	return tools.ToolInfo{
		Name:        "get_weather",
		Description: "Get the current weather for a specific location",
		Parameters: map[string]any{
			"location": map[string]any{
				"type":        "string",
				"description": "The city and country, e.g. 'London, UK'",
			},
			"unit": map[string]any{
				"type":        "string",
				"description": "Temperature unit (celsius or fahrenheit)",
				"enum":        []string{"celsius", "fahrenheit"},
				"default":     "celsius",
			},
		},
		Required: []string{"location"},
	}
}

func (w *WeatherTool) Run(ctx context.Context, params tools.ToolCall) (tools.ToolResponse, error) {
	// Simulate weather lookup with some fake data
	location := "Unknown"
	if strings.Contains(params.Input, "pristina") || strings.Contains(params.Input, "Pristina") {
		location = "Pristina, Kosovo"
	} else if strings.Contains(params.Input, "london") || strings.Contains(params.Input, "London") {
		location = "London, UK"
	} else if strings.Contains(params.Input, "new york") || strings.Contains(params.Input, "New York") {
		location = "New York, USA"
	}

	unit := "celsius"
	if strings.Contains(params.Input, "fahrenheit") {
		unit = "fahrenheit"
	}

	var temp string
	if unit == "fahrenheit" {
		temp = "72°F"
	} else {
		temp = "22°C"
	}

	weather := fmt.Sprintf("The current weather in %s is %s with partly cloudy skies and light winds.", location, temp)
	return tools.NewTextResponse(weather), nil
}

// CalculatorTool demonstrates a second tool for multi-tool scenarios
type CalculatorTool struct{}

func (c *CalculatorTool) Info() tools.ToolInfo {
	return tools.ToolInfo{
		Name:        "calculate",
		Description: "Perform basic mathematical calculations",
		Parameters: map[string]any{
			"expression": map[string]any{
				"type":        "string",
				"description": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')",
			},
		},
		Required: []string{"expression"},
	}
}

func (c *CalculatorTool) Run(ctx context.Context, params tools.ToolCall) (tools.ToolResponse, error) {
	// Simple calculator simulation
	expr := strings.ReplaceAll(params.Input, "\"", "")
	if strings.Contains(expr, "2 + 2") || strings.Contains(expr, "2+2") {
		return tools.NewTextResponse("2 + 2 = 4"), nil
	} else if strings.Contains(expr, "10 * 5") || strings.Contains(expr, "10*5") {
		return tools.NewTextResponse("10 * 5 = 50"), nil
	}
	return tools.NewTextResponse("I can calculate simple expressions like '2 + 2' or '10 * 5'"), nil
}

func main() {
	// Check for API key
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("❌ Please set OPENAI_API_KEY environment variable")
		fmt.Println("   export OPENAI_API_KEY=your_api_key_here")
		os.Exit(1)
	}

	fmt.Println("🚀 Streaming Agent Example")
	fmt.Println("==========================")
	fmt.Println()

	// Create OpenAI provider and model
	provider := providers.NewOpenAIProvider(
		providers.WithOpenAIApiKey(apiKey),
	)
	model := provider.LanguageModel("gpt-4o-mini") // Using mini for faster/cheaper responses

	// Create agent with tools
	agent := ai.NewAgent(
		model,
		ai.WithSystemPrompt("You are a helpful assistant that can check weather and do calculations. Be concise and friendly."),
		ai.WithTools(&WeatherTool{}, &CalculatorTool{}),
	)

	ctx := context.Background()

	// Demonstrate streaming with comprehensive callbacks
	fmt.Println("💬 Asking: \"What's the weather in Pristina and what's 2 + 2?\"")
	fmt.Println()

	// Track streaming events
	var stepCount int
	var textBuffer strings.Builder
	var reasoningBuffer strings.Builder

	// Create streaming call with all callbacks
	streamCall := ai.AgentStreamCall{
		Prompt: "What's the weather in Pristina and what's 2 + 2?",

		// Agent-level callbacks
		OnAgentStart: func() {
			fmt.Println("🎬 Agent started")
		},
		OnAgentFinish: func(result *ai.AgentResult) {
			fmt.Printf("🏁 Agent finished with %d steps, total tokens: %d\n", len(result.Steps), result.TotalUsage.TotalTokens)
		},
		OnStepStart: func(stepNumber int) {
			stepCount++
			fmt.Printf("📝 Step %d started\n", stepNumber+1)
		},
		OnStepFinish: func(stepResult ai.StepResult) {
			fmt.Printf("✅ Step completed (reason: %s, tokens: %d)\n", stepResult.FinishReason, stepResult.Usage.TotalTokens)
		},
		OnFinish: func(result *ai.AgentResult) {
			fmt.Printf("🎯 Final result ready with %d steps\n", len(result.Steps))
		},
		OnError: func(err error) {
			fmt.Printf("❌ Error: %v\n", err)
		},

		// Stream part callbacks
		OnWarnings: func(warnings []ai.CallWarning) {
			for _, warning := range warnings {
				fmt.Printf("⚠️  Warning: %s\n", warning.Message)
			}
		},
		OnTextStart: func(id string) {
			fmt.Print("💭 Assistant: ")
		},
		OnTextDelta: func(id, text string) {
			fmt.Print(text)
			textBuffer.WriteString(text)
		},
		OnTextEnd: func(id string) {
			fmt.Println()
		},
		OnReasoningStart: func(id string) {
			fmt.Print("🤔 Thinking: ")
		},
		OnReasoningDelta: func(id, text string) {
			reasoningBuffer.WriteString(text)
		},
		OnReasoningEnd: func(id string) {
			if reasoningBuffer.Len() > 0 {
				fmt.Printf("%s\n", reasoningBuffer.String())
				reasoningBuffer.Reset()
			}
		},
		OnToolInputStart: func(id, toolName string) {
			fmt.Printf("🔧 Calling tool: %s\n", toolName)
		},
		OnToolInputDelta: func(id, delta string) {
			// Could show tool input being built, but it's often noisy
		},
		OnToolInputEnd: func(id string) {
			// Tool input complete
		},
		OnToolCall: func(toolCall ai.ToolCallContent) {
			fmt.Printf("🛠️  Tool call: %s\n", toolCall.ToolName)
			fmt.Printf("   Input: %s\n", toolCall.Input)
		},
		OnToolResult: func(result ai.ToolResultContent) {
			fmt.Printf("🎯 Tool result from %s:\n", result.ToolName)
			switch output := result.Result.(type) {
			case ai.ToolResultOutputContentText:
				fmt.Printf("   %s\n", output.Text)
			case ai.ToolResultOutputContentError:
				fmt.Printf("   Error: %s\n", output.Error.Error())
			}
		},
		OnSource: func(source ai.SourceContent) {
			fmt.Printf("📚 Source: %s (%s)\n", source.Title, source.URL)
		},
		OnStreamFinish: func(usage ai.Usage, finishReason ai.FinishReason, providerMetadata ai.ProviderOptions) {
			fmt.Printf("📊 Stream finished (reason: %s, tokens: %d)\n", finishReason, usage.TotalTokens)
		},
		OnStreamError: func(err error) {
			fmt.Printf("💥 Stream error: %v\n", err)
		},
	}

	// Execute streaming agent
	result, err := agent.Stream(ctx, streamCall)
	if err != nil {
		fmt.Printf("❌ Agent failed: %v\n", err)
		os.Exit(1)
	}

	// Display final results
	fmt.Println()
	fmt.Println("📋 Final Summary")
	fmt.Println("================")
	fmt.Printf("Steps executed: %d\n", len(result.Steps))
	fmt.Printf("Total tokens used: %d (input: %d, output: %d)\n", 
		result.TotalUsage.TotalTokens, 
		result.TotalUsage.InputTokens, 
		result.TotalUsage.OutputTokens)
	
	if result.TotalUsage.ReasoningTokens > 0 {
		fmt.Printf("Reasoning tokens: %d\n", result.TotalUsage.ReasoningTokens)
	}

	fmt.Printf("Final response: %s\n", result.Response.Content.Text())

	// Show step details
	fmt.Println()
	fmt.Println("🔍 Step Details")
	fmt.Println("===============")
	for i, step := range result.Steps {
		fmt.Printf("Step %d:\n", i+1)
		fmt.Printf("  Finish reason: %s\n", step.FinishReason)
		fmt.Printf("  Content types: ")
		
		var contentTypes []string
		for _, content := range step.Content {
			contentTypes = append(contentTypes, string(content.GetType()))
		}
		fmt.Printf("%s\n", strings.Join(contentTypes, ", "))
		
		// Show tool calls and results
		toolCalls := step.Content.ToolCalls()
		if len(toolCalls) > 0 {
			fmt.Printf("  Tool calls: ")
			var toolNames []string
			for _, tc := range toolCalls {
				toolNames = append(toolNames, tc.ToolName)
			}
			fmt.Printf("%s\n", strings.Join(toolNames, ", "))
		}
		
		toolResults := step.Content.ToolResults()
		if len(toolResults) > 0 {
			fmt.Printf("  Tool results: %d\n", len(toolResults))
		}
		
		fmt.Printf("  Tokens: %d\n", step.Usage.TotalTokens)
		fmt.Println()
	}

	fmt.Println("✨ Example completed successfully!")
}