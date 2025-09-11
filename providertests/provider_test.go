package providertests

import (
	"context"
	"strconv"
	"strings"
	"testing"

	"github.com/charmbracelet/fantasy/ai"
	_ "github.com/joho/godotenv/autoload"
)

func TestSimple(t *testing.T) {
	for _, pair := range languageModelBuilders {
		t.Run(pair.name, func(t *testing.T) {
			r := newRecorder(t)

			languageModel, err := pair.builder(r)
			if err != nil {
				t.Fatalf("failed to build language model: %v", err)
			}

			agent := ai.NewAgent(
				languageModel,
				ai.WithSystemPrompt("You are a helpful assistant"),
			)
			result, err := agent.Generate(t.Context(), ai.AgentCall{
				Prompt: "Say hi in Portuguese",
			})
			if err != nil {
				t.Fatalf("failed to generate: %v", err)
			}

			option1 := "Oi"
			option2 := "Olá"
			got := result.Response.Content.Text()
			if !strings.Contains(got, option1) && !strings.Contains(got, option2) {
				t.Fatalf("unexpected response: got %q, want %q or %q", got, option1, option2)
			}
		})
	}
}

func TestTool(t *testing.T) {
	for _, pair := range languageModelBuilders {
		t.Run(pair.name, func(t *testing.T) {
			r := newRecorder(t)

			languageModel, err := pair.builder(r)
			if err != nil {
				t.Fatalf("failed to build language model: %v", err)
			}

			type WeatherInput struct {
				Location string `json:"location" description:"the city"`
			}

			weatherTool := ai.NewAgentTool(
				"weather",
				"Get weather information for a location",
				func(ctx context.Context, input WeatherInput, _ ai.ToolCall) (ai.ToolResponse, error) {
					return ai.NewTextResponse("40 C"), nil
				},
			)

			agent := ai.NewAgent(
				languageModel,
				ai.WithSystemPrompt("You are a helpful assistant"),
				ai.WithTools(weatherTool),
			)
			result, err := agent.Generate(t.Context(), ai.AgentCall{
				Prompt: "What's the weather in Florence?",
			})
			if err != nil {
				t.Fatalf("failed to generate: %v", err)
			}

			want1 := "Florence"
			want2 := "40"
			got := result.Response.Content.Text()
			if !strings.Contains(got, want1) || !strings.Contains(got, want2) {
				t.Fatalf("unexpected response: got %q, want %q %q", got, want1, want2)
			}
		})
	}
}

func TestStream(t *testing.T) {
	for _, pair := range languageModelBuilders {
		t.Run(pair.name, func(t *testing.T) {
			r := newRecorder(t)

			languageModel, err := pair.builder(r)
			if err != nil {
				t.Fatalf("failed to build language model: %v", err)
			}

			agent := ai.NewAgent(
				languageModel,
				ai.WithSystemPrompt("You are a helpful assistant"),
			)

			var collectedText strings.Builder
			textDeltaCount := 0
			stepCount := 0

			streamCall := ai.AgentStreamCall{
				Prompt: "Count from 1 to 3 in Spanish",
				OnTextDelta: func(id, text string) error {
					textDeltaCount++
					collectedText.WriteString(text)
					return nil
				},
				OnStepFinish: func(step ai.StepResult) error {
					stepCount++
					return nil
				},
			}

			result, err := agent.Stream(t.Context(), streamCall)
			if err != nil {
				t.Fatalf("failed to stream: %v", err)
			}

			finalText := result.Response.Content.Text()
			if finalText == "" {
				t.Fatal("expected non-empty response")
			}

			if !strings.Contains(strings.ToLower(finalText), "uno") ||
				!strings.Contains(strings.ToLower(finalText), "dos") ||
				!strings.Contains(strings.ToLower(finalText), "tres") {
				t.Fatalf("unexpected response: %q", finalText)
			}

			if textDeltaCount == 0 {
				t.Fatal("expected at least one text delta callback")
			}

			if stepCount == 0 {
				t.Fatal("expected at least one step finish callback")
			}

			if collectedText.String() == "" {
				t.Fatal("expected collected text from deltas to be non-empty")
			}
		})
	}
}

func TestStreamWithTools(t *testing.T) {
	for _, pair := range languageModelBuilders {
		t.Run(pair.name, func(t *testing.T) {
			r := newRecorder(t)

			languageModel, err := pair.builder(r)
			if err != nil {
				t.Fatalf("failed to build language model: %v", err)
			}

			type CalculatorInput struct {
				A int `json:"a" description:"first number"`
				B int `json:"b" description:"second number"`
			}

			calculatorTool := ai.NewAgentTool(
				"add",
				"Add two numbers",
				func(ctx context.Context, input CalculatorInput, _ ai.ToolCall) (ai.ToolResponse, error) {
					result := input.A + input.B
					return ai.NewTextResponse(strings.TrimSpace(strconv.Itoa(result))), nil
				},
			)

			agent := ai.NewAgent(
				languageModel,
				ai.WithSystemPrompt("You are a helpful assistant. Use the add tool to perform calculations."),
				ai.WithTools(calculatorTool),
			)

			toolCallCount := 0
			toolResultCount := 0
			var collectedText strings.Builder

			streamCall := ai.AgentStreamCall{
				Prompt: "What is 15 + 27?",
				OnTextDelta: func(id, text string) error {
					collectedText.WriteString(text)
					return nil
				},
				OnToolCall: func(toolCall ai.ToolCallContent) error {
					toolCallCount++
					if toolCall.ToolName != "add" {
						t.Errorf("unexpected tool name: %s", toolCall.ToolName)
					}
					return nil
				},
				OnToolResult: func(result ai.ToolResultContent) error {
					toolResultCount++
					return nil
				},
			}

			result, err := agent.Stream(t.Context(), streamCall)
			if err != nil {
				t.Fatalf("failed to stream: %v", err)
			}

			finalText := result.Response.Content.Text()
			if !strings.Contains(finalText, "42") {
				t.Fatalf("expected response to contain '42', got: %q", finalText)
			}

			if toolCallCount == 0 {
				t.Fatal("expected at least one tool call")
			}

			if toolResultCount == 0 {
				t.Fatal("expected at least one tool result")
			}
		})
	}
}
