package providertests

import (
	"context"
	"strconv"
	"strings"
	"testing"

	"github.com/charmbracelet/fantasy/ai"
	"github.com/charmbracelet/fantasy/anthropic"
	"github.com/charmbracelet/fantasy/google"
	"github.com/charmbracelet/fantasy/openai"
	"github.com/charmbracelet/fantasy/openrouter"
	_ "github.com/joho/godotenv/autoload"
	"github.com/stretchr/testify/require"
)

func TestSimple(t *testing.T) {
	for _, pair := range languageModelBuilders {
		t.Run(pair.name, func(t *testing.T) {
			r := newRecorder(t)

			languageModel, err := pair.builder(r)
			require.NoError(t, err, "failed to build language model")

			agent := ai.NewAgent(
				languageModel,
				ai.WithSystemPrompt("You are a helpful assistant"),
			)
			result, err := agent.Generate(t.Context(), ai.AgentCall{
				Prompt: "Say hi in Portuguese",
			})
			require.NoError(t, err, "failed to generate")

			option1 := "Oi"
			option2 := "Olá"
			got := result.Response.Content.Text()
			require.True(t, strings.Contains(got, option1) || strings.Contains(got, option2), "unexpected response: got %q, want %q or %q", got, option1, option2)
		})
	}
}

func TestTool(t *testing.T) {
	for _, pair := range languageModelBuilders {
		t.Run(pair.name, func(t *testing.T) {
			r := newRecorder(t)

			languageModel, err := pair.builder(r)
			require.NoError(t, err, "failed to build language model")

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
			require.NoError(t, err, "failed to generate")

			want1 := "Florence"
			want2 := "40"
			got := result.Response.Content.Text()
			require.True(t, strings.Contains(got, want1) && strings.Contains(got, want2), "unexpected response: got %q, want %q %q", got, want1, want2)
		})
	}
}

func TestThinking(t *testing.T) {
	for _, pair := range thinkingLanguageModelBuilders {
		t.Run(pair.name, func(t *testing.T) {
			r := newRecorder(t)

			languageModel, err := pair.builder(r)
			require.NoError(t, err, "failed to build language model")

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
				Prompt: "What's the weather in Florence, Italy?",
				ProviderOptions: ai.ProviderOptions{
					"anthropic": &anthropic.ProviderOptions{
						Thinking: &anthropic.ThinkingProviderOption{
							BudgetTokens: 10_000,
						},
					},
					"google": &google.ProviderOptions{
						ThinkingConfig: &google.ThinkingConfig{
							ThinkingBudget:  ai.IntOption(100),
							IncludeThoughts: ai.BoolOption(true),
						},
					},
					"openai": &openai.ProviderOptions{
						ReasoningEffort: openai.ReasoningEffortOption(openai.ReasoningEffortMedium),
					},
					"openrouter": &openrouter.ProviderOptions{
						Reasoning: &openrouter.ReasoningOptions{
							Effort: openrouter.ReasoningEffortOption(openrouter.ReasoningEffortHigh),
						},
					},
				},
			})
			require.NoError(t, err, "failed to generate")

			want1 := "Florence"
			want2 := "40"
			got := result.Response.Content.Text()
			require.True(t, strings.Contains(got, want1) && strings.Contains(got, want2), "unexpected response: got %q, want %q %q", got, want1, want2)

			testThinking(t, languageModel.Provider(), result.Steps)
		})
	}
}

func TestThinkingStreaming(t *testing.T) {
	for _, pair := range thinkingLanguageModelBuilders {
		t.Run(pair.name, func(t *testing.T) {
			r := newRecorder(t)

			languageModel, err := pair.builder(r)
			require.NoError(t, err, "failed to build language model")

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
			result, err := agent.Stream(t.Context(), ai.AgentStreamCall{
				Prompt: "What's the weather in Florence, Italy?",
				ProviderOptions: ai.ProviderOptions{
					"anthropic": &anthropic.ProviderOptions{
						Thinking: &anthropic.ThinkingProviderOption{
							BudgetTokens: 10_000,
						},
					},
					"google": &google.ProviderOptions{
						ThinkingConfig: &google.ThinkingConfig{
							ThinkingBudget:  ai.IntOption(100),
							IncludeThoughts: ai.BoolOption(true),
						},
					},
					"openai": &openai.ProviderOptions{
						ReasoningEffort: openai.ReasoningEffortOption(openai.ReasoningEffortMedium),
					},
				},
			})
			require.NoError(t, err, "failed to generate")

			want1 := "Florence"
			want2 := "40"
			got := result.Response.Content.Text()
			require.True(t, strings.Contains(got, want1) && strings.Contains(got, want2), "unexpected response: got %q, want %q %q", got, want1, want2)

			testThinking(t, languageModel.Provider(), result.Steps)
		})
	}
}

func TestStream(t *testing.T) {
	for _, pair := range languageModelBuilders {
		t.Run(pair.name, func(t *testing.T) {
			r := newRecorder(t)

			languageModel, err := pair.builder(r)
			require.NoError(t, err, "failed to build language model")

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
			require.NoError(t, err, "failed to stream")

			finalText := result.Response.Content.Text()
			require.NotEmpty(t, finalText, "expected non-empty response")

			require.True(t, strings.Contains(strings.ToLower(finalText), "uno") &&
				strings.Contains(strings.ToLower(finalText), "dos") &&
				strings.Contains(strings.ToLower(finalText), "tres"), "unexpected response: %q", finalText)

			require.Greater(t, textDeltaCount, 0, "expected at least one text delta callback")

			require.Greater(t, stepCount, 0, "expected at least one step finish callback")

			require.NotEmpty(t, collectedText.String(), "expected collected text from deltas to be non-empty")
		})
	}
}

func TestStreamWithTools(t *testing.T) {
	for _, pair := range languageModelBuilders {
		t.Run(pair.name, func(t *testing.T) {
			r := newRecorder(t)

			languageModel, err := pair.builder(r)
			require.NoError(t, err, "failed to build language model")

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
					require.Equal(t, "add", toolCall.ToolName, "unexpected tool name")
					return nil
				},
				OnToolResult: func(result ai.ToolResultContent) error {
					toolResultCount++
					return nil
				},
			}

			result, err := agent.Stream(t.Context(), streamCall)
			require.NoError(t, err, "failed to stream")

			finalText := result.Response.Content.Text()
			require.Contains(t, finalText, "42", "expected response to contain '42', got: %q", finalText)

			require.Greater(t, toolCallCount, 0, "expected at least one tool call")

			require.Greater(t, toolResultCount, 0, "expected at least one tool result")
		})
	}
}

func TestStreamWithMultipleTools(t *testing.T) {
	for _, pair := range languageModelBuilders {
		t.Run(pair.name, func(t *testing.T) {
			r := newRecorder(t)

			languageModel, err := pair.builder(r)
			require.NoError(t, err, "failed to build language model")

			type CalculatorInput struct {
				A int `json:"a" description:"first number"`
				B int `json:"b" description:"second number"`
			}

			addTool := ai.NewAgentTool(
				"add",
				"Add two numbers",
				func(ctx context.Context, input CalculatorInput, _ ai.ToolCall) (ai.ToolResponse, error) {
					result := input.A + input.B
					return ai.NewTextResponse(strings.TrimSpace(strconv.Itoa(result))), nil
				},
			)
			multiplyTool := ai.NewAgentTool(
				"multiply",
				"Multiply two numbers",
				func(ctx context.Context, input CalculatorInput, _ ai.ToolCall) (ai.ToolResponse, error) {
					result := input.A * input.B
					return ai.NewTextResponse(strings.TrimSpace(strconv.Itoa(result))), nil
				},
			)

			agent := ai.NewAgent(
				languageModel,
				ai.WithSystemPrompt("You are a helpful assistant. Always use both add and multiply at the same time."),
				ai.WithTools(addTool),
				ai.WithTools(multiplyTool),
			)

			toolCallCount := 0
			toolResultCount := 0
			var collectedText strings.Builder

			streamCall := ai.AgentStreamCall{
				Prompt: "Add and multiply the number 2 and 3",
				OnTextDelta: func(id, text string) error {
					collectedText.WriteString(text)
					return nil
				},
				OnToolCall: func(toolCall ai.ToolCallContent) error {
					toolCallCount++
					return nil
				},
				OnToolResult: func(result ai.ToolResultContent) error {
					toolResultCount++
					return nil
				},
			}

			result, err := agent.Stream(t.Context(), streamCall)
			require.NoError(t, err, "failed to stream")
			require.Equal(t, len(result.Steps), 2, "expected all tool calls in step 1")
			finalText := result.Response.Content.Text()
			require.Contains(t, finalText, "5", "expected response to contain '5', got: %q", finalText)
			require.Contains(t, finalText, "6", "expected response to contain '5', got: %q", finalText)

			require.Greater(t, toolCallCount, 0, "expected at least one tool call")

			require.Greater(t, toolResultCount, 0, "expected at least one tool result")
		})
	}
}
