package providertests

import (
	"context"
	"strconv"
	"strings"
	"testing"

	"github.com/charmbracelet/fantasy/ai"
	"github.com/stretchr/testify/require"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

type builderFunc func(r *recorder.Recorder) (ai.LanguageModel, error)

type builderPair struct {
	name            string
	builder         builderFunc
	providerOptions ai.ProviderOptions
}

func testCommon(t *testing.T, pairs []builderPair) {
	for _, pair := range pairs {
		testSimple(t, pair)
	}
}

func testSimple(t *testing.T, pair builderPair) {
	checkResult := func(t *testing.T, result *ai.AgentResult) {
		option1 := "Oi"
		option2 := "Olá"
		got := result.Response.Content.Text()
		require.True(t, strings.Contains(got, option1) || strings.Contains(got, option2), "unexpected response: got %q, want %q or %q", got, option1, option2)
	}

	t.Run("simple "+pair.name, func(t *testing.T) {
		r := newRecorder(t)

		languageModel, err := pair.builder(r)
		require.NoError(t, err, "failed to build language model")

		agent := ai.NewAgent(
			languageModel,
			ai.WithSystemPrompt("You are a helpful assistant"),
		)
		result, err := agent.Generate(t.Context(), ai.AgentCall{
			Prompt:          "Say hi in Portuguese",
			ProviderOptions: pair.providerOptions,
			MaxOutputTokens: ai.IntOption(4000),
		})
		require.NoError(t, err, "failed to generate")
		checkResult(t, result)
	})
	t.Run("simple streaming "+pair.name, func(t *testing.T) {
		r := newRecorder(t)

		languageModel, err := pair.builder(r)
		require.NoError(t, err, "failed to build language model")

		agent := ai.NewAgent(
			languageModel,
			ai.WithSystemPrompt("You are a helpful assistant"),
		)
		result, err := agent.Stream(t.Context(), ai.AgentStreamCall{
			Prompt:          "Say hi in Portuguese",
			ProviderOptions: pair.providerOptions,
			MaxOutputTokens: ai.IntOption(4000),
		})
		require.NoError(t, err, "failed to generate")
		checkResult(t, result)
	})
}

func testTool(t *testing.T, pair builderPair) {
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
	checkResult := func(t *testing.T, result *ai.AgentResult) {
		require.Len(t, result.Steps, 2)

		var toolCalls []ai.ToolCallContent
		for _, content := range result.Steps[0].Content {
			if content.GetType() == ai.ContentTypeToolCall {
				toolCalls = append(toolCalls, content.(ai.ToolCallContent))
			}
		}
		require.Len(t, toolCalls, 1)
		require.Equal(t, toolCalls[0].ToolName, "weather")

		want1 := "Florence"
		want2 := "40"
		got := result.Response.Content.Text()
		require.True(t, strings.Contains(got, want1) && strings.Contains(got, want2), "unexpected response: got %q, want %q %q", got, want1, want2)
	}

	t.Run("tool "+pair.name, func(t *testing.T) {
		r := newRecorder(t)

		languageModel, err := pair.builder(r)
		require.NoError(t, err, "failed to build language model")

		agent := ai.NewAgent(
			languageModel,
			ai.WithSystemPrompt("You are a helpful assistant"),
			ai.WithTools(weatherTool),
		)
		result, err := agent.Generate(t.Context(), ai.AgentCall{
			Prompt:          "What's the weather in Florence,Italy?",
			ProviderOptions: pair.providerOptions,
			MaxOutputTokens: ai.IntOption(4000),
		})
		require.NoError(t, err, "failed to generate")
		checkResult(t, result)
	})
	t.Run("tool streaming "+pair.name, func(t *testing.T) {
		r := newRecorder(t)

		languageModel, err := pair.builder(r)
		require.NoError(t, err, "failed to build language model")

		agent := ai.NewAgent(
			languageModel,
			ai.WithSystemPrompt("You are a helpful assistant"),
			ai.WithTools(weatherTool),
		)
		result, err := agent.Stream(t.Context(), ai.AgentStreamCall{
			Prompt:          "What's the weather in Florence,Italy?",
			ProviderOptions: pair.providerOptions,
			MaxOutputTokens: ai.IntOption(4000),
		})
		require.NoError(t, err, "failed to generate")
		checkResult(t, result)
	})
}

func testMultiTool(t *testing.T, pair builderPair) {
	type WeatherInput struct {
		Location string `json:"location" description:"the city"`
	}

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
	checkResult := func(t *testing.T, result *ai.AgentResult) {
		require.Len(t, result.Steps, 2)

		var toolCalls []ai.ToolCallContent
		for _, content := range result.Steps[0].Content {
			if content.GetType() == ai.ContentTypeToolCall {
				toolCalls = append(toolCalls, content.(ai.ToolCallContent))
			}
		}
		require.Len(t, toolCalls, 2)

		finalText := result.Response.Content.Text()
		require.Contains(t, finalText, "5", "expected response to contain '5', got: %q", finalText)
		require.Contains(t, finalText, "6", "expected response to contain '6', got: %q", finalText)
	}

	t.Run("multi tool "+pair.name, func(t *testing.T) {
		r := newRecorder(t)

		languageModel, err := pair.builder(r)
		require.NoError(t, err, "failed to build language model")

		agent := ai.NewAgent(
			languageModel,
			ai.WithSystemPrompt("You are a helpful assistant. Always use both add and multiply at the same time."),
			ai.WithTools(addTool),
			ai.WithTools(multiplyTool),
		)
		result, err := agent.Generate(t.Context(), ai.AgentCall{
			Prompt:          "Add and multiply the number 2 and 3",
			ProviderOptions: pair.providerOptions,
			MaxOutputTokens: ai.IntOption(4000),
		})
		require.NoError(t, err, "failed to generate")
		checkResult(t, result)
	})
	t.Run("multi tool streaming "+pair.name, func(t *testing.T) {
		r := newRecorder(t)

		languageModel, err := pair.builder(r)
		require.NoError(t, err, "failed to build language model")

		agent := ai.NewAgent(
			languageModel,
			ai.WithSystemPrompt("You are a helpful assistant. Always use both add and multiply at the same time."),
			ai.WithTools(addTool),
			ai.WithTools(multiplyTool),
		)
		result, err := agent.Stream(t.Context(), ai.AgentStreamCall{
			Prompt:          "Add and multiply the number 2 and 3",
			ProviderOptions: pair.providerOptions,
			MaxOutputTokens: ai.IntOption(4000),
		})
		require.NoError(t, err, "failed to generate")
		checkResult(t, result)
	})
}
