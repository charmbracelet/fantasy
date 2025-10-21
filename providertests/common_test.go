package providertests

import (
	"context"
	"os"
	"strconv"
	"strings"
	"testing"

	"charm.land/fantasy"
	"github.com/joho/godotenv"
	"github.com/stretchr/testify/require"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func init() {
	if _, err := os.Stat(".env"); err == nil {
		godotenv.Load(".env")
	} else {
		godotenv.Load(".env.sample")
	}
}

type testModel struct {
	name      string
	model     string
	reasoning bool
}

type builderFunc func(r *recorder.Recorder) (fantasy.LanguageModel, error)

type builderPair struct {
	name            string
	builder         builderFunc
	providerOptions fantasy.ProviderOptions
	prepareStep     fantasy.PrepareStepFunction
}

func testCommon(t *testing.T, pairs []builderPair) {
	for _, pair := range pairs {
		t.Run(pair.name, func(t *testing.T) {
			testSimple(t, pair)
			testTool(t, pair)
			testMultiTool(t, pair)
		})
	}
}

func testSimple(t *testing.T, pair builderPair) {
	checkResult := func(t *testing.T, result *fantasy.AgentResult) {
		options := []string{"Oi", "oi", "Olá", "olá"}
		got := result.Response.Content.Text()
		require.True(t, containsAny(got, options...), "unexpected response: got %q, want any of: %q", got, options)
	}

	t.Run("simple", func(t *testing.T) {
		r := newRecorder(t)

		languageModel, err := pair.builder(r)
		require.NoError(t, err, "failed to build language model")

		agent := fantasy.NewAgent(
			languageModel,
			fantasy.WithSystemPrompt("You are a helpful assistant"),
		)
		result, err := agent.Generate(t.Context(), fantasy.AgentCall{
			Prompt:          "Say hi in Portuguese",
			ProviderOptions: pair.providerOptions,
			MaxOutputTokens: fantasy.Opt(int64(4000)),
			PrepareStep:     pair.prepareStep,
		})
		require.NoError(t, err, "failed to generate")
		checkResult(t, result)
	})
	t.Run("simple streaming", func(t *testing.T) {
		r := newRecorder(t)

		languageModel, err := pair.builder(r)
		require.NoError(t, err, "failed to build language model")

		agent := fantasy.NewAgent(
			languageModel,
			fantasy.WithSystemPrompt("You are a helpful assistant"),
		)
		result, err := agent.Stream(t.Context(), fantasy.AgentStreamCall{
			Prompt:          "Say hi in Portuguese",
			ProviderOptions: pair.providerOptions,
			MaxOutputTokens: fantasy.Opt(int64(4000)),
			PrepareStep:     pair.prepareStep,
		})
		require.NoError(t, err, "failed to generate")
		checkResult(t, result)
	})
}

func testTool(t *testing.T, pair builderPair) {
	type WeatherInput struct {
		Location string `json:"location" description:"the city"`
	}

	weatherTool := fantasy.NewAgentTool(
		"weather",
		"Get weather information for a location",
		func(ctx context.Context, input WeatherInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			return fantasy.NewTextResponse("40 C"), nil
		},
	)
	checkResult := func(t *testing.T, result *fantasy.AgentResult) {
		require.GreaterOrEqual(t, len(result.Steps), 2)

		var toolCalls []fantasy.ToolCallContent
		for _, content := range result.Steps[0].Content {
			if content.GetType() == fantasy.ContentTypeToolCall {
				toolCalls = append(toolCalls, content.(fantasy.ToolCallContent))
			}
		}
		for _, tc := range toolCalls {
			require.False(t, tc.Invalid)
		}
		require.Len(t, toolCalls, 1)
		require.Equal(t, toolCalls[0].ToolName, "weather")

		want1 := "Florence"
		want2 := "40"
		got := result.Response.Content.Text()
		require.True(t, strings.Contains(got, want1) && strings.Contains(got, want2), "unexpected response: got %q, want %q %q", got, want1, want2)
	}

	t.Run("tool", func(t *testing.T) {
		r := newRecorder(t)

		languageModel, err := pair.builder(r)
		require.NoError(t, err, "failed to build language model")

		agent := fantasy.NewAgent(
			languageModel,
			fantasy.WithSystemPrompt("You are a helpful assistant"),
			fantasy.WithTools(weatherTool),
		)
		result, err := agent.Generate(t.Context(), fantasy.AgentCall{
			Prompt:          "What's the weather in Florence,Italy?",
			ProviderOptions: pair.providerOptions,
			MaxOutputTokens: fantasy.Opt(int64(4000)),
			PrepareStep:     pair.prepareStep,
		})
		require.NoError(t, err, "failed to generate")
		checkResult(t, result)
	})
	t.Run("tool streaming", func(t *testing.T) {
		r := newRecorder(t)

		languageModel, err := pair.builder(r)
		require.NoError(t, err, "failed to build language model")

		agent := fantasy.NewAgent(
			languageModel,
			fantasy.WithSystemPrompt("You are a helpful assistant"),
			fantasy.WithTools(weatherTool),
		)
		result, err := agent.Stream(t.Context(), fantasy.AgentStreamCall{
			Prompt:          "What's the weather in Florence,Italy?",
			ProviderOptions: pair.providerOptions,
			MaxOutputTokens: fantasy.Opt(int64(4000)),
			PrepareStep:     pair.prepareStep,
		})
		require.NoError(t, err, "failed to generate")
		checkResult(t, result)
	})
}

func testMultiTool(t *testing.T, pair builderPair) {
	// Apparently, Azure and Vertex+Anthropic do not support multi-tools calls at all?
	if strings.Contains(pair.name, "azure") {
		t.Skip("skipping multi-tool tests for azure as it does not support parallel multi-tool calls")
	}
	if strings.Contains(pair.name, "vertex") && strings.Contains(pair.name, "claude") {
		t.Skip("skipping multi-tool tests for vertex claude as it does not support parallel multi-tool calls")
	}
	if strings.Contains(pair.name, "bedrock") && strings.Contains(pair.name, "claude") {
		t.Skip("skipping multi-tool tests for bedrock claude as it does not support parallel multi-tool calls")
	}
	if strings.Contains(pair.name, "openai") && strings.Contains(pair.name, "o4-mini") {
		t.Skip("skipping multi-tool tests for openai o4-mini it for some reason is not doing parallel tool calls even if asked")
	}

	type CalculatorInput struct {
		A int `json:"a" description:"first number"`
		B int `json:"b" description:"second number"`
	}

	addTool := fantasy.NewAgentTool(
		"add",
		"Add two numbers",
		func(ctx context.Context, input CalculatorInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			result := input.A + input.B
			return fantasy.NewTextResponse(strings.TrimSpace(strconv.Itoa(result))), nil
		},
	)
	multiplyTool := fantasy.NewAgentTool(
		"multiply",
		"Multiply two numbers",
		func(ctx context.Context, input CalculatorInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			result := input.A * input.B
			return fantasy.NewTextResponse(strings.TrimSpace(strconv.Itoa(result))), nil
		},
	)
	checkResult := func(t *testing.T, result *fantasy.AgentResult) {
		require.Len(t, result.Steps, 2)

		var toolCalls []fantasy.ToolCallContent
		for _, content := range result.Steps[0].Content {
			if content.GetType() == fantasy.ContentTypeToolCall {
				toolCalls = append(toolCalls, content.(fantasy.ToolCallContent))
			}
		}
		for _, tc := range toolCalls {
			require.False(t, tc.Invalid)
		}
		require.Len(t, toolCalls, 2)

		finalText := result.Response.Content.Text()
		require.Contains(t, finalText, "5", "expected response to contain '5', got: %q", finalText)
		require.Contains(t, finalText, "6", "expected response to contain '6', got: %q", finalText)
	}

	t.Run("multi tool", func(t *testing.T) {
		r := newRecorder(t)

		languageModel, err := pair.builder(r)
		require.NoError(t, err, "failed to build language model")

		agent := fantasy.NewAgent(
			languageModel,
			fantasy.WithSystemPrompt("You are a helpful assistant. CRITICAL: Always use both add and multiply at the same time ALWAYS."),
			fantasy.WithTools(addTool),
			fantasy.WithTools(multiplyTool),
		)
		result, err := agent.Generate(t.Context(), fantasy.AgentCall{
			Prompt:          "Add and multiply the number 2 and 3",
			ProviderOptions: pair.providerOptions,
			MaxOutputTokens: fantasy.Opt(int64(4000)),
			PrepareStep:     pair.prepareStep,
		})
		require.NoError(t, err, "failed to generate")
		checkResult(t, result)
	})
	t.Run("multi tool streaming", func(t *testing.T) {
		r := newRecorder(t)

		languageModel, err := pair.builder(r)
		require.NoError(t, err, "failed to build language model")

		agent := fantasy.NewAgent(
			languageModel,
			fantasy.WithSystemPrompt("You are a helpful assistant. Always use both add and multiply at the same time."),
			fantasy.WithTools(addTool),
			fantasy.WithTools(multiplyTool),
		)
		result, err := agent.Stream(t.Context(), fantasy.AgentStreamCall{
			Prompt:          "Add and multiply the number 2 and 3",
			ProviderOptions: pair.providerOptions,
			MaxOutputTokens: fantasy.Opt(int64(4000)),
			PrepareStep:     pair.prepareStep,
		})
		require.NoError(t, err, "failed to generate")
		checkResult(t, result)
	})
}

func testThinking(t *testing.T, pairs []builderPair, thinkChecks func(*testing.T, *fantasy.AgentResult)) {
	for _, pair := range pairs {
		t.Run(pair.name, func(t *testing.T) {
			t.Run("thinking", func(t *testing.T) {
				r := newRecorder(t)

				languageModel, err := pair.builder(r)
				require.NoError(t, err, "failed to build language model")

				type WeatherInput struct {
					Location string `json:"location" description:"the city"`
				}

				weatherTool := fantasy.NewAgentTool(
					"weather",
					"Get weather information for a location",
					func(ctx context.Context, input WeatherInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
						return fantasy.NewTextResponse("40 C"), nil
					},
				)

				agent := fantasy.NewAgent(
					languageModel,
					fantasy.WithSystemPrompt("You are a helpful assistant"),
					fantasy.WithTools(weatherTool),
				)
				result, err := agent.Generate(t.Context(), fantasy.AgentCall{
					Prompt:          "What's the weather in Florence, Italy?",
					ProviderOptions: pair.providerOptions,
					PrepareStep:     pair.prepareStep,
				})
				require.NoError(t, err, "failed to generate")

				want1 := "Florence"
				want2 := "40"
				got := result.Response.Content.Text()
				require.True(t, strings.Contains(got, want1) && strings.Contains(got, want2), "unexpected response: got %q, want %q %q", got, want1, want2)

				thinkChecks(t, result)
			})
			t.Run("thinking-streaming", func(t *testing.T) {
				r := newRecorder(t)

				languageModel, err := pair.builder(r)
				require.NoError(t, err, "failed to build language model")

				type WeatherInput struct {
					Location string `json:"location" description:"the city"`
				}

				weatherTool := fantasy.NewAgentTool(
					"weather",
					"Get weather information for a location",
					func(ctx context.Context, input WeatherInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
						return fantasy.NewTextResponse("40 C"), nil
					},
				)

				agent := fantasy.NewAgent(
					languageModel,
					fantasy.WithSystemPrompt("You are a helpful assistant"),
					fantasy.WithTools(weatherTool),
				)
				result, err := agent.Stream(t.Context(), fantasy.AgentStreamCall{
					Prompt:          "What's the weather in Florence, Italy?",
					ProviderOptions: pair.providerOptions,
					PrepareStep:     pair.prepareStep,
				})
				require.NoError(t, err, "failed to generate")

				want1 := "Florence"
				want2 := "40"
				got := result.Response.Content.Text()
				require.True(t, strings.Contains(got, want1) && strings.Contains(got, want2), "unexpected response: got %q, want %q %q", got, want1, want2)

				thinkChecks(t, result)
			})
		})
	}
}

func containsAny(s string, subs ...string) bool {
	for _, sub := range subs {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}
