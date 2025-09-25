package providertests

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/charmbracelet/fantasy/ai"
	"github.com/charmbracelet/fantasy/openrouter"
	"github.com/stretchr/testify/require"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func TestOpenRouterCommon(t *testing.T) {
	testCommon(t, []builderPair{
		{"kimi-k2", builderOpenRouterKimiK2, nil},
		{"grok-code-fast-1", builderOpenRouterGrokCodeFast1, nil},
		{"claude-sonnet-4", builderOpenRouterClaudeSonnet4, nil},
		{"grok-4-fast-free", builderOpenRouterGrok4FastFree, nil},
		{"gemini-2.5-flash", builderOpenRouterGemini25Flash, nil},
		{"gemini-2.0-flash", builderOpenRouterGemini20Flash, nil},
		{"deepseek-chat-v3.1-free", builderOpenRouterDeepseekV31Free, nil},
		{"qwen3-235b-a22b-2507", builderOpenRouterQwen3Instruct, nil},
		{"gpt-5", builderOpenRouterGPT5, nil},
		{"glm-4.5", builderOpenRouterGLM45, nil},
	})
	opts := ai.ProviderOptions{
		openrouter.Name: &openrouter.ProviderOptions{
			Reasoning: &openrouter.ReasoningOptions{
				Effort: openrouter.ReasoningEffortOption(openrouter.ReasoningEffortMedium),
			},
		},
	}
	testThinking(t, []builderPair{
		{"gpt-5", builderOpenRouterGPT5, opts},
		{"glm-4.5", builderOpenRouterGLM45, opts},
	}, testOpenrouterThinking)
}

func testOpenrouterThinking(t *testing.T, result *ai.AgentResult) {
	reasoningContentCount := 0
	for _, step := range result.Steps {
		for _, msg := range step.Messages {
			for _, content := range msg.Content {
				if content.GetType() == ai.ContentTypeReasoning {
					reasoningContentCount += 1
				}
			}
		}
	}
	require.Greater(t, reasoningContentCount, 0)
}

func TestWithUniqueToolCallIDs(t *testing.T) {
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
		for _, tc := range toolCalls {
			require.False(t, tc.Invalid)
			require.Contains(t, tc.ToolCallID, "test-")
		}
		require.Len(t, toolCalls, 2)

		finalText := result.Response.Content.Text()
		require.Contains(t, finalText, "5", "expected response to contain '5', got: %q", finalText)
		require.Contains(t, finalText, "6", "expected response to contain '6', got: %q", finalText)
	}

	id := 0
	generateIDFunc := func() string {
		id += 1
		return fmt.Sprintf("test-%d", id)
	}

	t.Run("unique tool call ids", func(t *testing.T) {
		r := newRecorder(t)

		provider := openrouter.New(
			openrouter.WithAPIKey(os.Getenv("OPENROUTER_API_KEY")),
			openrouter.WithHTTPClient(&http.Client{Transport: r}),
			openrouter.WithLanguageUniqueToolCallIds(),
			openrouter.WithLanguageModelGenerateIDFunc(generateIDFunc),
		)
		languageModel, err := provider.LanguageModel("moonshotai/kimi-k2-0905")
		require.NoError(t, err, "failed to build language model")

		agent := ai.NewAgent(
			languageModel,
			ai.WithSystemPrompt("You are a helpful assistant. CRITICAL: Always use both add and multiply at the same time ALWAYS."),
			ai.WithTools(addTool),
			ai.WithTools(multiplyTool),
		)
		result, err := agent.Generate(t.Context(), ai.AgentCall{
			Prompt:          "Add and multiply the number 2 and 3",
			MaxOutputTokens: ai.IntOption(4000),
		})
		require.NoError(t, err, "failed to generate")
		checkResult(t, result)
	})
	t.Run("stream unique tool call ids", func(t *testing.T) {
		r := newRecorder(t)

		provider := openrouter.New(
			openrouter.WithAPIKey(os.Getenv("OPENROUTER_API_KEY")),
			openrouter.WithHTTPClient(&http.Client{Transport: r}),
			openrouter.WithLanguageUniqueToolCallIds(),
			openrouter.WithLanguageModelGenerateIDFunc(generateIDFunc),
		)
		languageModel, err := provider.LanguageModel("moonshotai/kimi-k2-0905")
		require.NoError(t, err, "failed to build language model")

		agent := ai.NewAgent(
			languageModel,
			ai.WithSystemPrompt("You are a helpful assistant. Always use both add and multiply at the same time."),
			ai.WithTools(addTool),
			ai.WithTools(multiplyTool),
		)
		result, err := agent.Stream(t.Context(), ai.AgentStreamCall{
			Prompt:          "Add and multiply the number 2 and 3",
			MaxOutputTokens: ai.IntOption(4000),
		})
		require.NoError(t, err, "failed to generate")
		checkResult(t, result)
	})
}

func builderOpenRouterKimiK2(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := openrouter.New(
		openrouter.WithAPIKey(os.Getenv("OPENROUTER_API_KEY")),
		openrouter.WithHTTPClient(&http.Client{Transport: r}),
	)
	return provider.LanguageModel("moonshotai/kimi-k2-0905")
}

func builderOpenRouterGrokCodeFast1(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := openrouter.New(
		openrouter.WithAPIKey(os.Getenv("OPENROUTER_API_KEY")),
		openrouter.WithHTTPClient(&http.Client{Transport: r}),
	)
	return provider.LanguageModel("x-ai/grok-code-fast-1")
}

func builderOpenRouterGrok4FastFree(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := openrouter.New(
		openrouter.WithAPIKey(os.Getenv("OPENROUTER_API_KEY")),
		openrouter.WithHTTPClient(&http.Client{Transport: r}),
	)
	return provider.LanguageModel("x-ai/grok-4-fast:free")
}

func builderOpenRouterGemini25Flash(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := openrouter.New(
		openrouter.WithAPIKey(os.Getenv("OPENROUTER_API_KEY")),
		openrouter.WithHTTPClient(&http.Client{Transport: r}),
	)
	return provider.LanguageModel("google/gemini-2.5-flash")
}

func builderOpenRouterGemini20Flash(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := openrouter.New(
		openrouter.WithAPIKey(os.Getenv("OPENROUTER_API_KEY")),
		openrouter.WithHTTPClient(&http.Client{Transport: r}),
	)
	return provider.LanguageModel("google/gemini-2.0-flash-001")
}

func builderOpenRouterDeepseekV31Free(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := openrouter.New(
		openrouter.WithAPIKey(os.Getenv("OPENROUTER_API_KEY")),
		openrouter.WithHTTPClient(&http.Client{Transport: r}),
	)
	return provider.LanguageModel("deepseek/deepseek-chat-v3.1:free")
}

func builderOpenRouterClaudeSonnet4(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := openrouter.New(
		openrouter.WithAPIKey(os.Getenv("OPENROUTER_API_KEY")),
		openrouter.WithHTTPClient(&http.Client{Transport: r}),
	)
	return provider.LanguageModel("anthropic/claude-sonnet-4")
}

func builderOpenRouterGPT5(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := openrouter.New(
		openrouter.WithAPIKey(os.Getenv("OPENROUTER_API_KEY")),
		openrouter.WithHTTPClient(&http.Client{Transport: r}),
	)
	return provider.LanguageModel("openai/gpt-5")
}

func builderOpenRouterGLM45(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := openrouter.New(
		openrouter.WithAPIKey(os.Getenv("OPENROUTER_API_KEY")),
		openrouter.WithHTTPClient(&http.Client{Transport: r}),
	)
	return provider.LanguageModel("z-ai/glm-4.5")
}

func builderOpenRouterQwen3Instruct(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := openrouter.New(
		openrouter.WithAPIKey(os.Getenv("OPENROUTER_API_KEY")),
		openrouter.WithHTTPClient(&http.Client{Transport: r}),
	)
	return provider.LanguageModel("qwen/qwen3-235b-a22b-2507")
}
