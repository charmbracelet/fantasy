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

var openrouterTestModels = []testModel{
	{"kimi-k2", "moonshotai/kimi-k2-0905", false},
	{"grok-code-fast-1", "x-ai/grok-code-fast-1", false},
	{"claude-sonnet-4", "anthropic/claude-sonnet-4", true},
	{"gemini-2.5-flash", "google/gemini-2.5-flash", false},
	{"deepseek-chat-v3.1-free", "deepseek/deepseek-chat-v3.1:free", false},
	{"qwen3-235b-a22b-2507", "qwen/qwen3-235b-a22b-2507", false},
	{"gpt-5", "openai/gpt-5", true},
	{"glm-4.5", "z-ai/glm-4.5", false},
}

func TestOpenRouterCommon(t *testing.T) {
	var pairs []builderPair
	for _, m := range openrouterTestModels {
		pairs = append(pairs, builderPair{m.name, openrouterBuilder(m.model), nil})
	}
	testCommon(t, pairs)
}

func TestOpenRouterThinking(t *testing.T) {
	opts := ai.ProviderOptions{
		openrouter.Name: &openrouter.ProviderOptions{
			Reasoning: &openrouter.ReasoningOptions{
				Effort: openrouter.ReasoningEffortOption(openrouter.ReasoningEffortMedium),
			},
		},
	}

	var pairs []builderPair
	for _, m := range openrouterTestModels {
		if !m.reasoning {
			continue
		}
		pairs = append(pairs, builderPair{m.name, openrouterBuilder(m.model), opts})
	}
	testThinking(t, pairs, testOpenrouterThinking)

	// test anthropic signature
	testThinking(t, []builderPair{
		{"claude-sonnet-4-sig", openrouterBuilder("anthropic/claude-sonnet-4"), opts},
	}, testOpenrouterThinkingWithSignature)
}

func testOpenrouterThinkingWithSignature(t *testing.T, result *ai.AgentResult) {
	reasoningContentCount := 0
	signaturesCount := 0
	// Test if we got the signature
	for _, step := range result.Steps {
		for _, msg := range step.Messages {
			for _, content := range msg.Content {
				if content.GetType() == ai.ContentTypeReasoning {
					reasoningContentCount += 1
					reasoningContent, ok := ai.AsContentType[ai.ReasoningPart](content)
					if !ok {
						continue
					}
					if len(reasoningContent.ProviderOptions) == 0 {
						continue
					}

					anthropicReasoningMetadata, ok := reasoningContent.ProviderOptions[openrouter.Name]
					if !ok {
						continue
					}
					if reasoningContent.Text != "" {
						if typed, ok := anthropicReasoningMetadata.(*openrouter.ReasoningMetadata); ok {
							require.NotEmpty(t, typed.Signature)
							signaturesCount += 1
						}
					}
				}
			}
		}
	}
	require.Greater(t, reasoningContentCount, 0)
	require.Greater(t, signaturesCount, 0)
	require.Equal(t, reasoningContentCount, signaturesCount)
	// we also add the anthropic metadata so test that
	testAnthropicThinking(t, result)
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

func TestOpenRouterWithUniqueToolCallIDs(t *testing.T) {
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

func openrouterBuilder(model string) builderFunc {
	return func(r *recorder.Recorder) (ai.LanguageModel, error) {
		provider := openrouter.New(
			openrouter.WithAPIKey(os.Getenv("OPENROUTER_API_KEY")),
			openrouter.WithHTTPClient(&http.Client{Transport: r}),
		)
		return provider.LanguageModel(model)
	}
}
