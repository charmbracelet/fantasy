package providertests

import (
	"net/http"
	"os"
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
