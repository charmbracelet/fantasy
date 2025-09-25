package providertests

import (
	"net/http"
	"os"
	"testing"

	"github.com/charmbracelet/fantasy/ai"
	"github.com/charmbracelet/fantasy/openai"
	"github.com/charmbracelet/fantasy/openaicompat"
	"github.com/stretchr/testify/require"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func TestOpenAICompatibleCommon(t *testing.T) {
	testCommon(t, []builderPair{
		{"xai-grok-4-fast", builderXAIGrok4Fast, nil},
		{"xai-grok-code-fast", builderXAIGrokCodeFast, nil},
		{"groq-kimi-k2", builderGroq, nil},
		{"zai-glm-4.5", builderZAIGLM45, nil},
	})
	opts := ai.ProviderOptions{
		openaicompat.Name: &openaicompat.ProviderOptions{
			ReasoningEffort: openai.ReasoningEffortOption(openai.ReasoningEffortHigh),
		},
	}
	testThinking(t, []builderPair{
		{"xai-grok-3-mini", builderXAIGrok3Mini, opts},
		{"zai-glm-4.5", builderZAIGLM45, opts},
	}, testOpenAICompatThinking)
}

func testOpenAICompatThinking(t *testing.T, result *ai.AgentResult) {
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

func builderXAIGrokCodeFast(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := openaicompat.New(
		"https://api.x.ai/v1",
		openaicompat.WithAPIKey(os.Getenv("XAI_API_KEY")),
		openaicompat.WithHTTPClient(&http.Client{Transport: r}),
	)
	return provider.LanguageModel("grok-code-fast-1")
}

func builderXAIGrok4Fast(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := openaicompat.New(
		"https://api.x.ai/v1",
		openaicompat.WithAPIKey(os.Getenv("XAI_API_KEY")),
		openaicompat.WithHTTPClient(&http.Client{Transport: r}),
	)
	return provider.LanguageModel("grok-4-fast")
}

func builderXAIGrok3Mini(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := openaicompat.New(
		"https://api.x.ai/v1",
		openaicompat.WithAPIKey(os.Getenv("XAI_API_KEY")),
		openaicompat.WithHTTPClient(&http.Client{Transport: r}),
	)
	return provider.LanguageModel("grok-3-mini")
}

func builderZAIGLM45(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := openaicompat.New(
		"https://api.z.ai/api/coding/paas/v4",
		openaicompat.WithAPIKey(os.Getenv("ZAI_API_KEY")),
		openaicompat.WithHTTPClient(&http.Client{Transport: r}),
	)
	return provider.LanguageModel("glm-4.5")
}

func builderGroq(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := openaicompat.New(
		"https://api.groq.com/openai/v1",
		openaicompat.WithAPIKey(os.Getenv("GROQ_API_KEY")),
		openaicompat.WithHTTPClient(&http.Client{Transport: r}),
	)
	return provider.LanguageModel("moonshotai/kimi-k2-instruct-0905")
}
