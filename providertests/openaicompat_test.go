package providertests

import (
	"net/http"
	"os"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openai"
	"charm.land/fantasy/providers/openaicompat"
	"github.com/stretchr/testify/require"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func TestOpenAICompatibleCommon(t *testing.T) {
	testCommon(t, []builderPair{
		{"xai-grok-4-fast", builderXAIGrok4Fast, nil, nil},
		{"xai-grok-code-fast", builderXAIGrokCodeFast, nil, nil},
		{"groq-kimi-k2", builderGroq, nil, nil},
		{"zai-glm-4.5", builderZAIGLM45, nil, nil},
		{"huggingface-qwen3-coder", builderHuggingFace, nil, nil},
		{"llama-cpp-gpt-oss", builderLlamaCppGptOss, nil, nil},
	})
}

func TestOpenAICompatibleThinking(t *testing.T) {
	opts := fantasy.ProviderOptions{
		openaicompat.Name: &openaicompat.ProviderOptions{
			ReasoningEffort: openai.ReasoningEffortOption(openai.ReasoningEffortHigh),
		},
	}
	testThinking(t, []builderPair{
		{"xai-grok-3-mini", builderXAIGrok3Mini, opts, nil},
		{"zai-glm-4.5", builderZAIGLM45, opts, nil},
		{"llama-cpp-gpt-oss", builderLlamaCppGptOss, opts, nil},
	}, testOpenAICompatThinking)
}

func testOpenAICompatThinking(t *testing.T, result *fantasy.AgentResult) {
	reasoningContentCount := 0
	for _, step := range result.Steps {
		for _, msg := range step.Messages {
			for _, content := range msg.Content {
				if content.GetType() == fantasy.ContentTypeReasoning {
					reasoningContentCount += 1
				}
			}
		}
	}
	require.Greater(t, reasoningContentCount, 0, "expected reasoning content, got none")
}

func builderXAIGrokCodeFast(t *testing.T, r *recorder.Recorder) (fantasy.LanguageModel, error) {
	provider, err := openaicompat.New(
		openaicompat.WithBaseURL("https://api.x.ai/v1"),
		openaicompat.WithAPIKey(os.Getenv("FANTASY_XAI_API_KEY")),
		openaicompat.WithHTTPClient(&http.Client{Transport: r}),
	)
	if err != nil {
		return nil, err
	}
	return provider.LanguageModel(t.Context(), "grok-code-fast-1")
}

func builderXAIGrok4Fast(t *testing.T, r *recorder.Recorder) (fantasy.LanguageModel, error) {
	provider, err := openaicompat.New(
		openaicompat.WithBaseURL("https://api.x.ai/v1"),
		openaicompat.WithAPIKey(os.Getenv("FANTASY_XAI_API_KEY")),
		openaicompat.WithHTTPClient(&http.Client{Transport: r}),
	)
	if err != nil {
		return nil, err
	}
	return provider.LanguageModel(t.Context(), "grok-4-fast")
}

func builderXAIGrok3Mini(t *testing.T, r *recorder.Recorder) (fantasy.LanguageModel, error) {
	provider, err := openaicompat.New(
		openaicompat.WithBaseURL("https://api.x.ai/v1"),
		openaicompat.WithAPIKey(os.Getenv("FANTASY_XAI_API_KEY")),
		openaicompat.WithHTTPClient(&http.Client{Transport: r}),
	)
	if err != nil {
		return nil, err
	}
	return provider.LanguageModel(t.Context(), "grok-3-mini")
}

func builderZAIGLM45(t *testing.T, r *recorder.Recorder) (fantasy.LanguageModel, error) {
	provider, err := openaicompat.New(
		openaicompat.WithBaseURL("https://api.z.ai/api/coding/paas/v4"),
		openaicompat.WithAPIKey(os.Getenv("FANTASY_ZAI_API_KEY")),
		openaicompat.WithHTTPClient(&http.Client{Transport: r}),
	)
	if err != nil {
		return nil, err
	}
	return provider.LanguageModel(t.Context(), "glm-4.5")
}

func builderGroq(t *testing.T, r *recorder.Recorder) (fantasy.LanguageModel, error) {
	provider, err := openaicompat.New(
		openaicompat.WithBaseURL("https://api.groq.com/openai/v1"),
		openaicompat.WithAPIKey(os.Getenv("FANTASY_GROQ_API_KEY")),
		openaicompat.WithHTTPClient(&http.Client{Transport: r}),
	)
	if err != nil {
		return nil, err
	}
	return provider.LanguageModel(t.Context(), "moonshotai/kimi-k2-instruct-0905")
}

func builderHuggingFace(t *testing.T, r *recorder.Recorder) (fantasy.LanguageModel, error) {
	provider, err := openaicompat.New(
		openaicompat.WithBaseURL("https://router.huggingface.co/v1"),
		openaicompat.WithAPIKey(os.Getenv("FANTASY_HUGGINGFACE_API_KEY")),
		openaicompat.WithHTTPClient(&http.Client{Transport: r}),
	)
	if err != nil {
		return nil, err
	}
	return provider.LanguageModel(t.Context(), "zai-org/GLM-4.6:cerebras")
}

func builderLlamaCppGptOss(t *testing.T, r *recorder.Recorder) (fantasy.LanguageModel, error) {
	provider, err := openaicompat.New(
		openaicompat.WithBaseURL("http://localhost:8080/v1"),
		openaicompat.WithHTTPClient(&http.Client{Transport: r}),
	)
	if err != nil {
		return nil, err
	}
	return provider.LanguageModel(t.Context(), "openai/gpt-oss-20b")
}
