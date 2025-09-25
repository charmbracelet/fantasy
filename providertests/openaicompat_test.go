package providertests

import (
	"net/http"
	"os"
	"testing"

	"github.com/charmbracelet/fantasy/ai"
	"github.com/charmbracelet/fantasy/openaicompat"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func TestOpenAICompatibleCommon(t *testing.T) {
	testCommon(t, []builderPair{
		{"xai-grok-4-fast", builderXAIGrok4Fast, nil},
		{"xai-grok-code-fast", builderXAIGrokCodeFast, nil},
		{"groq-kimi-k2", builderGroq, nil},
	})
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

func builderGroq(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := openaicompat.New(
		"https://api.groq.com/openai/v1",
		openaicompat.WithAPIKey(os.Getenv("GROQ_API_KEY")),
		openaicompat.WithHTTPClient(&http.Client{Transport: r}),
	)
	return provider.LanguageModel("moonshotai/kimi-k2-instruct-0905")
}
