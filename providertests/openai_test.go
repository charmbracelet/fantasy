package providertests

import (
	"net/http"
	"os"
	"testing"

	"github.com/charmbracelet/fantasy/ai"
	"github.com/charmbracelet/fantasy/openai"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func TestOpenAICommon(t *testing.T) {
	testCommon(t, []builderPair{
		{"gpt-4o", builderOpenaiGpt4o, nil},
		{"gpt-4o-mini", builderOpenaiGpt4oMini, nil},
		{"gpt-5", builderOpenaiGpt5, nil},
	})
}

func builderOpenaiGpt4o(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := openai.New(
		openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
		openai.WithHTTPClient(&http.Client{Transport: r}),
	)
	return provider.LanguageModel("gpt-4o")
}

func builderOpenaiGpt4oMini(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := openai.New(
		openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
		openai.WithHTTPClient(&http.Client{Transport: r}),
	)
	return provider.LanguageModel("gpt-4o-mini")
}

func builderOpenaiGpt5(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := openai.New(
		openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
		openai.WithHTTPClient(&http.Client{Transport: r}),
	)
	return provider.LanguageModel("gpt-5")
}
