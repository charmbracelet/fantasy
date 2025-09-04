package providertests

import (
	"net/http"
	"os"

	"github.com/charmbracelet/ai/ai"
	"github.com/charmbracelet/ai/anthropic"
	"github.com/charmbracelet/ai/openai"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

type builderFunc func(r *recorder.Recorder) (ai.LanguageModel, error)

type builderPair struct {
	name    string
	builder builderFunc
}

var languageModelBuilders = []builderPair{
	{"openai-gpt-4o", builderOpenaiGpt4o},
	{"openai-gpt-4o-mini", builderOpenaiGpt4oMini},
	{"anthropic-claude-sonnet", builderAnthropicClaudeSonnet4},
}

func builderOpenaiGpt4o(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := openai.New(
		openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
		openai.WithHTTPClient(&http.Client{Transport: r}),
	)
	model, err := provider.LanguageModel("gpt-4o")
	if err != nil {
		return nil, err
	}
	return model, nil
}

func builderOpenaiGpt4oMini(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := openai.New(
		openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
		openai.WithHTTPClient(&http.Client{Transport: r}),
	)
	model, err := provider.LanguageModel("gpt-4o-mini")
	if err != nil {
		return nil, err
	}
	return model, nil
}

func builderAnthropicClaudeSonnet4(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := anthropic.New(
		anthropic.WithAPIKey(os.Getenv("ANTHROPIC_API_KEY")),
		anthropic.WithHTTPClient(&http.Client{Transport: r}),
	)
	model, err := provider.LanguageModel("claude-sonnet-4-20250514")
	if err != nil {
		return nil, err
	}
	return model, nil
}
