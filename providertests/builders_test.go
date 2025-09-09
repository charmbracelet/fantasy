package providertests

import (
	"cmp"
	"net/http"
	"os"

	"github.com/charmbracelet/fantasy/ai"
	"github.com/charmbracelet/fantasy/anthropic"
	"github.com/charmbracelet/fantasy/google"
	"github.com/charmbracelet/fantasy/openai"
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
	{"google-gemini-2.5-flash", builderGoogleGemini25Flash},
	{"google-gemini-2.5-pro", builderGoogleGemini25Pro},
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

func builderAnthropicClaudeSonnet4(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := anthropic.New(
		anthropic.WithAPIKey(os.Getenv("ANTHROPIC_API_KEY")),
		anthropic.WithHTTPClient(&http.Client{Transport: r}),
	)
	return provider.LanguageModel("claude-sonnet-4-20250514")
}

func builderGoogleGemini25Flash(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := google.New(
		google.WithAPIKey(cmp.Or(os.Getenv("GEMINI_API_KEY"), "(missing)")),
		google.WithHTTPClient(&http.Client{Transport: r}),
	)
	return provider.LanguageModel("gemini-2.5-flash")
}

func builderGoogleGemini25Pro(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := google.New(
		google.WithAPIKey(cmp.Or(os.Getenv("GEMINI_API_KEY"), "(missing)")),
		google.WithHTTPClient(&http.Client{Transport: r}),
	)
	return provider.LanguageModel("gemini-2.5-flash")
}
