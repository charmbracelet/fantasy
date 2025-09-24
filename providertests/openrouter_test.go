package providertests

import (
	"net/http"
	"os"
	"testing"

	"github.com/charmbracelet/fantasy/ai"
	"github.com/charmbracelet/fantasy/openrouter"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

type openrouterModel struct {
	name        string
	builderFunc builderFunc
	providers   []string
}

func TestOpenRouterCommon(t *testing.T) {
	var pairs []builderPair
	models := []openrouterModel{
		kimiK2(),
		grokCodeFast1(),
		claudeSonnet4(),
		grok4FastFree(),
		gemini25Flash(),
		gemini20Flash(),
		deepseekV31Free(),
	}

	for _, model := range models {
		for _, provider := range model.providers {
			pairs = append(
				pairs,
				builderPair{
					model.name + "_" + provider,
					model.builderFunc,
					ai.ProviderOptions{
						openrouter.Name: &openrouter.ProviderOptions{
							Provider: &openrouter.Provider{
								Only: []string{provider},
							},
						},
					},
				})
		}
	}

	testCommon(t, pairs)
}

func kimiK2() openrouterModel {
	return openrouterModel{
		name: "kimi-k2",
		builderFunc: func(r *recorder.Recorder) (ai.LanguageModel, error) {
			provider := openrouter.New(
				openrouter.WithAPIKey(os.Getenv("OPENROUTER_API_KEY")),
				openrouter.WithHTTPClient(&http.Client{Transport: r}),
			)
			return provider.LanguageModel("moonshotai/kimi-k2-0905")
		},
		providers: []string{
			"chutes",
			"deepinfra",
			"siliconflow",
			"fireworks",
			"moonshotai",
			"novita",
			"baseten",
			"together",
			"groq",
			"moonshotai/turbo",
			"wandb",
		},
	}
}

func grokCodeFast1() openrouterModel {
	return openrouterModel{
		name: "grok-code-fast-1",
		builderFunc: func(r *recorder.Recorder) (ai.LanguageModel, error) {
			provider := openrouter.New(
				openrouter.WithAPIKey(os.Getenv("OPENROUTER_API_KEY")),
				openrouter.WithHTTPClient(&http.Client{Transport: r}),
			)
			return provider.LanguageModel("x-ai/grok-code-fast-1")
		},
		providers: []string{
			"xai",
		},
	}
}

func grok4FastFree() openrouterModel {
	return openrouterModel{
		name: "grok-4-fast-free",
		builderFunc: func(r *recorder.Recorder) (ai.LanguageModel, error) {
			provider := openrouter.New(
				openrouter.WithAPIKey(os.Getenv("OPENROUTER_API_KEY")),
				openrouter.WithHTTPClient(&http.Client{Transport: r}),
			)
			return provider.LanguageModel("x-ai/grok-4-fast:free")
		},
		providers: []string{
			"xai",
		},
	}
}

func gemini25Flash() openrouterModel {
	return openrouterModel{
		name: "gemini-2.5-flash",
		builderFunc: func(r *recorder.Recorder) (ai.LanguageModel, error) {
			provider := openrouter.New(
				openrouter.WithAPIKey(os.Getenv("OPENROUTER_API_KEY")),
				openrouter.WithHTTPClient(&http.Client{Transport: r}),
			)
			return provider.LanguageModel("google/gemini-2.5-flash")
		},
		providers: []string{
			"google-vertex/global",
			"google-ai-studio",
			"google-vertex",
		},
	}
}

func gemini20Flash() openrouterModel {
	return openrouterModel{
		name: "gemini-2.0-flash",
		builderFunc: func(r *recorder.Recorder) (ai.LanguageModel, error) {
			provider := openrouter.New(
				openrouter.WithAPIKey(os.Getenv("OPENROUTER_API_KEY")),
				openrouter.WithHTTPClient(&http.Client{Transport: r}),
			)
			return provider.LanguageModel("google/gemini-2.0-flash-001")
		},
		providers: []string{
			"google-ai-studio",
			"google-vertex",
		},
	}
}

func deepseekV31Free() openrouterModel {
	return openrouterModel{
		name: "deepseek-chat-v3.1-free",
		builderFunc: func(r *recorder.Recorder) (ai.LanguageModel, error) {
			provider := openrouter.New(
				openrouter.WithAPIKey(os.Getenv("OPENROUTER_API_KEY")),
				openrouter.WithHTTPClient(&http.Client{Transport: r}),
			)
			return provider.LanguageModel("deepseek/deepseek-chat-v3.1:free")
		},
		providers: []string{
			"deepinfra",
		},
	}
}

func claudeSonnet4() openrouterModel {
	return openrouterModel{
		name: "claude-sonnet-4",
		builderFunc: func(r *recorder.Recorder) (ai.LanguageModel, error) {
			provider := openrouter.New(
				openrouter.WithAPIKey(os.Getenv("OPENROUTER_API_KEY")),
				openrouter.WithHTTPClient(&http.Client{Transport: r}),
			)
			return provider.LanguageModel("anthropic/claude-sonnet-4")
		},
		providers: []string{
			"google-vertex",
			"google-vertex/global",
			"anthropic",
			"google-vertex/europe",
			"amazon-bedrock",
		},
	}
}
