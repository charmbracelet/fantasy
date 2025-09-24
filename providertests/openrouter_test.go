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
		openrouterKimiK2(),
		openrouterGrokCodeFast1(),
		openrouterClaudeSonnet4(),
		openrouterGrok4FastFree(),
		openrouterGemini25Flash(),
		openrouterGemini20Flash(),
		openrouterDeepseekV31Free(),
		openrouterGpt5(),
	}

	for _, model := range models {
		// add one entry for multi provider tests
		pairs = append(
			pairs,
			builderPair{
				model.name,
				model.builderFunc,
				nil,
			})
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

func openrouterKimiK2() openrouterModel {
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

func openrouterGrokCodeFast1() openrouterModel {
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

func openrouterGrok4FastFree() openrouterModel {
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

func openrouterGemini25Flash() openrouterModel {
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

func openrouterGemini20Flash() openrouterModel {
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

func openrouterDeepseekV31Free() openrouterModel {
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

func openrouterClaudeSonnet4() openrouterModel {
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

func openrouterGpt5() openrouterModel {
	return openrouterModel{
		name: "gpt-5",
		builderFunc: func(r *recorder.Recorder) (ai.LanguageModel, error) {
			provider := openrouter.New(
				openrouter.WithAPIKey(os.Getenv("OPENROUTER_API_KEY")),
				openrouter.WithHTTPClient(&http.Client{Transport: r}),
			)
			return provider.LanguageModel("openai/gpt-5")
		},
		providers: []string{
			"openai",
			"azure",
		},
	}
}
