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
