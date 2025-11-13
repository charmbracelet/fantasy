package providertests

import (
	"net/http"
	"os"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openai"
	"charm.land/x/vcr"
)

var openaiTestModels = []testModel{
	{"openai-gpt-4o", "gpt-4o", false},
	{"openai-gpt-4o-mini", "gpt-4o-mini", false},
	{"openai-gpt-5", "gpt-5", true},
	{"openai-o4-mini", "o4-mini", true},
}

func TestOpenAICommon(t *testing.T) {
	var pairs []builderPair
	for _, m := range openaiTestModels {
		pairs = append(pairs, builderPair{m.name, openAIBuilder(m.model), nil, nil})
	}
	testCommon(t, pairs)
}

func TestOpenAIObjectGeneration(t *testing.T) {
	var pairs []builderPair
	for _, m := range openaiTestModels {
		pairs = append(pairs, builderPair{m.name, openAIBuilder(m.model), nil, nil})
	}
	testObjectGeneration(t, pairs)
}

func openAIBuilder(model string) builderFunc {
	return func(t *testing.T, r *vcr.Recorder) (fantasy.LanguageModel, error) {
		provider, err := openai.New(
			openai.WithAPIKey(os.Getenv("FANTASY_OPENAI_API_KEY")),
			openai.WithHTTPClient(&http.Client{Transport: r}),
		)
		if err != nil {
			return nil, err
		}
		return provider.LanguageModel(t.Context(), model)
	}
}
