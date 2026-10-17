package providertests

import (
	"cmp"
	"net/http"
	"os"
	"testing"

	"charm.land/fantasy/providers/openai"
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
	return vcrProviderBuilder(model,
		openai.New,
		openai.WithAPIKey,
		func(client *http.Client) openai.Option { return openai.WithHTTPClient(client) },
		cmp.Or(os.Getenv("FANTASY_OPENAI_API_KEY"), "(missing)"),
	)
}
