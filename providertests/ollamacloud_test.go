package providertests

import (
	"net/http"
	"os"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/ollamacloud"
	"charm.land/x/vcr"
)

var ollamacloudTestModels = []testModel{
	{"gpt-oss-20b", "gpt-oss:20b-cloud", false},
	{"gpt-oss-120b", "gpt-oss:120b-cloud", true},
}

func TestOllamaCloudCommon(t *testing.T) {
	var pairs []builderPair
	for _, m := range ollamacloudTestModels {
		pairs = append(pairs, builderPair{m.name, ollamacloudBuilder(m.model), nil, nil})
	}
	testCommon(t, pairs)
}

func ollamacloudBuilder(model string) builderFunc {
	return func(t *testing.T, r *vcr.Recorder) (fantasy.LanguageModel, error) {
		provider, err := ollamacloud.New(
			ollamacloud.WithAPIKey(os.Getenv("FANTASY_OLLAMACLOUD_API_KEY")),
			ollamacloud.WithHTTPClient(&http.Client{Transport: r}),
		)
		if err != nil {
			return nil, err
		}
		return provider.LanguageModel(t.Context(), model)
	}
}
