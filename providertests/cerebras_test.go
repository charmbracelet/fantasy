package providertests

import (
	"net/http"
	"os"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/cerebras"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func TestCerebrasCommon(t *testing.T) {
	testCommon(t, []builderPair{
		{"cerebras-qwen-3-coder-480b", builderCerebras, nil, nil},
	})
}

func builderCerebras(t *testing.T, r *recorder.Recorder) (fantasy.LanguageModel, error) {
	provider, err := cerebras.New(
		cerebras.WithAPIKey(os.Getenv("FANTASY_CEREBRAS_API_KEY")),
		cerebras.WithHTTPClient(&http.Client{Transport: r}),
	)
	if err != nil {
		return nil, err
	}
	return provider.LanguageModel(t.Context(), "qwen-3-coder-480b")
}
