package providertests

import (
	"net/http"
	"os"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/xai"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func TestXAICommon(t *testing.T) {
	testCommon(t, []builderPair{
		{"xai-grok-code-fast", builderXAI, nil, nil},
	})
}

func builderXAI(t *testing.T, r *recorder.Recorder) (fantasy.LanguageModel, error) {
	provider, err := xai.New(
		xai.WithAPIKey(os.Getenv("FANTASY_XAI_API_KEY")),
		xai.WithHTTPClient(&http.Client{Transport: r}),
	)
	if err != nil {
		return nil, err
	}
	return provider.LanguageModel(t.Context(), "grok-code-fast-1")
}
