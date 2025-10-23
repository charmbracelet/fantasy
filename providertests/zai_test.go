package providertests

import (
	"net/http"
	"os"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openaicompat"
	"charm.land/fantasy/providers/zai"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func TestZAICommon(t *testing.T) {
	testCommon(t, []builderPair{
		{"zai-glm-4.6", builderZAI, nil, nil},
	})
}

func builderZAI(t *testing.T, r *recorder.Recorder) (fantasy.LanguageModel, error) {
	provider, err := zai.New(
		openaicompat.WithAPIKey(os.Getenv("FANTASY_ZAI_API_KEY")),
		openaicompat.WithHTTPClient(&http.Client{Transport: r}),
	)
	if err != nil {
		return nil, err
	}
	return provider.LanguageModel(t.Context(), "glm-4.6")
}
