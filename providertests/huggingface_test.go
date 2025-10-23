package providertests

import (
	"net/http"
	"os"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/huggingface"
	"charm.land/fantasy/providers/openaicompat"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func TestHuggingFaceCommon(t *testing.T) {
	testCommon(t, []builderPair{
		{"huggingface-glm-4.6", builderHuggingFaceProvider, nil, nil},
	})
}

func builderHuggingFaceProvider(t *testing.T, r *recorder.Recorder) (fantasy.LanguageModel, error) {
	provider, err := huggingface.New(
		openaicompat.WithAPIKey(os.Getenv("FANTASY_HUGGINGFACE_API_KEY")),
		openaicompat.WithHTTPClient(&http.Client{Transport: r}),
	)
	if err != nil {
		return nil, err
	}
	return provider.LanguageModel(t.Context(), "zai-org/GLM-4.6")
}
