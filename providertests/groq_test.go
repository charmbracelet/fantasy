package providertests

import (
	"net/http"
	"os"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/groq"
	"charm.land/fantasy/providers/openaicompat"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func TestGroqCommon(t *testing.T) {
	testCommon(t, []builderPair{
		{"groq-kimi-k2-0905", builderGroqProvider, nil, nil},
	})
}

func builderGroqProvider(t *testing.T, r *recorder.Recorder) (fantasy.LanguageModel, error) {
	provider, err := groq.New(
		openaicompat.WithAPIKey(os.Getenv("FANTASY_GROQ_API_KEY")),
		openaicompat.WithHTTPClient(&http.Client{Transport: r}),
	)
	if err != nil {
		return nil, err
	}
	return provider.LanguageModel(t.Context(), "moonshotai/kimi-k2-instruct-0905")
}
