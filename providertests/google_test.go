package providertests

import (
	"cmp"
	"net/http"
	"os"
	"testing"

	"github.com/charmbracelet/fantasy/ai"
	"github.com/charmbracelet/fantasy/google"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func TestGoogleCommon(t *testing.T) {
	testCommon(t, []builderPair{
		{"gemini-2.5-flash", builderGoogleGemini25Flash, nil},
		{"gemini-2.5-pro", builderGoogleGemini25Pro, nil},
	})
}

func builderGoogleGemini25Flash(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := google.New(
		google.WithAPIKey(cmp.Or(os.Getenv("GEMINI_API_KEY"), "(missing)")),
		google.WithHTTPClient(&http.Client{Transport: r}),
	)
	return provider.LanguageModel("gemini-2.5-flash")
}

func builderGoogleGemini25Pro(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := google.New(
		google.WithAPIKey(cmp.Or(os.Getenv("GEMINI_API_KEY"), "(missing)")),
		google.WithHTTPClient(&http.Client{Transport: r}),
	)
	return provider.LanguageModel("gemini-2.5-pro")
}
