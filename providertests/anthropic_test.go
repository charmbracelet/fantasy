package providertests

import (
	"net/http"
	"os"
	"testing"

	"github.com/charmbracelet/fantasy/ai"
	"github.com/charmbracelet/fantasy/anthropic"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func TestAnthropicCommon(t *testing.T) {
	testCommon(t, []builderPair{
		{"claude-sonnet-4", builderAnthropicClaudeSonnet4, nil},
	})
}

func builderAnthropicClaudeSonnet4(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := anthropic.New(
		anthropic.WithAPIKey(os.Getenv("ANTHROPIC_API_KEY")),
		anthropic.WithHTTPClient(&http.Client{Transport: r}),
	)
	return provider.LanguageModel("claude-sonnet-4-20250514")
}
