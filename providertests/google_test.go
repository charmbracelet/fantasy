package providertests

import (
	"cmp"
	"net/http"
	"os"
	"testing"

	"github.com/charmbracelet/fantasy/ai"
	"github.com/charmbracelet/fantasy/google"
	"github.com/stretchr/testify/require"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

var googleTestModels = []testModel{
	{"gemini-2.5-flash", "gemini-2.5-flash", true},
	{"gemini-2.5-pro", "gemini-2.5-pro", true},
}

func TestGoogleCommon(t *testing.T) {
	var pairs []builderPair
	for _, m := range googleTestModels {
		pairs = append(pairs, builderPair{m.name, googleBuilder(m.model), nil})
	}
	testCommon(t, pairs)
}

func TestGoogleThinking(t *testing.T) {
	opts := ai.ProviderOptions{
		google.Name: &google.ProviderOptions{
			ThinkingConfig: &google.ThinkingConfig{
				ThinkingBudget:  ai.IntOption(100),
				IncludeThoughts: ai.BoolOption(true),
			},
		},
	}

	var pairs []builderPair
	for _, m := range googleTestModels {
		if !m.reasoning {
			continue
		}
		pairs = append(pairs, builderPair{m.name, googleBuilder(m.model), opts})
	}
	testThinking(t, pairs, testGoogleThinking)
}

func testGoogleThinking(t *testing.T, result *ai.AgentResult) {
	reasoningContentCount := 0
	// Test if we got the signature
	for _, step := range result.Steps {
		for _, msg := range step.Messages {
			for _, content := range msg.Content {
				if content.GetType() == ai.ContentTypeReasoning {
					reasoningContentCount += 1
				}
			}
		}
	}
	require.Greater(t, reasoningContentCount, 0)
}

func googleBuilder(model string) builderFunc {
	return func(r *recorder.Recorder) (ai.LanguageModel, error) {
		provider := google.New(
			google.WithAPIKey(cmp.Or(os.Getenv("FANTASY_GEMINI_API_KEY"), "(missing)")),
			google.WithHTTPClient(&http.Client{Transport: r}),
		)
		return provider.LanguageModel(model)
	}
}
