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

func TestGoogleCommon(t *testing.T) {
	testCommon(t, []builderPair{
		{"gemini-2.5-flash", builderGoogleGemini25Flash, nil},
		{"gemini-2.5-pro", builderGoogleGemini25Pro, nil},
	})
	opts := ai.ProviderOptions{
		google.Name: &google.ProviderOptions{
			ThinkingConfig: &google.ThinkingConfig{
				ThinkingBudget:  ai.IntOption(100),
				IncludeThoughts: ai.BoolOption(true),
			},
		},
	}
	testThinking(t, []builderPair{
		{"gemini-2.5-flash", builderGoogleGemini25Flash, opts},
		{"gemini-2.5-pro", builderGoogleGemini25Pro, opts},
	}, testGoogleThinking)
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
