package providertests

import (
	"cmp"
	"net/http"
	"os"
	"testing"

	"github.com/charmbracelet/fantasy/ai"
	"github.com/charmbracelet/fantasy/azure"
	"github.com/charmbracelet/fantasy/openai"
	"github.com/stretchr/testify/require"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

const defaultBaseURL = "https://fantasy-playground-resource.services.ai.azure.com/"

func TestAzureCommon(t *testing.T) {
	testCommon(t, []builderPair{
		{"azure-o4-mini", builderAzureO4Mini, nil},
		{"azure-gpt-5-mini", builderAzureGpt5Mini, nil},
		{"azure-grok-3-mini", builderAzureGrok3Mini, nil},
	})
}

func TestAzureThinking(t *testing.T) {
	opts := ai.ProviderOptions{
		openai.Name: &openai.ProviderOptions{
			ReasoningEffort: openai.ReasoningEffortOption(openai.ReasoningEffortLow),
		},
	}
	testThinking(t, []builderPair{
		{"azure-gpt-5-mini", builderAzureGpt5Mini, opts},
		{"azure-grok-3-mini", builderAzureGrok3Mini, opts},
	}, testAzureThinking)
}

func testAzureThinking(t *testing.T, result *ai.AgentResult) {
	require.Greater(t, result.Response.Usage.ReasoningTokens, int64(0), "expected reasoning tokens, got none")
}

func builderAzureO4Mini(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := azure.New(
		azure.WithBaseURL(cmp.Or(os.Getenv("FANTASY_AZURE_BASE_URL"), defaultBaseURL)),
		azure.WithAPIKey(cmp.Or(os.Getenv("FANTASY_AZURE_API_KEY"), "(missing)")),
		azure.WithHTTPClient(&http.Client{Transport: r}),
	)
	return provider.LanguageModel("o4-mini")
}

func builderAzureGpt5Mini(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := azure.New(
		azure.WithBaseURL(cmp.Or(os.Getenv("FANTASY_AZURE_BASE_URL"), defaultBaseURL)),
		azure.WithAPIKey(cmp.Or(os.Getenv("FANTASY_AZURE_API_KEY"), "(missing)")),
		azure.WithHTTPClient(&http.Client{Transport: r}),
	)
	return provider.LanguageModel("gpt-5-mini")
}

func builderAzureGrok3Mini(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := azure.New(
		azure.WithBaseURL(cmp.Or(os.Getenv("FANTASY_AZURE_BASE_URL"), defaultBaseURL)),
		azure.WithAPIKey(cmp.Or(os.Getenv("FANTASY_AZURE_API_KEY"), "(missing)")),
		azure.WithHTTPClient(&http.Client{Transport: r}),
	)
	return provider.LanguageModel("grok-3-mini")
}
