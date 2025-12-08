package providertests

import (
	"cmp"
	"net/http"
	"os"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/azure"
	"charm.land/fantasy/providers/openai"
	"charm.land/x/vcr"
	"github.com/stretchr/testify/require"
)

const defaultBaseURL = "https://fantasy-playground-resource.openai.azure.com"

func TestAzureCommon(t *testing.T) {
	testCommon(t, []builderPair{
		{"azure-o4-mini", builderAzureO4Mini, nil, nil},
		{"azure-gpt-5-mini", builderAzureGpt5Mini, nil, nil},
		{"azure-grok-3-mini", builderAzureGrok3Mini, nil, nil},
	})
}

func TestAzureThinking(t *testing.T) {
	opts := fantasy.ProviderOptions{
		openai.Name: &openai.ProviderOptions{
			ReasoningEffort: openai.ReasoningEffortOption(openai.ReasoningEffortHigh),
		},
	}
	testThinking(t, []builderPair{
		{"azure-gpt-5-mini", builderAzureGpt5Mini, opts, nil},
		{"azure-grok-3-mini", builderAzureGrok3Mini, opts, nil},
	}, testAzureThinking)
}

func testAzureThinking(t *testing.T, result *fantasy.AgentResult) {
	require.Greater(t, result.Response.Usage.ReasoningTokens, int64(0), "expected reasoning tokens, got none")
}

func builderAzureO4Mini(t *testing.T, r *vcr.Recorder) (fantasy.LanguageModel, error) {
	provider, err := azure.New(
		azure.WithBaseURL(cmp.Or(os.Getenv("FANTASY_AZURE_BASE_URL"), defaultBaseURL)),
		azure.WithAPIKey(cmp.Or(os.Getenv("FANTASY_AZURE_API_KEY"), "(missing)")),
		azure.WithHTTPClient(&http.Client{Transport: r}),
	)
	if err != nil {
		return nil, err
	}
	return provider.LanguageModel(t.Context(), "o4-mini")
}

func builderAzureGpt5Mini(t *testing.T, r *vcr.Recorder) (fantasy.LanguageModel, error) {
	provider, err := azure.New(
		azure.WithBaseURL(cmp.Or(os.Getenv("FANTASY_AZURE_BASE_URL"), defaultBaseURL)),
		azure.WithAPIKey(cmp.Or(os.Getenv("FANTASY_AZURE_API_KEY"), "(missing)")),
		azure.WithHTTPClient(&http.Client{Transport: r}),
	)
	if err != nil {
		return nil, err
	}
	return provider.LanguageModel(t.Context(), "gpt-5-mini")
}

func builderAzureGrok3Mini(t *testing.T, r *vcr.Recorder) (fantasy.LanguageModel, error) {
	provider, err := azure.New(
		azure.WithBaseURL(cmp.Or(os.Getenv("FANTASY_AZURE_BASE_URL"), defaultBaseURL)),
		azure.WithAPIKey(cmp.Or(os.Getenv("FANTASY_AZURE_API_KEY"), "(missing)")),
		azure.WithHTTPClient(&http.Client{Transport: r}),
	)
	if err != nil {
		return nil, err
	}
	return provider.LanguageModel(t.Context(), "grok-3-mini")
}
