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

func azureModel(model string) builderFunc {
	return func(t *testing.T, r *vcr.Recorder) (fantasy.LanguageModel, error) {
		provider, err := azure.New(
			azure.WithBaseURL(cmp.Or(os.Getenv("FANTASY_AZURE_BASE_URL"), defaultBaseURL)),
			azure.WithAPIKey(cmp.Or(os.Getenv("FANTASY_AZURE_API_KEY"), "(missing)")),
			azure.WithHTTPClient(&http.Client{Transport: r}),
		)
		if err != nil {
			return nil, err
		}
		return provider.LanguageModel(t.Context(), model)
	}
}

func TestAzureCommon(t *testing.T) {
	testCommon(t, []builderPair{
		{"azure-o4-mini", azureModel("o4-mini"), nil, nil},
		{"azure-gpt-5-mini", azureModel("gpt-5-mini"), nil, nil},
	})
}

func TestAzureThinking(t *testing.T) {
	opts := fantasy.ProviderOptions{
		openai.Name: &openai.ProviderOptions{
			ReasoningEffort: openai.ReasoningEffortOption(openai.ReasoningEffortHigh),
		},
	}
	testThinking(t, []builderPair{
		{"azure-gpt-5-mini", azureModel("gpt-5-mini"), opts, nil},
	}, testAzureThinking)
}

func testAzureThinking(t *testing.T, result *fantasy.AgentResult) {
	require.Greater(t, result.Response.Usage.ReasoningTokens, int64(0), "expected reasoning tokens, got none")
}
