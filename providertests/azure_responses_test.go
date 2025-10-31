package providertests

import (
	"cmp"
	"net/http"
	"os"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/azure"
	"charm.land/fantasy/providers/openai"
	"github.com/stretchr/testify/require"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func TestAzureResponsesCommon(t *testing.T) {
	var pairs []builderPair
	models := []testModel{
		{"azure-gpt-5-mini", "gpt-5-mini", true},
		{"azure-o4-mini", "o4-mini", true},
	}
	for _, m := range models {
		pairs = append(pairs, builderPair{m.name, azureReasoningBuilder(m.model), nil, nil})
	}
	testCommon(t, pairs)
}

func azureReasoningBuilder(model string) builderFunc {
	return func(t *testing.T, r *recorder.Recorder) (fantasy.LanguageModel, error) {
		provider, err := azure.New(
			azure.WithBaseURL(cmp.Or(os.Getenv("FANTASY_AZURE_BASE_URL"), defaultBaseURL)),
			azure.WithAPIKey(cmp.Or(os.Getenv("FANTASY_AZURE_API_KEY"), "(missing)")),
			azure.WithHTTPClient(&http.Client{Transport: r}),
			azure.WithUseResponsesAPI(),
		)
		if err != nil {
			return nil, err
		}
		return provider.LanguageModel(t.Context(), model)
	}
}

func TestAzureResponsesWithSummaryThinking(t *testing.T) {
	opts := fantasy.ProviderOptions{
		openai.Name: &openai.ResponsesProviderOptions{
			Include: []openai.IncludeType{
				openai.IncludeReasoningEncryptedContent,
			},
			ReasoningEffort:  openai.ReasoningEffortOption(openai.ReasoningEffortHigh),
			ReasoningSummary: fantasy.Opt("auto"),
		},
	}
	var pairs []builderPair
	models := []testModel{
		{"azure-gpt-5-mini", "gpt-5-mini", true},
	}
	for _, m := range models {
		if !m.reasoning {
			continue
		}
		pairs = append(pairs, builderPair{m.name, azureReasoningBuilder(m.model), opts, nil})
	}
	testThinking(t, pairs, testAzureResponsesThinkingWithSummaryThinking)
}

func testAzureResponsesThinkingWithSummaryThinking(t *testing.T, result *fantasy.AgentResult) {
	reasoningContentCount := 0
	encryptedData := 0
	// Test if we got the signature
	for _, step := range result.Steps {
		for _, msg := range step.Messages {
			for _, content := range msg.Content {
				if content.GetType() == fantasy.ContentTypeReasoning {
					reasoningContentCount += 1
					reasoningContent, ok := fantasy.AsContentType[fantasy.ReasoningPart](content)
					if !ok {
						continue
					}
					if len(reasoningContent.ProviderOptions) == 0 {
						continue
					}

					openaiReasoningMetadata, ok := reasoningContent.ProviderOptions[openai.Name]
					if !ok {
						continue
					}
					if typed, ok := openaiReasoningMetadata.(*openai.ResponsesReasoningMetadata); ok {
						require.NotEmpty(t, typed.EncryptedContent)
						encryptedData += 1
					}
				}
			}
		}
	}
	require.Greater(t, reasoningContentCount, 0)
	require.Greater(t, encryptedData, 0)
	require.Equal(t, reasoningContentCount, encryptedData)
}
