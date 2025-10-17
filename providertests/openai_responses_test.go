package providertests

import (
	"net/http"
	"os"
	"testing"

	"charm.land/fantasy/ai"
	"charm.land/fantasy/openai"
	"github.com/stretchr/testify/require"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func TestOpenAIResponsesCommon(t *testing.T) {
	var pairs []builderPair
	for _, m := range openaiTestModels {
		pairs = append(pairs, builderPair{m.name, openAIReasoningBuilder(m.model), nil})
	}
	testCommon(t, pairs)
}

func openAIReasoningBuilder(model string) builderFunc {
	return func(r *recorder.Recorder) (ai.LanguageModel, error) {
		provider := openai.New(
			openai.WithAPIKey(os.Getenv("FANTASY_OPENAI_API_KEY")),
			openai.WithHTTPClient(&http.Client{Transport: r}),
			openai.WithUseResponsesAPI(),
		)
		return provider.LanguageModel(model)
	}
}

func TestOpenAIResponsesWithSummaryThinking(t *testing.T) {
	opts := ai.ProviderOptions{
		openai.Name: &openai.ResponsesProviderOptions{
			Include: []openai.IncludeType{
				openai.IncludeReasoningEncryptedContent,
			},
			ReasoningEffort:  openai.ReasoningEffortOption(openai.ReasoningEffortHigh),
			ReasoningSummary: ai.Opt("auto"),
		},
	}
	var pairs []builderPair
	for _, m := range openaiTestModels {
		if !m.reasoning {
			continue
		}
		pairs = append(pairs, builderPair{m.name, openAIReasoningBuilder(m.model), opts})
	}
	testThinking(t, pairs, testOpenAIResponsesThinkingWithSummaryThinking)
}

func testOpenAIResponsesThinkingWithSummaryThinking(t *testing.T, result *ai.AgentResult) {
	reasoningContentCount := 0
	encryptedData := 0
	// Test if we got the signature
	for _, step := range result.Steps {
		for _, msg := range step.Messages {
			for _, content := range msg.Content {
				if content.GetType() == ai.ContentTypeReasoning {
					reasoningContentCount += 1
					reasoningContent, ok := ai.AsContentType[ai.ReasoningPart](content)
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
