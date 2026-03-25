package providertests

import (
	"cmp"
	"net/http"
	"os"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openai"
	"charm.land/x/vcr"
	"github.com/stretchr/testify/require"
)

func openAIWebSearchBuilder(model string) builderFunc {
	return func(t *testing.T, r *vcr.Recorder) (fantasy.LanguageModel, error) {
		opts := []openai.Option{
			openai.WithAPIKey(cmp.Or(os.Getenv("FANTASY_OPENAI_API_KEY"), os.Getenv("OPENAI_API_KEY"), "(missing)")),
			openai.WithHTTPClient(&http.Client{Transport: r}),
			openai.WithUseResponsesAPI(),
		}
		provider, err := openai.New(opts...)
		if err != nil {
			return nil, err
		}
		return provider.LanguageModel(t.Context(), model)
	}
}

// TestOpenAIWebSearch tests web search tool support via the agent
// using WithProviderDefinedTools on the OpenAI Responses API.
func TestOpenAIWebSearch(t *testing.T) {
	model := "gpt-4.1"
	webSearchTool := openai.WebSearchTool(nil)

	t.Run("generate", func(t *testing.T) {
		r := vcr.NewRecorder(t)

		lm, err := openAIWebSearchBuilder(model)(t, r)
		require.NoError(t, err)

		agent := fantasy.NewAgent(
			lm,
			fantasy.WithSystemPrompt("You are a helpful assistant"),
			fantasy.WithProviderDefinedTools(webSearchTool),
		)

		result, err := agent.Generate(t.Context(), fantasy.AgentCall{
			Prompt:          "What is the current population of Tokyo? Cite your source.",
			MaxOutputTokens: new(int64(4000)),
		})
		require.NoError(t, err)

		got := result.Response.Content.Text()
		require.NotEmpty(t, got, "should have a text response")
		require.Contains(t, got, "Tokyo", "response should mention Tokyo")

		// Walk the steps and verify web search content was produced.
		var sources []fantasy.SourceContent
		var providerToolCalls []fantasy.ToolCallContent
		for _, step := range result.Steps {
			for _, c := range step.Content {
				switch v := c.(type) {
				case fantasy.ToolCallContent:
					if v.ProviderExecuted {
						providerToolCalls = append(providerToolCalls, v)
					}
				case fantasy.SourceContent:
					sources = append(sources, v)
				}
			}
		}

		require.NotEmpty(t, providerToolCalls, "should have provider-executed tool calls")
		require.Equal(t, "web_search", providerToolCalls[0].ToolName)
		// Sources come from url_citation annotations; the model
		// may or may not include inline citations so we don't
		// require them, but if present they should have URLs.
		for _, src := range sources {
			require.NotEmpty(t, src.URL, "source should have a URL")
		}
	})

	t.Run("stream", func(t *testing.T) {
		r := vcr.NewRecorder(t)

		lm, err := openAIWebSearchBuilder(model)(t, r)
		require.NoError(t, err)

		agent := fantasy.NewAgent(
			lm,
			fantasy.WithSystemPrompt("You are a helpful assistant"),
			fantasy.WithProviderDefinedTools(webSearchTool),
		)

		// Turn 1: initial query triggers web search.
		result, err := agent.Stream(t.Context(), fantasy.AgentStreamCall{
			Prompt:          "What is the current population of Tokyo? Cite your source.",
			MaxOutputTokens: new(int64(4000)),
		})
		require.NoError(t, err)

		got := result.Response.Content.Text()
		require.NotEmpty(t, got, "should have a text response")
		require.Contains(t, got, "Tokyo", "response should mention Tokyo")

		// Verify provider-executed tool calls and results in steps.
		var providerToolCalls []fantasy.ToolCallContent
		var providerToolResults []fantasy.ToolResultContent
		for _, step := range result.Steps {
			for _, c := range step.Content {
				switch v := c.(type) {
				case fantasy.ToolCallContent:
					if v.ProviderExecuted {
						providerToolCalls = append(providerToolCalls, v)
					}
				case fantasy.ToolResultContent:
					if v.ProviderExecuted {
						providerToolResults = append(providerToolResults, v)
					}
				}
			}
		}
		require.NotEmpty(t, providerToolCalls, "should have provider-executed tool calls")
		require.Equal(t, "web_search", providerToolCalls[0].ToolName)
		require.NotEmpty(t, providerToolResults, "should have provider-executed tool results")
	})
}
