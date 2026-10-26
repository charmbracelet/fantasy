package providertests

import (
	"cmp"
	"net/http"
	"os"
	"slices"
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
				case fantasy.SourceContent:
					sources = append(sources, v)
				}
			}
		}

		require.NotEmpty(t, providerToolCalls, "should have provider-executed tool calls")
		require.Equal(t, "web_search", providerToolCalls[0].ToolName)
		requireWebSearchActionMetadata(t, providerToolResults, sources)
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
		var sources []fantasy.SourceContent
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
				case fantasy.SourceContent:
					sources = append(sources, v)
				}
			}
		}
		require.NotEmpty(t, providerToolCalls, "should have provider-executed tool calls")
		require.Equal(t, "web_search", providerToolCalls[0].ToolName)
		require.NotEmpty(t, providerToolResults, "should have provider-executed tool results")
		requireWebSearchActionMetadata(t, providerToolResults, sources)
	})
}

// requireWebSearchActionMetadata checks that web_search results carry the
// search queries, and that every page a search found is a source tagged
// with that search's ID. The recorded gpt-4.1 stream reports no found pages
// on its search item, so found pages are not required.
func requireWebSearchActionMetadata(t *testing.T, results []fantasy.ToolResultContent, sources []fantasy.SourceContent) {
	t.Helper()

	require.NotEmpty(t, results, "should have provider-executed tool results")
	for _, result := range results {
		meta, ok := result.ProviderMetadata[openai.Name].(*openai.WebSearchCallMetadata)
		require.True(t, ok, "web_search result metadata should be *openai.WebSearchCallMetadata, got %T", result.ProviderMetadata[openai.Name])
		require.NotNil(t, meta.Action)
		require.Equal(t, "search", meta.Action.Type)
		require.NotEmpty(t, meta.Action.Queries, "search action should report its queries")
		for _, found := range meta.Action.Sources {
			require.True(t, slices.ContainsFunc(sources, func(source fantasy.SourceContent) bool {
				return source.URL == found.URL && source.ToolCallID == meta.ItemID
			}), "found page %q should be a source tagged with %q", found.URL, meta.ItemID)
		}
	}
}
