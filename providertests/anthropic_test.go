package providertests

import (
	"context"
	"net/http"
	"os"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/anthropic"
	"charm.land/x/vcr"
	"github.com/stretchr/testify/require"
)

var anthropicTestModels = []testModel{
	{"claude-sonnet-4", "claude-sonnet-4-20250514", true},
}

func TestAnthropicCommon(t *testing.T) {
	var pairs []builderPair
	for _, m := range anthropicTestModels {
		pairs = append(pairs, builderPair{m.name, anthropicBuilder(m.model), nil, nil})
	}
	testCommon(t, pairs)
}

func addAnthropicCaching(ctx context.Context, options fantasy.PrepareStepFunctionOptions) (context.Context, fantasy.PrepareStepResult, error) {
	prepared := fantasy.PrepareStepResult{}
	prepared.Messages = options.Messages

	for i := range prepared.Messages {
		prepared.Messages[i].ProviderOptions = nil
	}
	providerOption := fantasy.ProviderOptions{
		anthropic.Name: &anthropic.ProviderCacheControlOptions{
			CacheControl: anthropic.CacheControl{Type: "ephemeral"},
		},
	}

	lastSystemRoleInx := 0
	systemMessageUpdated := false
	for i, msg := range prepared.Messages {
		// only add cache control to the last message
		if msg.Role == fantasy.MessageRoleSystem {
			lastSystemRoleInx = i
		} else if !systemMessageUpdated {
			prepared.Messages[lastSystemRoleInx].ProviderOptions = providerOption
			systemMessageUpdated = true
		}
		// than add cache control to the last 2 messages
		if i > len(prepared.Messages)-3 {
			prepared.Messages[i].ProviderOptions = providerOption
		}
	}
	return ctx, prepared, nil
}

func TestAnthropicCommonWithCacheControl(t *testing.T) {
	var pairs []builderPair
	for _, m := range anthropicTestModels {
		pairs = append(pairs, builderPair{m.name, anthropicBuilder(m.model), nil, addAnthropicCaching})
	}
	testCommon(t, pairs)
}

func TestAnthropicThinking(t *testing.T) {
	opts := fantasy.ProviderOptions{
		anthropic.Name: &anthropic.ProviderOptions{
			Thinking: &anthropic.ThinkingProviderOption{
				BudgetTokens: 4000,
			},
		},
	}
	var pairs []builderPair
	for _, m := range anthropicTestModels {
		if !m.reasoning {
			continue
		}
		pairs = append(pairs, builderPair{m.name, anthropicBuilder(m.model), opts, nil})
	}
	testThinking(t, pairs, testAnthropicThinking)
}

func TestAnthropicThinkingWithCacheControl(t *testing.T) {
	opts := fantasy.ProviderOptions{
		anthropic.Name: &anthropic.ProviderOptions{
			Thinking: &anthropic.ThinkingProviderOption{
				BudgetTokens: 4000,
			},
		},
	}
	var pairs []builderPair
	for _, m := range anthropicTestModels {
		if !m.reasoning {
			continue
		}
		pairs = append(pairs, builderPair{m.name, anthropicBuilder(m.model), opts, addAnthropicCaching})
	}
	testThinking(t, pairs, testAnthropicThinking)
}

func TestAnthropicObjectGeneration(t *testing.T) {
	var pairs []builderPair
	for _, m := range anthropicTestModels {
		pairs = append(pairs, builderPair{m.name, anthropicBuilder(m.model), nil, nil})
	}
	testObjectGeneration(t, pairs)
}

func testAnthropicThinking(t *testing.T, result *fantasy.AgentResult) {
	reasoningContentCount := 0
	signaturesCount := 0
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

					anthropicReasoningMetadata, ok := reasoningContent.ProviderOptions[anthropic.Name]
					if !ok {
						continue
					}
					if reasoningContent.Text != "" {
						if typed, ok := anthropicReasoningMetadata.(*anthropic.ReasoningOptionMetadata); ok {
							require.NotEmpty(t, typed.Signature)
							signaturesCount += 1
						}
					}
				}
			}
		}
	}
	require.Greater(t, reasoningContentCount, 0)
	require.Greater(t, signaturesCount, 0)
	require.Equal(t, reasoningContentCount, signaturesCount)
}

func anthropicBuilder(model string) builderFunc {
	return func(t *testing.T, r *vcr.Recorder) (fantasy.LanguageModel, error) {
		provider, err := anthropic.New(
			anthropic.WithAPIKey(os.Getenv("FANTASY_ANTHROPIC_API_KEY")),
			anthropic.WithHTTPClient(&http.Client{Transport: r}),
		)
		if err != nil {
			return nil, err
		}
		return provider.LanguageModel(t.Context(), model)
	}
}

// TestAnthropicWebSearch tests web search tool support via the agent
// using WithProviderDefinedTools.
func TestAnthropicWebSearch(t *testing.T) {
	model := "claude-sonnet-4-20250514"
	webSearchTool := anthropic.WebSearchTool(nil)

	t.Run("generate", func(t *testing.T) {
		r := vcr.NewRecorder(t)

		lm, err := anthropicBuilder(model)(t, r)
		require.NoError(t, err)

		agent := fantasy.NewAgent(
			lm,
			fantasy.WithSystemPrompt("You are a helpful assistant"),
			fantasy.WithProviderDefinedTools(webSearchTool),
		)

		result, err := agent.Generate(t.Context(), fantasy.AgentCall{
			Prompt:          "What is the current population of Tokyo? Cite your source.",
			MaxOutputTokens: fantasy.Opt(int64(4000)),
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
		require.NotEmpty(t, sources, "should have source citations")
		require.NotEmpty(t, sources[0].URL, "source should have a URL")
	})

	t.Run("stream", func(t *testing.T) {
		r := vcr.NewRecorder(t)

		lm, err := anthropicBuilder(model)(t, r)
		require.NoError(t, err)

		agent := fantasy.NewAgent(
			lm,
			fantasy.WithSystemPrompt("You are a helpful assistant"),
			fantasy.WithProviderDefinedTools(webSearchTool),
		)

		// Turn 1: initial query triggers web search.
		result, err := agent.Stream(t.Context(), fantasy.AgentStreamCall{
			Prompt:          "What is the current population of Tokyo? Cite your source.",
			MaxOutputTokens: fantasy.Opt(int64(4000)),
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

		// Turn 2: follow-up using step messages from turn 1.
		// This verifies that the web_search_tool_result block
		// round-trips correctly through toPrompt.
		var history fantasy.Prompt
		history = append(history, fantasy.Message{
			Role:    fantasy.MessageRoleUser,
			Content: []fantasy.MessagePart{fantasy.TextPart{Text: "What is the current population of Tokyo? Cite your source."}},
		})
		for _, step := range result.Steps {
			history = append(history, step.Messages...)
		}

		result2, err := agent.Stream(t.Context(), fantasy.AgentStreamCall{
			Messages:        history,
			Prompt:          "How does that compare to Osaka?",
			MaxOutputTokens: fantasy.Opt(int64(4000)),
		})
		require.NoError(t, err)

		got2 := result2.Response.Content.Text()
		require.NotEmpty(t, got2, "turn 2 should have a text response")
		require.Contains(t, got2, "Osaka", "turn 2 response should mention Osaka")
	})
}
