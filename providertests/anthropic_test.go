package providertests

import (
	"context"
	"encoding/json"
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
			MaxOutputTokens: new(int64(4000)),
		})
		require.NoError(t, err)

		got2 := result2.Response.Content.Text()
		require.NotEmpty(t, got2, "turn 2 should have a text response")
		require.Contains(t, got2, "Osaka", "turn 2 response should mention Osaka")
	})
}

// screenshotBase64 is a tiny valid 1x1 PNG encoded as base64,
// used as a stub screenshot result in computer use tests.
const screenshotBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

func TestAnthropicComputerUse(t *testing.T) {
	type computerUseModel struct {
		name        string
		model       string
		toolVersion anthropic.ComputerUseToolVersion
	}
	computerUseModels := []computerUseModel{
		{"claude-sonnet-4", "claude-sonnet-4-20250514", anthropic.ComputerUse20250124},
		{"claude-opus-4-6", "claude-opus-4-6", anthropic.ComputerUse20251124},
	}
	for _, m := range computerUseModels {
		t.Run(m.name, func(t *testing.T) {
			t.Run("computer use", func(t *testing.T) {
				r := vcr.NewRecorder(t)

				model, err := anthropicBuilder(m.model)(t, r)
				require.NoError(t, err)

				cuTool := jsonRoundTripTool(t, anthropic.NewComputerUseTool(anthropic.ComputerUseToolOptions{
					DisplayWidthPx:  1920,
					DisplayHeightPx: 1080,
					ToolVersion:     m.toolVersion,
				}, noopComputerRun))

				// First call: expect a screenshot tool call.
				resp, err := model.Generate(t.Context(), fantasy.Call{
					Prompt: fantasy.Prompt{
						{Role: fantasy.MessageRoleSystem, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "You are a helpful assistant"}}},
						{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "Take a screenshot of the desktop"}}},
					},
					Tools: []fantasy.Tool{cuTool},
				})
				require.NoError(t, err)
				require.Equal(t, fantasy.FinishReasonToolCalls, resp.FinishReason)

				toolCalls := resp.Content.ToolCalls()
				require.Len(t, toolCalls, 1)
				require.Equal(t, "computer", toolCalls[0].ToolName)
				require.Contains(t, toolCalls[0].Input, "screenshot")

				// Second call: send the tool result back, expect text.
				resp2, err := model.Generate(t.Context(), fantasy.Call{
					Prompt: fantasy.Prompt{
						{Role: fantasy.MessageRoleSystem, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "You are a helpful assistant"}}},
						{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "Take a screenshot of the desktop"}}},
						{
							Role: fantasy.MessageRoleAssistant,
							Content: []fantasy.MessagePart{
								fantasy.ToolCallPart{
									ToolCallID: toolCalls[0].ToolCallID,
									ToolName:   toolCalls[0].ToolName,
									Input:      toolCalls[0].Input,
								},
							},
						},
						{
							Role: fantasy.MessageRoleTool,
							Content: []fantasy.MessagePart{
								fantasy.ToolResultPart{
									ToolCallID: toolCalls[0].ToolCallID,
									Output: fantasy.ToolResultOutputContentMedia{
										Data:      screenshotBase64,
										MediaType: "image/png",
									},
								},
							},
						},
					},
					Tools: []fantasy.Tool{cuTool},
				})
				require.NoError(t, err)
				require.NotEmpty(t, resp2.Content.Text())
				require.Contains(t, resp2.Content.Text(), "desktop")
			})

			t.Run("computer use streaming", func(t *testing.T) {
				r := vcr.NewRecorder(t)

				model, err := anthropicBuilder(m.model)(t, r)
				require.NoError(t, err)

				cuTool := jsonRoundTripTool(t, anthropic.NewComputerUseTool(anthropic.ComputerUseToolOptions{
					DisplayWidthPx:  1920,
					DisplayHeightPx: 1080,
					ToolVersion:     m.toolVersion,
				}, noopComputerRun))

				// First call: stream, collect tool call.
				stream, err := model.Stream(t.Context(), fantasy.Call{
					Prompt: fantasy.Prompt{
						{Role: fantasy.MessageRoleSystem, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "You are a helpful assistant"}}},
						{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "Take a screenshot of the desktop"}}},
					},
					Tools: []fantasy.Tool{cuTool},
				})
				require.NoError(t, err)

				var toolCallID, toolCallName, toolCallInput string
				var finishReason fantasy.FinishReason
				stream(func(part fantasy.StreamPart) bool {
					switch part.Type {
					case fantasy.StreamPartTypeToolCall:
						toolCallID = part.ID
						toolCallName = part.ToolCallName
						toolCallInput = part.ToolCallInput
					case fantasy.StreamPartTypeFinish:
						finishReason = part.FinishReason
					}
					return true
				})

				require.Equal(t, fantasy.FinishReasonToolCalls, finishReason)
				require.Equal(t, "computer", toolCallName)
				require.Contains(t, toolCallInput, "screenshot")

				// Second call: send tool result, stream text back.
				stream2, err := model.Stream(t.Context(), fantasy.Call{
					Prompt: fantasy.Prompt{
						{Role: fantasy.MessageRoleSystem, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "You are a helpful assistant"}}},
						{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "Take a screenshot of the desktop"}}},
						{
							Role: fantasy.MessageRoleAssistant,
							Content: []fantasy.MessagePart{
								fantasy.ToolCallPart{
									ToolCallID: toolCallID,
									ToolName:   toolCallName,
									Input:      toolCallInput,
								},
							},
						},
						{
							Role: fantasy.MessageRoleTool,
							Content: []fantasy.MessagePart{
								fantasy.ToolResultPart{
									ToolCallID: toolCallID,
									Output: fantasy.ToolResultOutputContentMedia{
										Data:      screenshotBase64,
										MediaType: "image/png",
									},
								},
							},
						},
					},
					Tools: []fantasy.Tool{cuTool},
				})
				require.NoError(t, err)

				var text string
				stream2(func(part fantasy.StreamPart) bool {
					if part.Type == fantasy.StreamPartTypeTextDelta {
						text += part.Delta
					}
					return true
				})
				require.NotEmpty(t, text)
				require.Contains(t, text, "desktop")
			})
		})
	}
}

// noopComputerRun is a no-op run function for tests that only need
// to inspect the tool definition, not execute it.
var noopComputerRun = func(_ context.Context, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
	return fantasy.ToolResponse{}, nil
}

// jsonRoundTripTool simulates a JSON round-trip on a ProviderDefinedTool
// so numeric values become float64 as they would in real usage.
func jsonRoundTripTool(t *testing.T, tool fantasy.ExecutableProviderTool) fantasy.ProviderDefinedTool {
	t.Helper()
	pdt := tool.Definition()
	data, err := json.Marshal(pdt.Args)
	require.NoError(t, err)
	var args map[string]any
	require.NoError(t, json.Unmarshal(data, &args))
	pdt.Args = args
	return pdt
}
