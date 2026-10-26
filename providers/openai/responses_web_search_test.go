package openai

import (
	"context"
	"encoding/json"
	"slices"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

// webSearchCallActionCases covers the web_search_call action shapes
// OpenAI returns: searches with queries and found pages, older responses
// with the deprecated single query, searches that report no query, and the
// open_page and find_in_page actions of reasoning models. Status defaults
// to "completed".
type webSearchCallActionCase struct {
	name      string
	status    string
	action    map[string]any
	want      *WebSearchAction
	wantInput string
	wantFound []string
}

var webSearchCallActionCases = []webSearchCallActionCase{
	{
		name: "QueriesAndSources",
		action: map[string]any{
			"type":    "search",
			"queries": []any{"coder agents web search", "openai responses sources"},
			"sources": []any{
				map[string]any{"type": "url", "url": "https://coder.com/docs/ai-coder"},
				map[string]any{"type": "url", "url": "https://developers.openai.com/api/docs/guides/tools-web-search"},
			},
		},
		want: &WebSearchAction{
			Type:    "search",
			Queries: []string{"coder agents web search", "openai responses sources"},
			Sources: []WebSearchSource{
				{Type: "url", URL: "https://coder.com/docs/ai-coder"},
				{Type: "url", URL: "https://developers.openai.com/api/docs/guides/tools-web-search"},
			},
		},
		wantInput: `{"type":"search","queries":["coder agents web search","openai responses sources"]}`,
		wantFound: []string{"https://coder.com/docs/ai-coder", "https://developers.openai.com/api/docs/guides/tools-web-search"},
	},
	{
		name: "DeprecatedQuery",
		action: map[string]any{
			"type":  "search",
			"query": "latest AI news",
		},
		want: &WebSearchAction{
			Type:  "search",
			Query: "latest AI news",
		},
		wantInput: `{"type":"search","queries":["latest AI news"]}`,
	},
	{
		name: "NoQueries",
		action: map[string]any{
			"type": "search",
			"sources": []any{
				map[string]any{"type": "url", "url": "https://example.com/weather"},
			},
		},
		want: &WebSearchAction{
			Type: "search",
			Sources: []WebSearchSource{
				{Type: "url", URL: "https://example.com/weather"},
			},
		},
		wantInput: `{"type":"search"}`,
		wantFound: []string{"https://example.com/weather"},
	},
	{
		name: "OpenPage",
		action: map[string]any{
			"type": "open_page",
			"url":  "https://go.dev/dl/",
		},
		want: &WebSearchAction{
			Type: "open_page",
			URL:  "https://go.dev/dl/",
		},
		wantInput: `{"type":"open_page","url":"https://go.dev/dl/"}`,
	},
	{
		name: "FindInPage",
		action: map[string]any{
			"type":    "find_in_page",
			"url":     "https://go.dev/doc/devel/release",
			"pattern": "go1.27",
		},
		want: &WebSearchAction{
			Type:    "find_in_page",
			URL:     "https://go.dev/doc/devel/release",
			Pattern: "go1.27",
		},
		wantInput: `{"type":"find_in_page","url":"https://go.dev/doc/devel/release","pattern":"go1.27"}`,
	},
	{
		name:   "Failed",
		status: "failed",
		action: map[string]any{
			"type":    "search",
			"queries": []any{"tokyo population"},
		},
		want: &WebSearchAction{
			Type:    "search",
			Queries: []string{"tokyo population"},
		},
		wantInput: `{"type":"search","queries":["tokyo population"]}`,
	},
}

func (tc webSearchCallActionCase) itemStatus() string {
	if tc.status == "" {
		return "completed"
	}
	return tc.status
}

func requireWebSearchCallMetadata(t *testing.T, metadata fantasy.ProviderMetadata, wantStatus string, want *WebSearchAction) {
	t.Helper()

	wsMeta, ok := metadata[Name].(*WebSearchCallMetadata)
	require.True(t, ok, "metadata should be *WebSearchCallMetadata, got %T", metadata[Name])
	require.Equal(t, "ws_01", wsMeta.ItemID)
	require.Equal(t, wantStatus, wsMeta.Status)
	require.Equal(t, want, wsMeta.Action)

	// Callers persist the metadata as JSON and read it back on later
	// turns, so the action fields must survive the round trip.
	encoded, err := json.Marshal(metadata)
	require.NoError(t, err)
	var raw map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(encoded, &raw))
	decoded, err := fantasy.UnmarshalProviderMetadata(raw)
	require.NoError(t, err)
	roundTripped, ok := decoded[Name].(*WebSearchCallMetadata)
	require.True(t, ok, "decoded metadata should be *WebSearchCallMetadata, got %T", decoded[Name])
	require.Equal(t, wsMeta, roundTripped)
}

func requireWebSearchSourcesInclude(t *testing.T, body map[string]any) {
	t.Helper()

	include, ok := body["include"].([]any)
	require.True(t, ok, "request body should have an include array, got %#v", body["include"])
	count := 0
	for _, value := range include {
		if value == string(IncludeWebSearchCallActionSources) {
			count++
		}
	}
	require.Equal(t, 1, count, "include should request web_search_call.action.sources exactly once: %v", include)
}

func TestResponsesGenerate_WebSearchCallAction(t *testing.T) {
	t.Parallel()

	for _, tc := range webSearchCallActionCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			server := newMockServer()
			defer server.close()
			server.response = map[string]any{
				"id":     "resp_01",
				"object": "response",
				"model":  "gpt-4.1",
				"status": "completed",
				"output": []any{
					map[string]any{
						"type":   "web_search_call",
						"id":     "ws_01",
						"status": tc.itemStatus(),
						"action": tc.action,
					},
					map[string]any{
						"type":   "message",
						"id":     "msg_01",
						"role":   "assistant",
						"status": "completed",
						"content": []any{
							map[string]any{
								"type": "output_text",
								"text": "Answer.",
								"annotations": []any{
									map[string]any{
										"type":        "url_citation",
										"url":         "https://example.com/cited",
										"title":       "Cited",
										"start_index": 0,
										"end_index":   7,
									},
								},
							},
						},
					},
				},
				"usage": map[string]any{"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
			}

			model := newResponsesProvider(t, server.server.URL)
			resp, err := model.Generate(context.Background(), fantasy.Call{
				Prompt: testPrompt,
				Tools:  []fantasy.Tool{WebSearchTool(nil)},
			})
			require.NoError(t, err)
			require.Len(t, server.calls, 1)
			requireWebSearchSourcesInclude(t, server.calls[0].body)

			var (
				toolCalls   []fantasy.ToolCallContent
				toolResults []fantasy.ToolResultContent
				sources     []fantasy.SourceContent
			)
			for _, c := range resp.Content {
				switch v := c.(type) {
				case fantasy.ToolCallContent:
					toolCalls = append(toolCalls, v)
				case fantasy.ToolResultContent:
					toolResults = append(toolResults, v)
				case fantasy.SourceContent:
					sources = append(sources, v)
				}
			}

			require.Len(t, toolCalls, 1)
			require.True(t, toolCalls[0].ProviderExecuted)
			require.JSONEq(t, tc.wantInput, toolCalls[0].Input)
			require.Len(t, toolResults, 1)
			require.True(t, toolResults[0].ProviderExecuted)
			require.Equal(t, "web_search", toolResults[0].ToolName)
			requireWebSearchCallMetadata(t, toolResults[0].ProviderMetadata, tc.itemStatus(), tc.want)

			var found, cited []string
			for _, source := range sources {
				switch source.ToolCallID {
				case "ws_01":
					found = append(found, source.URL)
				case "":
					cited = append(cited, source.URL)
				default:
					t.Fatalf("source %q tagged with unknown tool call %q", source.URL, source.ToolCallID)
				}
			}
			require.Equal(t, tc.wantFound, found)
			require.Equal(t, []string{"https://example.com/cited"}, cited)
		})
	}
}

func TestResponsesStream_WebSearchCallAction(t *testing.T) {
	t.Parallel()

	for _, tc := range webSearchCallActionCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			item, err := json.Marshal(map[string]any{
				"type":   "web_search_call",
				"id":     "ws_01",
				"status": tc.itemStatus(),
				"action": tc.action,
			})
			require.NoError(t, err)

			sms := newStreamingMockServer()
			defer sms.close()
			sms.chunks = []string{
				responsesSSEEvent("response.output_item.added",
					`{"type":"response.output_item.added","output_index":0,"item":{"type":"web_search_call","id":"ws_01","status":"in_progress"}}`),
				responsesSSEEvent("response.output_item.done",
					`{"type":"response.output_item.done","output_index":0,"item":`+string(item)+`}`),
				responsesSSEEvent("response.output_item.added",
					`{"type":"response.output_item.added","output_index":1,"item":{"type":"message","id":"msg_01","role":"assistant","status":"in_progress","content":[]}}`),
				responsesSSEEvent("response.output_text.delta",
					`{"type":"response.output_text.delta","item_id":"msg_01","output_index":1,"content_index":0,"delta":"Answer."}`),
				responsesSSEEvent("response.output_text.annotation.added",
					`{"type":"response.output_text.annotation.added","annotation":{"type":"url_citation","url":"https://example.com/cited","title":"Cited","start_index":0,"end_index":7},"annotation_index":0,"content_index":0,"item_id":"msg_01","output_index":1,"sequence_number":5}`),
				responsesSSEEvent("response.output_item.done",
					`{"type":"response.output_item.done","output_index":1,"item":{"type":"message","id":"msg_01","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Answer.","annotations":[]}]}}`),
				responsesSSEEvent("response.completed",
					`{"type":"response.completed","response":{"id":"resp_01","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}`),
			}

			model := newResponsesProvider(t, sms.server.URL)
			stream, err := model.Stream(context.Background(), fantasy.Call{
				Prompt: testPrompt,
				Tools:  []fantasy.Tool{WebSearchTool(nil)},
			})
			require.NoError(t, err)

			var toolCalls, toolResults, sources []fantasy.StreamPart
			for part := range stream {
				require.NotEqual(t, fantasy.StreamPartTypeError, part.Type, "unexpected stream error: %v", part.Error)
				switch part.Type {
				case fantasy.StreamPartTypeToolCall:
					toolCalls = append(toolCalls, part)
				case fantasy.StreamPartTypeToolResult:
					toolResults = append(toolResults, part)
				case fantasy.StreamPartTypeSource:
					sources = append(sources, part)
				}
			}
			require.Len(t, sms.calls, 1)
			requireWebSearchSourcesInclude(t, sms.calls[0].body)

			require.Len(t, toolCalls, 1)
			require.True(t, toolCalls[0].ProviderExecuted)
			require.JSONEq(t, tc.wantInput, toolCalls[0].ToolCallInput)
			require.Len(t, toolResults, 1)
			require.True(t, toolResults[0].ProviderExecuted)
			require.Equal(t, "web_search", toolResults[0].ToolCallName)
			requireWebSearchCallMetadata(t, toolResults[0].ProviderMetadata, tc.itemStatus(), tc.want)

			var found, cited []string
			for _, source := range sources {
				switch source.SourceToolCallID {
				case "ws_01":
					found = append(found, source.URL)
				case "":
					cited = append(cited, source.URL)
				default:
					t.Fatalf("source %q tagged with unknown tool call %q", source.URL, source.SourceToolCallID)
				}
			}
			require.Equal(t, tc.wantFound, found)
			require.Equal(t, []string{"https://example.com/cited"}, cited)
		})
	}
}

// TestResponsesStream_WebSearchCallFinishesWithItem covers live gpt-4.1-mini
// and gpt-5-mini streams: a search's found pages are on its output_item.done,
// and the final response summary also lists pages the answer cited among
// them. gpt-4.1 search items carried no pages, and its summary listed only
// cited pages. The search finishes, with its own pages, before the answer
// streams.
func TestResponsesStream_WebSearchCallFinishesWithItem(t *testing.T) {
	t.Parallel()

	sms := newStreamingMockServer()
	defer sms.close()
	sms.chunks = []string{
		responsesSSEEvent("response.output_item.added",
			`{"type":"response.output_item.added","output_index":0,"item":{"type":"web_search_call","id":"ws_01","status":"in_progress"}}`),
		responsesSSEEvent("response.output_item.done",
			`{"type":"response.output_item.done","output_index":0,"item":{"type":"web_search_call","id":"ws_01","status":"completed","action":{"type":"search","queries":["tokyo population"],"sources":[{"type":"url","url":"https://www.metro.tokyo.lg.jp/"}]}}}`),
		responsesSSEEvent("response.output_item.added",
			`{"type":"response.output_item.added","output_index":1,"item":{"type":"message","id":"msg_01","role":"assistant","status":"in_progress","content":[]}}`),
		responsesSSEEvent("response.output_text.delta",
			`{"type":"response.output_text.delta","item_id":"msg_01","output_index":1,"content_index":0,"delta":"About 14 million."}`),
		responsesSSEEvent("response.output_text.annotation.added",
			`{"type":"response.output_text.annotation.added","annotation":{"type":"url_citation","url":"https://example.com/tokyo","title":"Tokyo","start_index":0,"end_index":5},"annotation_index":0,"content_index":0,"item_id":"msg_01","output_index":1}`),
		responsesSSEEvent("response.output_item.done",
			`{"type":"response.output_item.done","output_index":1,"item":{"type":"message","id":"msg_01","role":"assistant","status":"completed","content":[{"type":"output_text","text":"About 14 million.","annotations":[]}]}}`),
		responsesSSEEvent("response.completed",
			`{"type":"response.completed","response":{"id":"resp_01","status":"completed","output":[`+
				`{"type":"web_search_call","id":"ws_01","status":"completed","action":{"type":"search","queries":["tokyo population"],"sources":[{"type":"url","url":"https://www.metro.tokyo.lg.jp/"},{"type":"url","url":"https://example.com/tokyo"}]}},`+
				`{"type":"message","id":"msg_01","role":"assistant","status":"completed","content":[{"type":"output_text","text":"About 14 million.","annotations":[]}]}`+
				`],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}`),
	}

	model := newResponsesProvider(t, sms.server.URL)
	stream, err := model.Stream(context.Background(), fantasy.Call{
		Prompt: testPrompt,
		Tools:  []fantasy.Tool{WebSearchTool(nil)},
	})
	require.NoError(t, err)

	var types []fantasy.StreamPartType
	var found, cited []string
	for part := range stream {
		require.NotEqual(t, fantasy.StreamPartTypeError, part.Type, "unexpected stream error: %v", part.Error)
		types = append(types, part.Type)
		if part.Type == fantasy.StreamPartTypeSource {
			if part.SourceToolCallID == "ws_01" {
				found = append(found, part.URL)
			} else {
				cited = append(cited, part.URL)
			}
		}
	}

	require.Less(t, slices.Index(types, fantasy.StreamPartTypeToolCall), slices.Index(types, fantasy.StreamPartTypeSource))
	require.Less(t, slices.Index(types, fantasy.StreamPartTypeToolResult), slices.Index(types, fantasy.StreamPartTypeTextDelta))
	require.Equal(t, []string{"https://www.metro.tokyo.lg.jp/"}, found)
	require.Equal(t, []string{"https://example.com/tokyo"}, cited)
}

func TestResponsesWebSearchSourcesInclude(t *testing.T) {
	t.Parallel()

	t.Run("NotRequestedWithoutWebSearchTool", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()
		server.response = mockResponsesWebSearchResponse()

		model := newResponsesProvider(t, server.server.URL)
		_, err := model.Generate(context.Background(), fantasy.Call{Prompt: testPrompt})
		require.NoError(t, err)
		require.Len(t, server.calls, 1)

		include, _ := server.calls[0].body["include"].([]any)
		require.NotContains(t, include, string(IncludeWebSearchCallActionSources))
	})

	t.Run("KeepsCallerIncludesWithoutDuplicates", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()
		server.response = mockResponsesWebSearchResponse()

		model := newResponsesProvider(t, server.server.URL)
		_, err := model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
			Tools:  []fantasy.Tool{WebSearchTool(nil)},
			ProviderOptions: fantasy.ProviderOptions{
				Name: &ResponsesProviderOptions{
					Include: []IncludeType{
						IncludeReasoningEncryptedContent,
						IncludeWebSearchCallActionSources,
					},
				},
			},
		})
		require.NoError(t, err)
		require.Len(t, server.calls, 1)
		requireWebSearchSourcesInclude(t, server.calls[0].body)
		require.Contains(t, server.calls[0].body["include"], string(IncludeReasoningEncryptedContent))
	})
}
