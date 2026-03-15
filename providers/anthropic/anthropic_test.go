package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"charm.land/fantasy"
	"github.com/charmbracelet/anthropic-sdk-go"
	"github.com/stretchr/testify/require"
)

func TestToPrompt_DropsEmptyMessages(t *testing.T) {
	t.Parallel()

	t.Run("should drop assistant messages with only reasoning content", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Hello"},
				},
			},
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.ReasoningPart{
						Text: "Let me think about this...",
						ProviderOptions: fantasy.ProviderOptions{
							Name: &ReasoningOptionMetadata{
								Signature: "abc123",
							},
						},
					},
				},
			},
		}

		systemBlocks, messages, warnings := toPrompt(prompt, true)

		require.Empty(t, systemBlocks)
		require.Len(t, messages, 1, "should only have user message, assistant message should be dropped")
		require.Len(t, warnings, 1)
		require.Equal(t, fantasy.CallWarningTypeOther, warnings[0].Type)
		require.Contains(t, warnings[0].Message, "dropping empty assistant message")
		require.Contains(t, warnings[0].Message, "neither user-facing content nor tool calls")
	})

	t.Run("should drop assistant reasoning when sendReasoning disabled", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Hello"},
				},
			},
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.ReasoningPart{
						Text: "Let me think about this...",
						ProviderOptions: fantasy.ProviderOptions{
							Name: &ReasoningOptionMetadata{
								Signature: "def456",
							},
						},
					},
				},
			},
		}

		systemBlocks, messages, warnings := toPrompt(prompt, false)

		require.Empty(t, systemBlocks)
		require.Len(t, messages, 1, "should only have user message, assistant message should be dropped")
		require.Len(t, warnings, 2)
		require.Equal(t, fantasy.CallWarningTypeOther, warnings[0].Type)
		require.Contains(t, warnings[0].Message, "sending reasoning content is disabled")
		require.Equal(t, fantasy.CallWarningTypeOther, warnings[1].Type)
		require.Contains(t, warnings[1].Message, "dropping empty assistant message")
	})

	t.Run("should drop truly empty assistant messages", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Hello"},
				},
			},
			{
				Role:    fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{},
			},
		}

		systemBlocks, messages, warnings := toPrompt(prompt, true)

		require.Empty(t, systemBlocks)
		require.Len(t, messages, 1, "should only have user message")
		require.Len(t, warnings, 1)
		require.Equal(t, fantasy.CallWarningTypeOther, warnings[0].Type)
		require.Contains(t, warnings[0].Message, "dropping empty assistant message")
	})

	t.Run("should keep assistant messages with text content", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Hello"},
				},
			},
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Hi there!"},
				},
			},
		}

		systemBlocks, messages, warnings := toPrompt(prompt, true)

		require.Empty(t, systemBlocks)
		require.Len(t, messages, 2, "should have both user and assistant messages")
		require.Empty(t, warnings)
	})

	t.Run("should keep assistant messages with tool calls", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "What's the weather?"},
				},
			},
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.ToolCallPart{
						ToolCallID: "call_123",
						ToolName:   "get_weather",
						Input:      `{"location":"NYC"}`,
					},
				},
			},
		}

		systemBlocks, messages, warnings := toPrompt(prompt, true)

		require.Empty(t, systemBlocks)
		require.Len(t, messages, 2, "should have both user and assistant messages")
		require.Empty(t, warnings)
	})

	t.Run("should drop assistant messages with invalid tool input", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Hi"},
				},
			},
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.ToolCallPart{
						ToolCallID: "call_123",
						ToolName:   "get_weather",
						Input:      "{not-json",
					},
				},
			},
		}

		systemBlocks, messages, warnings := toPrompt(prompt, true)

		require.Empty(t, systemBlocks)
		require.Len(t, messages, 1, "should only have user message")
		require.Len(t, warnings, 1)
		require.Equal(t, fantasy.CallWarningTypeOther, warnings[0].Type)
		require.Contains(t, warnings[0].Message, "dropping empty assistant message")
	})

	t.Run("should keep assistant messages with reasoning and text", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Hello"},
				},
			},
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.ReasoningPart{
						Text: "Let me think...",
						ProviderOptions: fantasy.ProviderOptions{
							Name: &ReasoningOptionMetadata{
								Signature: "abc123",
							},
						},
					},
					fantasy.TextPart{Text: "Hi there!"},
				},
			},
		}

		systemBlocks, messages, warnings := toPrompt(prompt, true)

		require.Empty(t, systemBlocks)
		require.Len(t, messages, 2, "should have both user and assistant messages")
		require.Empty(t, warnings)
	})

	t.Run("should keep user messages with image content", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.FilePart{
						Data:      []byte{0x01, 0x02, 0x03},
						MediaType: "image/png",
					},
				},
			},
		}

		systemBlocks, messages, warnings := toPrompt(prompt, true)

		require.Empty(t, systemBlocks)
		require.Len(t, messages, 1)
		require.Empty(t, warnings)
	})

	t.Run("should drop user messages without visible content", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.FilePart{
						Data:      []byte("not supported"),
						MediaType: "application/pdf",
					},
				},
			},
		}

		systemBlocks, messages, warnings := toPrompt(prompt, true)

		require.Empty(t, systemBlocks)
		require.Empty(t, messages)
		require.Len(t, warnings, 1)
		require.Equal(t, fantasy.CallWarningTypeOther, warnings[0].Type)
		require.Contains(t, warnings[0].Message, "dropping empty user message")
		require.Contains(t, warnings[0].Message, "neither user-facing content nor tool results")
	})

	t.Run("should keep user messages with tool results", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleTool,
				Content: []fantasy.MessagePart{
					fantasy.ToolResultPart{
						ToolCallID: "call_123",
						Output:     fantasy.ToolResultOutputContentText{Text: "done"},
					},
				},
			},
		}

		systemBlocks, messages, warnings := toPrompt(prompt, true)

		require.Empty(t, systemBlocks)
		require.Len(t, messages, 1)
		require.Empty(t, warnings)
	})

	t.Run("should keep user messages with tool error results", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleTool,
				Content: []fantasy.MessagePart{
					fantasy.ToolResultPart{
						ToolCallID: "call_456",
						Output:     fantasy.ToolResultOutputContentError{Error: errors.New("boom")},
					},
				},
			},
		}

		systemBlocks, messages, warnings := toPrompt(prompt, true)

		require.Empty(t, systemBlocks)
		require.Len(t, messages, 1)
		require.Empty(t, warnings)
	})

	t.Run("should keep user messages with tool media results", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleTool,
				Content: []fantasy.MessagePart{
					fantasy.ToolResultPart{
						ToolCallID: "call_789",
						Output: fantasy.ToolResultOutputContentMedia{
							Data:      "AQID",
							MediaType: "image/png",
						},
					},
				},
			},
		}

		systemBlocks, messages, warnings := toPrompt(prompt, true)

		require.Empty(t, systemBlocks)
		require.Len(t, messages, 1)
		require.Empty(t, warnings)
	})
}

func TestParseContextTooLargeError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		message  string
		wantErr  bool
		wantUsed int
		wantMax  int
	}{
		{
			name:     "matches anthropic format",
			message:  "prompt is too long: 202630 tokens > 200000 maximum",
			wantErr:  true,
			wantUsed: 202630,
			wantMax:  200000,
		},
		{
			name:     "matches with different numbers",
			message:  "prompt is too long: 150000 tokens > 128000 maximum",
			wantErr:  true,
			wantUsed: 150000,
			wantMax:  128000,
		},
		{
			name:     "matches with extra whitespace",
			message:  "prompt is too long:  202630  tokens  >  200000  maximum",
			wantErr:  true,
			wantUsed: 202630,
			wantMax:  200000,
		},
		{
			name:    "does not match unrelated error",
			message: "invalid api key",
			wantErr: false,
		},
		{
			name:    "does not match rate limit error",
			message: "rate limit exceeded",
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			providerErr := &fantasy.ProviderError{Message: tt.message}
			parseContextTooLargeError(tt.message, providerErr)

			if tt.wantErr {
				require.True(t, providerErr.IsContextTooLarge())
				require.Equal(t, tt.wantUsed, providerErr.ContextUsedTokens)
				require.Equal(t, tt.wantMax, providerErr.ContextMaxTokens)
			} else {
				require.False(t, providerErr.IsContextTooLarge())
			}
		})
	}
}

func TestParseOptions_Effort(t *testing.T) {
	t.Parallel()

	options, err := ParseOptions(map[string]any{
		"send_reasoning":            true,
		"thinking":                  map[string]any{"budget_tokens": int64(2048)},
		"effort":                    "medium",
		"disable_parallel_tool_use": true,
	})
	require.NoError(t, err)
	require.NotNil(t, options.SendReasoning)
	require.True(t, *options.SendReasoning)
	require.NotNil(t, options.Thinking)
	require.Equal(t, int64(2048), options.Thinking.BudgetTokens)
	require.NotNil(t, options.Effort)
	require.Equal(t, EffortMedium, *options.Effort)
	require.NotNil(t, options.DisableParallelToolUse)
	require.True(t, *options.DisableParallelToolUse)
}

func TestGenerate_SendsOutputConfigEffort(t *testing.T) {
	t.Parallel()

	server, calls := newAnthropicJSONServer(mockAnthropicGenerateResponse())
	defer server.Close()

	provider, err := New(
		WithAPIKey("test-api-key"),
		WithBaseURL(server.URL),
	)
	require.NoError(t, err)

	model, err := provider.LanguageModel(context.Background(), "claude-sonnet-4-20250514")
	require.NoError(t, err)

	effort := EffortMedium
	_, err = model.Generate(context.Background(), fantasy.Call{
		Prompt: testPrompt(),
		ProviderOptions: NewProviderOptions(&ProviderOptions{
			Effort: &effort,
		}),
	})
	require.NoError(t, err)

	call := awaitAnthropicCall(t, calls)
	require.Equal(t, "POST", call.method)
	require.Equal(t, "/v1/messages", call.path)
	requireAnthropicEffort(t, call.body, EffortMedium)
}

func TestStream_SendsOutputConfigEffort(t *testing.T) {
	t.Parallel()

	server, calls := newAnthropicStreamingServer([]string{
		"event: message_start\n",
		"data: {\"type\":\"message_start\",\"message\":{}}\n\n",
		"event: message_stop\n",
		"data: {\"type\":\"message_stop\"}\n\n",
	})
	defer server.Close()

	provider, err := New(
		WithAPIKey("test-api-key"),
		WithBaseURL(server.URL),
	)
	require.NoError(t, err)

	model, err := provider.LanguageModel(context.Background(), "claude-sonnet-4-20250514")
	require.NoError(t, err)

	effort := EffortHigh
	stream, err := model.Stream(context.Background(), fantasy.Call{
		Prompt: testPrompt(),
		ProviderOptions: NewProviderOptions(&ProviderOptions{
			Effort: &effort,
		}),
	})
	require.NoError(t, err)

	stream(func(fantasy.StreamPart) bool { return true })

	call := awaitAnthropicCall(t, calls)
	require.Equal(t, "POST", call.method)
	require.Equal(t, "/v1/messages", call.path)
	requireAnthropicEffort(t, call.body, EffortHigh)
}

type anthropicCall struct {
	method string
	path   string
	body   map[string]any
}

func newAnthropicJSONServer(response map[string]any) (*httptest.Server, <-chan anthropicCall) {
	calls := make(chan anthropicCall, 4)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if r.Body != nil {
			_ = json.NewDecoder(r.Body).Decode(&body)
		}

		calls <- anthropicCall{
			method: r.Method,
			path:   r.URL.Path,
			body:   body,
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(response)
	}))

	return server, calls
}

func newAnthropicStreamingServer(chunks []string) (*httptest.Server, <-chan anthropicCall) {
	calls := make(chan anthropicCall, 4)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if r.Body != nil {
			_ = json.NewDecoder(r.Body).Decode(&body)
		}

		calls <- anthropicCall{
			method: r.Method,
			path:   r.URL.Path,
			body:   body,
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.WriteHeader(http.StatusOK)

		for _, chunk := range chunks {
			_, _ = fmt.Fprint(w, chunk)
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
		}
	}))

	return server, calls
}

func awaitAnthropicCall(t *testing.T, calls <-chan anthropicCall) anthropicCall {
	t.Helper()

	select {
	case call := <-calls:
		return call
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for Anthropic request")
		return anthropicCall{}
	}
}

func assertNoAnthropicCall(t *testing.T, calls <-chan anthropicCall) {
	t.Helper()

	select {
	case call := <-calls:
		t.Fatalf("expected no Anthropic API call, but got %s %s", call.method, call.path)
	case <-time.After(200 * time.Millisecond):
	}
}

func requireAnthropicEffort(t *testing.T, body map[string]any, expected Effort) {
	t.Helper()

	outputConfig, ok := body["output_config"].(map[string]any)
	thinking, ok := body["thinking"].(map[string]any)
	require.True(t, ok)
	require.Equal(t, string(expected), outputConfig["effort"])
	require.Equal(t, "adaptive", thinking["type"])
}

func testPrompt() fantasy.Prompt {
	return fantasy.Prompt{
		{
			Role: fantasy.MessageRoleUser,
			Content: []fantasy.MessagePart{
				fantasy.TextPart{Text: "Hello"},
			},
		},
	}
}

func mockAnthropicGenerateResponse() map[string]any {
	return map[string]any{
		"id":    "msg_01Test",
		"type":  "message",
		"role":  "assistant",
		"model": "claude-sonnet-4-20250514",
		"content": []any{
			map[string]any{
				"type": "text",
				"text": "Hi there",
			},
		},
		"stop_reason":   "end_turn",
		"stop_sequence": "",
		"usage": map[string]any{
			"cache_creation": map[string]any{
				"ephemeral_1h_input_tokens": 0,
				"ephemeral_5m_input_tokens": 0,
			},
			"cache_creation_input_tokens": 0,
			"cache_read_input_tokens":     0,
			"input_tokens":                5,
			"output_tokens":               2,
			"server_tool_use": map[string]any{
				"web_search_requests": 0,
			},
			"service_tier": "standard",
		},
	}
}

func mockAnthropicWebSearchResponse() map[string]any {
	return map[string]any{
		"id":    "msg_01WebSearch",
		"type":  "message",
		"role":  "assistant",
		"model": "claude-sonnet-4-20250514",
		"content": []any{
			map[string]any{
				"type":   "server_tool_use",
				"id":     "srvtoolu_01",
				"name":   "web_search",
				"input":  map[string]any{"query": "latest AI news"},
				"caller": map[string]any{"type": "direct"},
			},
			map[string]any{
				"type":        "web_search_tool_result",
				"tool_use_id": "srvtoolu_01",
				"caller":      map[string]any{"type": "direct"},
				"content": []any{
					map[string]any{
						"type":              "web_search_result",
						"url":               "https://example.com/ai-news",
						"title":             "Latest AI News",
						"encrypted_content": "encrypted_abc123",
						"page_age":          "2 hours ago",
					},
					map[string]any{
						"type":              "web_search_result",
						"url":               "https://example.com/ml-update",
						"title":             "ML Update",
						"encrypted_content": "encrypted_def456",
						"page_age":          "",
					},
				},
			},
			map[string]any{
				"type": "text",
				"text": "Based on recent search results, here is the latest AI news.",
			},
		},
		"stop_reason":   "end_turn",
		"stop_sequence": nil,
		"usage": map[string]any{
			"input_tokens":                100,
			"output_tokens":               50,
			"cache_creation_input_tokens": 0,
			"cache_read_input_tokens":     0,
			"server_tool_use": map[string]any{
				"web_search_requests": 1,
			},
		},
	}
}

func TestToPrompt_WebSearchProviderExecutedToolResults(t *testing.T) {
	t.Parallel()

	prompt := fantasy.Prompt{
		// User message.
		{
			Role: fantasy.MessageRoleUser,
			Content: []fantasy.MessagePart{
				fantasy.TextPart{Text: "Search for the latest AI news"},
			},
		},
		// Assistant message with a provider-executed tool call, its
		// result, and trailing text. toResponseMessages routes
		// provider-executed results into the assistant message, so
		// the prompt already reflects that structure.
		{
			Role: fantasy.MessageRoleAssistant,
			Content: []fantasy.MessagePart{
				fantasy.ToolCallPart{
					ToolCallID:       "srvtoolu_01",
					ToolName:         "web_search",
					Input:            `{"query":"latest AI news"}`,
					ProviderExecuted: true,
				},
				fantasy.ToolResultPart{
					ToolCallID:       "srvtoolu_01",
					ProviderExecuted: true,
					ProviderOptions: fantasy.ProviderOptions{
						Name: &WebSearchResultMetadata{
							Results: []WebSearchResultItem{
								{
									URL:              "https://example.com/ai-news",
									Title:            "Latest AI News",
									EncryptedContent: "encrypted_abc123",
									PageAge:          "2 hours ago",
								},
								{
									URL:              "https://example.com/ml-update",
									Title:            "ML Update",
									EncryptedContent: "encrypted_def456",
								},
							},
						},
					},
				},
				fantasy.TextPart{Text: "Here is what I found."},
			},
		},
	}

	_, messages, warnings := toPrompt(prompt, true)

	// No warnings expected; the provider-executed result is in the
	// assistant message so there is no empty tool message to drop.
	require.Empty(t, warnings)

	// We should have a user message and an assistant message.
	require.Len(t, messages, 2, "expected user + assistant messages")

	assistantMsg := messages[1]
	require.Len(t, assistantMsg.Content, 3,
		"expected server_tool_use + web_search_tool_result + text")

	// First content block: reconstructed server_tool_use.
	serverToolUse := assistantMsg.Content[0]
	require.NotNil(t, serverToolUse.OfServerToolUse,
		"first block should be a server_tool_use")
	require.Equal(t, "srvtoolu_01", serverToolUse.OfServerToolUse.ID)
	require.Equal(t, anthropic.ServerToolUseBlockParamName("web_search"),
		serverToolUse.OfServerToolUse.Name)

	// Second content block: reconstructed web_search_tool_result with
	// encrypted_content preserved for multi-turn round-tripping.
	webResult := assistantMsg.Content[1]
	require.NotNil(t, webResult.OfWebSearchToolResult,
		"second block should be a web_search_tool_result")
	require.Equal(t, "srvtoolu_01", webResult.OfWebSearchToolResult.ToolUseID)

	results := webResult.OfWebSearchToolResult.Content.OfWebSearchToolResultBlockItem
	require.Len(t, results, 2)
	require.Equal(t, "https://example.com/ai-news", results[0].URL)
	require.Equal(t, "Latest AI News", results[0].Title)
	require.Equal(t, "encrypted_abc123", results[0].EncryptedContent)
	require.Equal(t, "https://example.com/ml-update", results[1].URL)
	require.Equal(t, "encrypted_def456", results[1].EncryptedContent)
	// PageAge should be set for the first result and absent for the second.
	require.True(t, results[0].PageAge.Valid())
	require.Equal(t, "2 hours ago", results[0].PageAge.Value)
	require.False(t, results[1].PageAge.Valid())

	// Third content block: plain text.
	require.NotNil(t, assistantMsg.Content[2].OfText)
	require.Equal(t, "Here is what I found.", assistantMsg.Content[2].OfText.Text)
}

func TestGenerate_WebSearchResponse(t *testing.T) {
	t.Parallel()

	server, calls := newAnthropicJSONServer(mockAnthropicWebSearchResponse())
	defer server.Close()

	provider, err := New(
		WithAPIKey("test-api-key"),
		WithBaseURL(server.URL),
	)
	require.NoError(t, err)

	model, err := provider.LanguageModel(context.Background(), "claude-sonnet-4-20250514")
	require.NoError(t, err)

		resp, err := model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt(),
			Tools: []fantasy.Tool{
				WebSearchTool(nil),
			},	})
	require.NoError(t, err)

	call := awaitAnthropicCall(t, calls)
	require.Equal(t, "POST", call.method)
	require.Equal(t, "/v1/messages", call.path)

	// Walk the response content and categorise each item.
	var (
		toolCalls   []fantasy.ToolCallContent
		sources     []fantasy.SourceContent
		toolResults []fantasy.ToolResultContent
		texts       []fantasy.TextContent
	)
	for _, c := range resp.Content {
		switch v := c.(type) {
		case fantasy.ToolCallContent:
			toolCalls = append(toolCalls, v)
		case fantasy.SourceContent:
			sources = append(sources, v)
		case fantasy.ToolResultContent:
			toolResults = append(toolResults, v)
		case fantasy.TextContent:
			texts = append(texts, v)
		}
	}

	// ToolCallContent for the provider-executed web_search.
	require.Len(t, toolCalls, 1)
	require.True(t, toolCalls[0].ProviderExecuted)
	require.Equal(t, "web_search", toolCalls[0].ToolName)
	require.Equal(t, "srvtoolu_01", toolCalls[0].ToolCallID)

	// SourceContent entries for each search result.
	require.Len(t, sources, 2)
	require.Equal(t, "https://example.com/ai-news", sources[0].URL)
	require.Equal(t, "Latest AI News", sources[0].Title)
	require.Equal(t, fantasy.SourceTypeURL, sources[0].SourceType)
	require.Equal(t, "https://example.com/ml-update", sources[1].URL)
	require.Equal(t, "ML Update", sources[1].Title)

	// ToolResultContent with provider metadata preserving encrypted_content.
	require.Len(t, toolResults, 1)
	require.True(t, toolResults[0].ProviderExecuted)
	require.Equal(t, "web_search", toolResults[0].ToolName)
	require.Equal(t, "srvtoolu_01", toolResults[0].ToolCallID)

	searchMeta, ok := toolResults[0].ProviderMetadata[Name]
	require.True(t, ok, "providerMetadata should contain anthropic key")
	webMeta, ok := searchMeta.(*WebSearchResultMetadata)
	require.True(t, ok, "metadata should be *WebSearchResultMetadata")
	require.Len(t, webMeta.Results, 2)
	require.Equal(t, "encrypted_abc123", webMeta.Results[0].EncryptedContent)
	require.Equal(t, "encrypted_def456", webMeta.Results[1].EncryptedContent)
	require.Equal(t, "2 hours ago", webMeta.Results[0].PageAge)

	// TextContent with the final answer.
	require.Len(t, texts, 1)
	require.Equal(t,
		"Based on recent search results, here is the latest AI news.",
		texts[0].Text,
	)
}

func TestGenerate_WebSearchToolInRequest(t *testing.T) {
	t.Parallel()

	t.Run("basic web_search tool", func(t *testing.T) {
		t.Parallel()

		server, calls := newAnthropicJSONServer(mockAnthropicGenerateResponse())
		defer server.Close()

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.URL),
		)
		require.NoError(t, err)

		model, err := provider.LanguageModel(context.Background(), "claude-sonnet-4-20250514")
		require.NoError(t, err)

		_, err = model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt(),
			Tools: []fantasy.Tool{
				WebSearchTool(nil),
			},
		})
		require.NoError(t, err)

		call := awaitAnthropicCall(t, calls)
		tools, ok := call.body["tools"].([]any)
		require.True(t, ok, "request body should have tools array")
		require.Len(t, tools, 1)

		tool, ok := tools[0].(map[string]any)
		require.True(t, ok)
		require.Equal(t, "web_search_20250305", tool["type"])
	})

	t.Run("with allowed_domains and blocked_domains", func(t *testing.T) {
		t.Parallel()

		server, calls := newAnthropicJSONServer(mockAnthropicGenerateResponse())
		defer server.Close()

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.URL),
		)
		require.NoError(t, err)

		model, err := provider.LanguageModel(context.Background(), "claude-sonnet-4-20250514")
		require.NoError(t, err)

		_, err = model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt(),
			Tools: []fantasy.Tool{
				WebSearchTool(&WebSearchToolOptions{
					AllowedDomains: []string{"example.com", "test.com"},
				}),
			},
		})
		require.NoError(t, err)

		call := awaitAnthropicCall(t, calls)
		tools, ok := call.body["tools"].([]any)
		require.True(t, ok)
		require.Len(t, tools, 1)

		tool, ok := tools[0].(map[string]any)
		require.True(t, ok)
		require.Equal(t, "web_search_20250305", tool["type"])

		domains, ok := tool["allowed_domains"].([]any)
		require.True(t, ok, "tool should have allowed_domains")
		require.Len(t, domains, 2)
		require.Equal(t, "example.com", domains[0])
		require.Equal(t, "test.com", domains[1])
	})

	t.Run("with max uses and user location", func(t *testing.T) {
		t.Parallel()

		server, calls := newAnthropicJSONServer(mockAnthropicGenerateResponse())
		defer server.Close()

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.URL),
		)
		require.NoError(t, err)

		model, err := provider.LanguageModel(context.Background(), "claude-sonnet-4-20250514")
		require.NoError(t, err)

		_, err = model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt(),
			Tools: []fantasy.Tool{
				WebSearchTool(&WebSearchToolOptions{
					MaxUses: 5,
					UserLocation: &UserLocation{
						City:    "San Francisco",
						Country: "US",
					},
				}),
			},
		})
		require.NoError(t, err)

		call := awaitAnthropicCall(t, calls)
		tools, ok := call.body["tools"].([]any)
		require.True(t, ok)
		require.Len(t, tools, 1)

		tool, ok := tools[0].(map[string]any)
		require.True(t, ok)
		require.Equal(t, "web_search_20250305", tool["type"])

		// max_uses is serialized as a JSON number; json.Unmarshal
		// into map[string]any decodes numbers as float64.
		maxUses, ok := tool["max_uses"].(float64)
		require.True(t, ok, "tool should have max_uses")
		require.Equal(t, float64(5), maxUses)

		userLoc, ok := tool["user_location"].(map[string]any)
		require.True(t, ok, "tool should have user_location")
		require.Equal(t, "San Francisco", userLoc["city"])
		require.Equal(t, "US", userLoc["country"])
		require.Equal(t, "approximate", userLoc["type"])
	})

	t.Run("with max uses", func(t *testing.T) {
		t.Parallel()

		server, calls := newAnthropicJSONServer(mockAnthropicGenerateResponse())
		defer server.Close()

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.URL),
		)
		require.NoError(t, err)

		model, err := provider.LanguageModel(context.Background(), "claude-sonnet-4-20250514")
		require.NoError(t, err)

		_, err = model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt(),
			Tools: []fantasy.Tool{
				WebSearchTool(&WebSearchToolOptions{
					MaxUses: 3,
				}),
			},
		})
		require.NoError(t, err)

		call := awaitAnthropicCall(t, calls)
		tools, ok := call.body["tools"].([]any)
		require.True(t, ok)
		require.Len(t, tools, 1)

		tool, ok := tools[0].(map[string]any)
		require.True(t, ok)
		require.Equal(t, "web_search_20250305", tool["type"])

		maxUses, ok := tool["max_uses"].(float64)
		require.True(t, ok, "tool should have max_uses")
		require.Equal(t, float64(3), maxUses)
	})

	t.Run("with json-round-tripped provider tool args", func(t *testing.T) {
		t.Parallel()

		server, calls := newAnthropicJSONServer(mockAnthropicGenerateResponse())
		defer server.Close()

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.URL),
		)
		require.NoError(t, err)

		model, err := provider.LanguageModel(context.Background(), "claude-sonnet-4-20250514")
		require.NoError(t, err)

		baseTool := WebSearchTool(&WebSearchToolOptions{
			MaxUses:        7,
			BlockedDomains: []string{"example.com", "test.com"},
			UserLocation: &UserLocation{
				City:     "San Francisco",
				Region:   "CA",
				Country:  "US",
				Timezone: "America/Los_Angeles",
			},
		})

		data, err := json.Marshal(baseTool)
		require.NoError(t, err)

		var roundTripped fantasy.ProviderDefinedTool
		err = json.Unmarshal(data, &roundTripped)
		require.NoError(t, err)

		_, err = model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt(),
			Tools:  []fantasy.Tool{roundTripped},
		})
		require.NoError(t, err)

		call := awaitAnthropicCall(t, calls)
		tools, ok := call.body["tools"].([]any)
		require.True(t, ok)
		require.Len(t, tools, 1)

		tool, ok := tools[0].(map[string]any)
		require.True(t, ok)
		require.Equal(t, "web_search_20250305", tool["type"])

		domains, ok := tool["blocked_domains"].([]any)
		require.True(t, ok, "tool should have blocked_domains")
		require.Len(t, domains, 2)
		require.Equal(t, "example.com", domains[0])
		require.Equal(t, "test.com", domains[1])

		maxUses, ok := tool["max_uses"].(float64)
		require.True(t, ok, "tool should have max_uses")
		require.Equal(t, float64(7), maxUses)

		userLoc, ok := tool["user_location"].(map[string]any)
		require.True(t, ok, "tool should have user_location")
		require.Equal(t, "San Francisco", userLoc["city"])
		require.Equal(t, "CA", userLoc["region"])
		require.Equal(t, "US", userLoc["country"])
		require.Equal(t, "America/Los_Angeles", userLoc["timezone"])
		require.Equal(t, "approximate", userLoc["type"])
	})
}

func TestAnyToStringSlice(t *testing.T) {
	t.Parallel()

	t.Run("from string slice", func(t *testing.T) {
		t.Parallel()

		got := anyToStringSlice([]string{"example.com", ""})
		require.Equal(t, []string{"example.com", ""}, got)
	})

	t.Run("from any slice filters non-strings and empty", func(t *testing.T) {
		t.Parallel()

		got := anyToStringSlice([]any{"example.com", 123, "", "test.com"})
		require.Equal(t, []string{"example.com", "test.com"}, got)
	})

	t.Run("unsupported type", func(t *testing.T) {
		t.Parallel()

		got := anyToStringSlice("example.com")
		require.Nil(t, got)
	})
}

func TestAnyToInt64(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		input  any
		want   int64
		wantOK bool
	}{
		{name: "int64", input: int64(7), want: 7, wantOK: true},
		{name: "float64 integer", input: float64(7), want: 7, wantOK: true},
		{name: "float32 integer", input: float32(9), want: 9, wantOK: true},
		{name: "float64 non-integer", input: float64(7.5), wantOK: false},
		{name: "float64 max exact int ok", input: float64(1<<53 - 1), want: 1<<53 - 1, wantOK: true},
		{name: "float64 over max exact int", input: float64(1 << 53), wantOK: false},
		{name: "json number int", input: json.Number("42"), want: 42, wantOK: true},
		{name: "json number float", input: json.Number("4.2"), wantOK: false},
		{name: "nan", input: math.NaN(), wantOK: false},
		{name: "inf", input: math.Inf(1), wantOK: false},
		{name: "uint64 overflow", input: uint64(math.MaxInt64) + 1, wantOK: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := anyToInt64(tt.input)
			require.Equal(t, tt.wantOK, ok)
			if tt.wantOK {
				require.Equal(t, tt.want, got)
			}
		})
	}
}

func TestAnyToUserLocation(t *testing.T) {
	t.Parallel()

	t.Run("pointer passthrough", func(t *testing.T) {
		t.Parallel()

		input := &UserLocation{City: "San Francisco", Country: "US"}
		got := anyToUserLocation(input)
		require.Same(t, input, got)
	})

	t.Run("struct value", func(t *testing.T) {
		t.Parallel()

		got := anyToUserLocation(UserLocation{City: "San Francisco", Country: "US"})
		require.NotNil(t, got)
		require.Equal(t, "San Francisco", got.City)
		require.Equal(t, "US", got.Country)
	})

	t.Run("map value", func(t *testing.T) {
		t.Parallel()

		got := anyToUserLocation(map[string]any{
			"city":     "San Francisco",
			"region":   "CA",
			"country":  "US",
			"timezone": "America/Los_Angeles",
			"type":     "approximate",
		})
		require.NotNil(t, got)
		require.Equal(t, "San Francisco", got.City)
		require.Equal(t, "CA", got.Region)
		require.Equal(t, "US", got.Country)
		require.Equal(t, "America/Los_Angeles", got.Timezone)
	})

	t.Run("empty map", func(t *testing.T) {
		t.Parallel()

		got := anyToUserLocation(map[string]any{"type": "approximate"})
		require.Nil(t, got)
	})

	t.Run("unsupported type", func(t *testing.T) {
		t.Parallel()

		got := anyToUserLocation("San Francisco")
		require.Nil(t, got)
	})
}

func TestStream_WebSearchResponse(t *testing.T) {
	t.Parallel()

	// Build SSE chunks that simulate a web search streaming response.
	// The Anthropic SDK accumulates content blocks via
	// acc.Accumulate(event). We read the Content and ToolUseID
	// directly from struct fields instead of using AsAny(), which
	// avoids the SDK's re-marshal limitation that previously dropped
	// source data.
	webSearchResultContent, _ := json.Marshal([]any{
		map[string]any{
			"type":              "web_search_result",
			"url":               "https://example.com/ai-news",
			"title":             "Latest AI News",
			"encrypted_content": "encrypted_abc123",
			"page_age":          "2 hours ago",
		},
	})

	chunks := []string{
		// message_start
		"event: message_start\n",
		`data: {"type":"message_start","message":{"id":"msg_01WebSearch","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[],"stop_reason":null,"usage":{"input_tokens":100,"output_tokens":0}}}` + "\n\n",
		// Block 0: server_tool_use
		"event: content_block_start\n",
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_01","name":"web_search","input":{}}}` + "\n\n",
		"event: content_block_stop\n",
		`data: {"type":"content_block_stop","index":0}` + "\n\n",
		// Block 1: web_search_tool_result
		"event: content_block_start\n",
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"web_search_tool_result","tool_use_id":"srvtoolu_01","content":` + string(webSearchResultContent) + `}}` + "\n\n",
		"event: content_block_stop\n",
		`data: {"type":"content_block_stop","index":1}` + "\n\n",
		// Block 2: text
		"event: content_block_start\n",
		`data: {"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}` + "\n\n",
		"event: content_block_delta\n",
		`data: {"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"Here are the results."}}` + "\n\n",
		"event: content_block_stop\n",
		`data: {"type":"content_block_stop","index":2}` + "\n\n",
		// message_stop
		"event: message_stop\n",
		`data: {"type":"message_stop"}` + "\n\n",
	}

	server, calls := newAnthropicStreamingServer(chunks)
	defer server.Close()

	provider, err := New(
		WithAPIKey("test-api-key"),
		WithBaseURL(server.URL),
	)
	require.NoError(t, err)

	model, err := provider.LanguageModel(context.Background(), "claude-sonnet-4-20250514")
	require.NoError(t, err)

		stream, err := model.Stream(context.Background(), fantasy.Call{
			Prompt: testPrompt(),
			Tools: []fantasy.Tool{
				WebSearchTool(nil),
			},	})
	require.NoError(t, err)

	var parts []fantasy.StreamPart
	stream(func(part fantasy.StreamPart) bool {
		parts = append(parts, part)
		return true
	})

	_ = awaitAnthropicCall(t, calls)

	// Collect parts by type for assertions.
	var (
		toolInputStarts []fantasy.StreamPart
		toolCalls       []fantasy.StreamPart
		toolResults     []fantasy.StreamPart
		sourceParts     []fantasy.StreamPart
		textDeltas      []fantasy.StreamPart
	)
	for _, p := range parts {
		switch p.Type {
		case fantasy.StreamPartTypeToolInputStart:
			toolInputStarts = append(toolInputStarts, p)
		case fantasy.StreamPartTypeToolCall:
			toolCalls = append(toolCalls, p)
		case fantasy.StreamPartTypeToolResult:
			toolResults = append(toolResults, p)
		case fantasy.StreamPartTypeSource:
			sourceParts = append(sourceParts, p)
		case fantasy.StreamPartTypeTextDelta:
			textDeltas = append(textDeltas, p)
		}
	}

	// server_tool_use emits a ToolInputStart with ProviderExecuted.
	require.NotEmpty(t, toolInputStarts, "should have a tool input start")
	require.True(t, toolInputStarts[0].ProviderExecuted)
	require.Equal(t, "web_search", toolInputStarts[0].ToolCallName)

	// server_tool_use emits a ToolCall with ProviderExecuted.
	require.NotEmpty(t, toolCalls, "should have a tool call")
	require.True(t, toolCalls[0].ProviderExecuted)

	// web_search_tool_result always emits a ToolResult even when
	// the SDK drops source data. The ToolUseID comes from the raw
	// union field as a fallback.
	require.NotEmpty(t, toolResults, "should have a tool result")
	require.True(t, toolResults[0].ProviderExecuted)
	require.Equal(t, "web_search", toolResults[0].ToolCallName)
	require.Equal(t, "srvtoolu_01", toolResults[0].ID,
		"tool result ID should match the tool_use_id")

	// Source parts are now correctly emitted by reading struct fields
	// directly instead of using AsAny().
	require.Len(t, sourceParts, 1)
	require.Equal(t, "https://example.com/ai-news", sourceParts[0].URL)
	require.Equal(t, "Latest AI News", sourceParts[0].Title)
	require.Equal(t, fantasy.SourceTypeURL, sourceParts[0].SourceType)

	// Text block emits a text delta.
	require.NotEmpty(t, textDeltas, "should have text deltas")
	require.Equal(t, "Here are the results.", textDeltas[0].Delta)
}
