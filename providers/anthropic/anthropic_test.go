package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"charm.land/fantasy"
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

func TestGenerate_InvalidEffortReturnsFantasyError(t *testing.T) {
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

	invalidEffort := Effort("invalid")
	_, err = model.Generate(context.Background(), fantasy.Call{
		Prompt: testPrompt(),
		ProviderOptions: NewProviderOptions(&ProviderOptions{
			Effort: &invalidEffort,
		}),
	})
	require.Error(t, err)

	var fantasyErr *fantasy.Error
	require.ErrorAs(t, err, &fantasyErr)
	require.Equal(t, "invalid argument", fantasyErr.Title)
	require.Contains(t, fantasyErr.Message, "low, medium, high, max")

	assertNoAnthropicCall(t, calls)
}

func TestGenerate_ThinkingBudgetStillSentWithEffort(t *testing.T) {
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

	effort := EffortMax
	_, err = model.Generate(context.Background(), fantasy.Call{
		Prompt: testPrompt(),
		ProviderOptions: NewProviderOptions(&ProviderOptions{
			Effort: &effort,
			Thinking: &ThinkingProviderOption{
				BudgetTokens: 2048,
			},
		}),
	})
	require.NoError(t, err)

	call := awaitAnthropicCall(t, calls)
	requireAnthropicEffort(t, call.body, EffortMax)

	thinking, ok := call.body["thinking"].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "enabled", thinking["type"])
	require.Equal(t, float64(2048), thinking["budget_tokens"])
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
	require.True(t, ok)
	require.Equal(t, string(expected), outputConfig["effort"])
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
