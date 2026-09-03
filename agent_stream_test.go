package fantasy

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// EchoTool is a simple tool that echoes back the input message
type EchoTool struct {
	providerOptions ProviderOptions
}

func (e *EchoTool) SetProviderOptions(opts ProviderOptions) {
	e.providerOptions = opts
}

func (e *EchoTool) ProviderOptions() ProviderOptions {
	return e.providerOptions
}

// Info returns the tool information
func (e *EchoTool) Info() ToolInfo {
	return ToolInfo{
		Name:        "echo",
		Description: "Echo back the provided message",
		Parameters: map[string]any{
			"message": map[string]any{
				"type":        "string",
				"description": "The message to echo back",
			},
		},
		Required: []string{"message"},
	}
}

// Run executes the echo tool
func (e *EchoTool) Run(ctx context.Context, params ToolCall) (ToolResponse, error) {
	var input struct {
		Message string `json:"message"`
	}

	if err := json.Unmarshal([]byte(params.Input), &input); err != nil {
		return NewTextErrorResponse("Invalid input: " + err.Error()), nil
	}

	if input.Message == "" {
		return NewTextErrorResponse("Message cannot be empty"), nil
	}

	return NewTextResponse("Echo: " + input.Message), nil
}

// TestStreamingAgentCallbacks tests that all streaming callbacks are called correctly
func TestStreamingAgentCallbacks(t *testing.T) {
	t.Parallel()

	// Track which callbacks were called
	callbacks := make(map[string]bool)

	// Create a mock language model that returns various stream parts
	mockModel := &mockLanguageModel{
		streamFunc: func(ctx context.Context, call Call) (StreamResponse, error) {
			return func(yield func(StreamPart) bool) {
				// Test all stream part types
				if !yield(StreamPart{Type: StreamPartTypeWarnings, Warnings: []CallWarning{{Type: CallWarningTypeOther, Message: "test warning"}}}) {
					return
				}
				if !yield(StreamPart{Type: StreamPartTypeTextStart, ID: "text-1"}) {
					return
				}
				if !yield(StreamPart{Type: StreamPartTypeTextDelta, ID: "text-1", Delta: "Hello"}) {
					return
				}
				if !yield(StreamPart{Type: StreamPartTypeTextEnd, ID: "text-1"}) {
					return
				}
				if !yield(StreamPart{Type: StreamPartTypeReasoningStart, ID: "reasoning-1"}) {
					return
				}
				if !yield(StreamPart{Type: StreamPartTypeReasoningDelta, ID: "reasoning-1", Delta: "thinking..."}) {
					return
				}
				if !yield(StreamPart{Type: StreamPartTypeReasoningEnd, ID: "reasoning-1"}) {
					return
				}
				if !yield(StreamPart{Type: StreamPartTypeToolInputStart, ID: "tool-1", ToolCallName: "test_tool"}) {
					return
				}
				if !yield(StreamPart{Type: StreamPartTypeToolInputDelta, ID: "tool-1", Delta: `{"param"`}) {
					return
				}
				if !yield(StreamPart{Type: StreamPartTypeToolInputEnd, ID: "tool-1"}) {
					return
				}
				if !yield(StreamPart{Type: StreamPartTypeSource, ID: "source-1", SourceType: SourceTypeURL, URL: "https://example.com", Title: "Example"}) {
					return
				}
				yield(StreamPart{
					Type:         StreamPartTypeFinish,
					Usage:        Usage{InputTokens: 5, OutputTokens: 2, TotalTokens: 7},
					FinishReason: FinishReasonStop,
				})
			}, nil
		},
	}

	// Create agent
	agent := NewAgent(mockModel)

	ctx := context.Background()

	// Create streaming call with all callbacks
	streamCall := AgentStreamCall{
		Prompt: "Test all callbacks",
		OnAgentStart: func() {
			callbacks["OnAgentStart"] = true
		},
		OnAgentFinish: func(result *AgentResult) error {
			callbacks["OnAgentFinish"] = true
			return nil
		},
		OnStepStart: func(stepNumber int) error {
			callbacks["OnStepStart"] = true
			return nil
		},
		OnStepFinish: func(stepResult StepResult) error {
			callbacks["OnStepFinish"] = true
			return nil
		},
		OnFinish: func(result *AgentResult) {
			callbacks["OnFinish"] = true
		},
		OnError: func(err error) {
			callbacks["OnError"] = true
		},
		OnChunk: func(part StreamPart) error {
			callbacks["OnChunk"] = true
			return nil
		},
		OnWarnings: func(warnings []CallWarning) error {
			callbacks["OnWarnings"] = true
			return nil
		},
		OnTextStart: func(id string) error {
			callbacks["OnTextStart"] = true
			return nil
		},
		OnTextDelta: func(id, text string) error {
			callbacks["OnTextDelta"] = true
			return nil
		},
		OnTextEnd: func(id string) error {
			callbacks["OnTextEnd"] = true
			return nil
		},
		OnReasoningStart: func(id string, _ ReasoningContent) error {
			callbacks["OnReasoningStart"] = true
			return nil
		},
		OnReasoningDelta: func(id, text string) error {
			callbacks["OnReasoningDelta"] = true
			return nil
		},
		OnReasoningEnd: func(id string, content ReasoningContent) error {
			callbacks["OnReasoningEnd"] = true
			return nil
		},
		OnToolInputStart: func(id, toolName string) error {
			callbacks["OnToolInputStart"] = true
			return nil
		},
		OnToolInputDelta: func(id, delta string) error {
			callbacks["OnToolInputDelta"] = true
			return nil
		},
		OnToolInputEnd: func(id string) error {
			callbacks["OnToolInputEnd"] = true
			return nil
		},
		OnToolCall: func(toolCall ToolCallContent) error {
			callbacks["OnToolCall"] = true
			return nil
		},
		OnToolResult: func(result ToolResultContent) error {
			callbacks["OnToolResult"] = true
			return nil
		},
		OnSource: func(source SourceContent) error {
			callbacks["OnSource"] = true
			return nil
		},
		OnStreamFinish: func(usage Usage, finishReason FinishReason, providerMetadata ProviderMetadata) error {
			callbacks["OnStreamFinish"] = true
			return nil
		},
	}

	// Execute streaming agent
	result, err := agent.Stream(ctx, streamCall)
	require.NoError(t, err)
	require.NotNil(t, result)

	// Verify that expected callbacks were called
	expectedCallbacks := []string{
		"OnAgentStart",
		"OnAgentFinish",
		"OnStepStart",
		"OnStepFinish",
		"OnFinish",
		"OnChunk",
		"OnWarnings",
		"OnTextStart",
		"OnTextDelta",
		"OnTextEnd",
		"OnReasoningStart",
		"OnReasoningDelta",
		"OnReasoningEnd",
		"OnToolInputStart",
		"OnToolInputDelta",
		"OnToolInputEnd",
		"OnSource",
		"OnStreamFinish",
	}

	for _, callback := range expectedCallbacks {
		require.True(t, callbacks[callback], "Expected callback %s to be called", callback)
	}

	// Verify that error callbacks were not called
	require.False(t, callbacks["OnError"], "OnError should not be called in successful case")
	require.False(t, callbacks["OnToolCall"], "OnToolCall should not be called without actual tool calls")
	require.False(t, callbacks["OnToolResult"], "OnToolResult should not be called without actual tool results")
}

func TestAgent_Stream_DisableAllTools(t *testing.T) {
	t.Parallel()

	model := &mockLanguageModel{
		streamFunc: func(ctx context.Context, call Call) (StreamResponse, error) {
			require.Empty(t, call.Tools)
			return func(yield func(StreamPart) bool) {
				yield(StreamPart{
					Type:         StreamPartTypeFinish,
					FinishReason: FinishReasonStop,
				})
			}, nil
		},
	}
	tool := &EchoTool{}
	providerTool := ProviderDefinedTool{ID: "provider.web_search", Name: "web_search"}
	agent := NewAgent(model, WithTools(tool), WithProviderDefinedTools(providerTool))

	result, err := agent.Stream(context.Background(), AgentStreamCall{
		Prompt:          "test-input",
		DisableAllTools: true,
		PrepareStep: func(ctx context.Context, _ PrepareStepFunctionOptions) (context.Context, PrepareStepResult, error) {
			return ctx, PrepareStepResult{}, nil
		},
	})
	require.NoError(t, err)
	require.NotNil(t, result)
}

// TestStreamingAgentWithTools tests streaming agent with tool calls (mirrors TS test patterns)
func TestStreamingAgentWithTools(t *testing.T) {
	t.Parallel()

	stepCount := 0
	// Create a mock language model that makes a tool call then finishes
	mockModel := &mockLanguageModel{
		streamFunc: func(ctx context.Context, call Call) (StreamResponse, error) {
			stepCount++
			return func(yield func(StreamPart) bool) {
				if stepCount == 1 {
					// First step: make tool call
					if !yield(StreamPart{Type: StreamPartTypeToolInputStart, ID: "tool-1", ToolCallName: "echo"}) {
						return
					}
					if !yield(StreamPart{Type: StreamPartTypeToolInputDelta, ID: "tool-1", Delta: `{"message"`}) {
						return
					}
					if !yield(StreamPart{Type: StreamPartTypeToolInputDelta, ID: "tool-1", Delta: `: "test"}`}) {
						return
					}
					if !yield(StreamPart{Type: StreamPartTypeToolInputEnd, ID: "tool-1"}) {
						return
					}
					if !yield(StreamPart{
						Type:          StreamPartTypeToolCall,
						ID:            "tool-1",
						ToolCallName:  "echo",
						ToolCallInput: `{"message": "test"}`,
					}) {
						return
					}
					yield(StreamPart{
						Type:         StreamPartTypeFinish,
						Usage:        Usage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15},
						FinishReason: FinishReasonToolCalls,
					})
				} else {
					// Second step: finish after tool execution
					if !yield(StreamPart{Type: StreamPartTypeTextStart, ID: "text-1"}) {
						return
					}
					if !yield(StreamPart{Type: StreamPartTypeTextDelta, ID: "text-1", Delta: "Tool executed successfully"}) {
						return
					}
					if !yield(StreamPart{Type: StreamPartTypeTextEnd, ID: "text-1"}) {
						return
					}
					yield(StreamPart{
						Type:         StreamPartTypeFinish,
						Usage:        Usage{InputTokens: 5, OutputTokens: 3, TotalTokens: 8},
						FinishReason: FinishReasonStop,
					})
				}
			}, nil
		},
	}

	// Create agent with echo tool
	agent := NewAgent(
		mockModel,
		WithSystemPrompt("You are a helpful assistant."),
		WithTools(&EchoTool{}),
	)

	ctx := context.Background()

	// Track callback invocations
	var toolInputStartCalled bool
	var toolInputDeltaCalled bool
	var toolInputEndCalled bool
	var toolCallCalled bool
	var toolResultCalled bool

	// Create streaming call with callbacks
	streamCall := AgentStreamCall{
		Prompt: "Echo 'test'",
		OnToolInputStart: func(id, toolName string) error {
			toolInputStartCalled = true
			require.Equal(t, "tool-1", id)
			require.Equal(t, "echo", toolName)
			return nil
		},
		OnToolInputDelta: func(id, delta string) error {
			toolInputDeltaCalled = true
			require.Equal(t, "tool-1", id)
			require.Contains(t, []string{`{"message"`, `: "test"}`}, delta)
			return nil
		},
		OnToolInputEnd: func(id string) error {
			toolInputEndCalled = true
			require.Equal(t, "tool-1", id)
			return nil
		},
		OnToolCall: func(toolCall ToolCallContent) error {
			toolCallCalled = true
			require.Equal(t, "echo", toolCall.ToolName)
			require.Equal(t, `{"message": "test"}`, toolCall.Input)
			return nil
		},
		OnToolResult: func(result ToolResultContent) error {
			toolResultCalled = true
			require.Equal(t, "echo", result.ToolName)
			return nil
		},
	}

	// Execute streaming agent
	result, err := agent.Stream(ctx, streamCall)
	require.NoError(t, err)

	// Verify results
	require.True(t, toolInputStartCalled, "OnToolInputStart should have been called")
	require.True(t, toolInputDeltaCalled, "OnToolInputDelta should have been called")
	require.True(t, toolInputEndCalled, "OnToolInputEnd should have been called")
	require.True(t, toolCallCalled, "OnToolCall should have been called")
	require.True(t, toolResultCalled, "OnToolResult should have been called")
	require.Equal(t, 2, len(result.Steps)) // Two steps: tool call + final response

	// Check that tool was executed in first step
	firstStep := result.Steps[0]
	toolCalls := firstStep.Content.ToolCalls()
	require.Equal(t, 1, len(toolCalls))
	require.Equal(t, "echo", toolCalls[0].ToolName)

	toolResults := firstStep.Content.ToolResults()
	require.Equal(t, 1, len(toolResults))
	require.Equal(t, "echo", toolResults[0].ToolName)
}

// TestStreamingAgentToolCallBeforeResult verifies that all OnToolCall callbacks
// complete before any OnToolResult fires. This is the ordering guarantee
// provided by buffering dispatches until the stream is fully consumed.
func TestStreamingAgentToolCallBeforeResult(t *testing.T) {
	t.Parallel()

	stepCount := 0
	mockModel := &mockLanguageModel{
		streamFunc: func(ctx context.Context, call Call) (StreamResponse, error) {
			stepCount++
			return func(yield func(StreamPart) bool) {
				if stepCount == 1 {
					// Emit two tool calls in the same step.
					for _, id := range []string{"tool-1", "tool-2"} {
						if !yield(StreamPart{Type: StreamPartTypeToolInputStart, ID: id, ToolCallName: "echo"}) {
							return
						}
						if !yield(StreamPart{Type: StreamPartTypeToolInputDelta, ID: id, Delta: `{"message": "` + id + `"}`}) {
							return
						}
						if !yield(StreamPart{Type: StreamPartTypeToolInputEnd, ID: id}) {
							return
						}
						if !yield(StreamPart{
							Type:          StreamPartTypeToolCall,
							ID:            id,
							ToolCallName:  "echo",
							ToolCallInput: `{"message": "` + id + `"}`,
						}) {
							return
						}
					}
					yield(StreamPart{
						Type:         StreamPartTypeFinish,
						FinishReason: FinishReasonToolCalls,
					})
				} else {
					yield(StreamPart{
						Type:         StreamPartTypeFinish,
						FinishReason: FinishReasonStop,
					})
				}
			}, nil
		},
	}

	agent := NewAgent(mockModel, WithTools(&EchoTool{}))

	var mu sync.Mutex
	var events []string

	_, err := agent.Stream(context.Background(), AgentStreamCall{
		Prompt: "echo twice",
		OnToolCall: func(tc ToolCallContent) error {
			mu.Lock()
			events = append(events, "call:"+tc.ToolCallID)
			mu.Unlock()
			return nil
		},
		OnToolResult: func(tr ToolResultContent) error {
			mu.Lock()
			events = append(events, "result:"+tr.ToolCallID)
			mu.Unlock()
			return nil
		},
	})
	require.NoError(t, err)

	// Both OnToolCall events must appear before any OnToolResult event.
	lastCallIdx := -1
	firstResultIdx := len(events)
	for i, e := range events {
		if strings.HasPrefix(e, "call:") {
			lastCallIdx = i
		}
		if strings.HasPrefix(e, "result:") && i < firstResultIdx {
			firstResultIdx = i
		}
	}
	require.Equal(t, 2, stepCount)
	require.Less(t, lastCallIdx, firstResultIdx,
		"all OnToolCall events must complete before the first OnToolResult; got %v", events)
}

// TestStreamingAgentTextDeltas tests text streaming (mirrors TS textStream tests)
func TestStreamingAgentTextDeltas(t *testing.T) {
	t.Parallel()

	// Create a mock language model that returns text deltas
	mockModel := &mockLanguageModel{
		streamFunc: func(ctx context.Context, call Call) (StreamResponse, error) {
			return func(yield func(StreamPart) bool) {
				if !yield(StreamPart{Type: StreamPartTypeTextStart, ID: "text-1"}) {
					return
				}
				if !yield(StreamPart{Type: StreamPartTypeTextDelta, ID: "text-1", Delta: "Hello"}) {
					return
				}
				if !yield(StreamPart{Type: StreamPartTypeTextDelta, ID: "text-1", Delta: ", "}) {
					return
				}
				if !yield(StreamPart{Type: StreamPartTypeTextDelta, ID: "text-1", Delta: "world!"}) {
					return
				}
				if !yield(StreamPart{Type: StreamPartTypeTextEnd, ID: "text-1"}) {
					return
				}
				yield(StreamPart{
					Type:         StreamPartTypeFinish,
					Usage:        Usage{InputTokens: 3, OutputTokens: 10, TotalTokens: 13},
					FinishReason: FinishReasonStop,
				})
			}, nil
		},
	}

	agent := NewAgent(mockModel)
	ctx := context.Background()

	// Track text deltas
	var textDeltas []string

	streamCall := AgentStreamCall{
		Prompt: "Say hello",
		OnTextDelta: func(id, text string) error {
			if text != "" {
				textDeltas = append(textDeltas, text)
			}
			return nil
		},
	}

	result, err := agent.Stream(ctx, streamCall)
	require.NoError(t, err)

	// Verify text deltas match expected pattern
	require.Equal(t, []string{"Hello", ", ", "world!"}, textDeltas)
	require.Equal(t, "Hello, world!", result.Response.Content.Text())
	require.Equal(t, int64(13), result.TotalUsage.TotalTokens)
}

// TestStreamingAgentReasoning tests reasoning content (mirrors TS reasoning tests)
func TestStreamingAgentReasoning(t *testing.T) {
	t.Parallel()

	mockModel := &mockLanguageModel{
		streamFunc: func(ctx context.Context, call Call) (StreamResponse, error) {
			return func(yield func(StreamPart) bool) {
				if !yield(StreamPart{Type: StreamPartTypeReasoningStart, ID: "reasoning-1"}) {
					return
				}
				if !yield(StreamPart{Type: StreamPartTypeReasoningDelta, ID: "reasoning-1", Delta: "I will open the conversation"}) {
					return
				}
				if !yield(StreamPart{Type: StreamPartTypeReasoningDelta, ID: "reasoning-1", Delta: " with witty banter."}) {
					return
				}
				if !yield(StreamPart{Type: StreamPartTypeReasoningEnd, ID: "reasoning-1"}) {
					return
				}
				if !yield(StreamPart{Type: StreamPartTypeTextStart, ID: "text-1"}) {
					return
				}
				if !yield(StreamPart{Type: StreamPartTypeTextDelta, ID: "text-1", Delta: "Hi there!"}) {
					return
				}
				if !yield(StreamPart{Type: StreamPartTypeTextEnd, ID: "text-1"}) {
					return
				}
				yield(StreamPart{
					Type:         StreamPartTypeFinish,
					Usage:        Usage{InputTokens: 5, OutputTokens: 15, TotalTokens: 20},
					FinishReason: FinishReasonStop,
				})
			}, nil
		},
	}

	agent := NewAgent(mockModel)
	ctx := context.Background()

	var reasoningDeltas []string
	var textDeltas []string

	streamCall := AgentStreamCall{
		Prompt: "Think and respond",
		OnReasoningDelta: func(id, text string) error {
			reasoningDeltas = append(reasoningDeltas, text)
			return nil
		},
		OnTextDelta: func(id, text string) error {
			textDeltas = append(textDeltas, text)
			return nil
		},
	}

	result, err := agent.Stream(ctx, streamCall)
	require.NoError(t, err)

	// Verify reasoning and text are separate
	require.Equal(t, []string{"I will open the conversation", " with witty banter."}, reasoningDeltas)
	require.Equal(t, []string{"Hi there!"}, textDeltas)
	require.Equal(t, "Hi there!", result.Response.Content.Text())
	require.Equal(t, "I will open the conversation with witty banter.", result.Response.Content.ReasoningText())
}

// TestStreamingAgentError tests error handling (mirrors TS error tests)
func TestStreamingAgentError(t *testing.T) {
	t.Parallel()

	// Create a mock language model that returns an error
	mockModel := &mockLanguageModel{
		streamFunc: func(ctx context.Context, call Call) (StreamResponse, error) {
			return func(yield func(StreamPart) bool) {
				yield(StreamPart{Type: StreamPartTypeError, Error: fmt.Errorf("mock stream error")})
			}, nil
		},
	}

	agent := NewAgent(mockModel)
	ctx := context.Background()

	// Track error callbacks
	var errorOccurred bool
	var errorMessage string

	streamCall := AgentStreamCall{
		Prompt: "This will fail",

		OnError: func(err error) {
			errorOccurred = true
			errorMessage = err.Error()
		},
	}

	// Execute streaming agent
	result, err := agent.Stream(ctx, streamCall)
	require.Error(t, err)
	require.Nil(t, result)
	require.True(t, errorOccurred, "OnError should have been called")
	require.Contains(t, errorMessage, "mock stream error")
}

// TestStreamingAgentSources tests source handling (mirrors TS source tests)
func TestStreamingAgentSources(t *testing.T) {
	t.Parallel()

	mockModel := &mockLanguageModel{
		streamFunc: func(ctx context.Context, call Call) (StreamResponse, error) {
			return func(yield func(StreamPart) bool) {
				if !yield(StreamPart{
					Type:       StreamPartTypeSource,
					ID:         "source-1",
					SourceType: SourceTypeURL,
					URL:        "https://example.com",
					Title:      "Example",
				}) {
					return
				}
				if !yield(StreamPart{Type: StreamPartTypeTextStart, ID: "text-1"}) {
					return
				}
				if !yield(StreamPart{Type: StreamPartTypeTextDelta, ID: "text-1", Delta: "Hello!"}) {
					return
				}
				if !yield(StreamPart{Type: StreamPartTypeTextEnd, ID: "text-1"}) {
					return
				}
				if !yield(StreamPart{
					Type:       StreamPartTypeSource,
					ID:         "source-2",
					SourceType: SourceTypeDocument,
					Title:      "Document Example",
				}) {
					return
				}
				yield(StreamPart{
					Type:         StreamPartTypeFinish,
					Usage:        Usage{InputTokens: 3, OutputTokens: 5, TotalTokens: 8},
					FinishReason: FinishReasonStop,
				})
			}, nil
		},
	}

	agent := NewAgent(mockModel)
	ctx := context.Background()

	var sources []SourceContent

	streamCall := AgentStreamCall{
		Prompt: "Search and respond",
		OnSource: func(source SourceContent) error {
			sources = append(sources, source)
			return nil
		},
	}

	result, err := agent.Stream(ctx, streamCall)
	require.NoError(t, err)

	// Verify sources were captured
	require.Equal(t, 2, len(sources))
	require.Equal(t, SourceTypeURL, sources[0].SourceType)
	require.Equal(t, "https://example.com", sources[0].URL)
	require.Equal(t, "Example", sources[0].Title)
	require.Equal(t, SourceTypeDocument, sources[1].SourceType)
	require.Equal(t, "Document Example", sources[1].Title)

	// Verify sources are in final result
	resultSources := result.Response.Content.Sources()
	require.Equal(t, 2, len(resultSources))
}

func TestStreamingAgent_StopTurn(t *testing.T) {
	t.Parallel()

	stepCount := 0
	mockModel := &mockLanguageModel{
		streamFunc: func(ctx context.Context, call Call) (StreamResponse, error) {
			stepCount++
			return func(yield func(StreamPart) bool) {
				if stepCount == 1 {
					if !yield(StreamPart{Type: StreamPartTypeToolInputStart, ID: "tool-1", ToolCallName: "blocked_tool"}) {
						return
					}
					if !yield(StreamPart{Type: StreamPartTypeToolInputDelta, ID: "tool-1", Delta: `{"message"`}) {
						return
					}
					if !yield(StreamPart{Type: StreamPartTypeToolInputDelta, ID: "tool-1", Delta: `: "test"}`}) {
						return
					}
					if !yield(StreamPart{Type: StreamPartTypeToolInputEnd, ID: "tool-1"}) {
						return
					}
					if !yield(StreamPart{
						Type:          StreamPartTypeToolCall,
						ID:            "tool-1",
						ToolCallName:  "blocked_tool",
						ToolCallInput: `{"message": "test"}`,
					}) {
						return
					}
					yield(StreamPart{
						Type:         StreamPartTypeFinish,
						Usage:        Usage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15},
						FinishReason: FinishReasonToolCalls,
					})
				} else {
					// Should not be reached because StopTurn prevents a second step
					t.Fatal("model should not be called a second time after StopTurn")
				}
			}, nil
		},
	}

	type BlockedInput struct {
		Message string `json:"message" description:"Message"`
	}

	blockedTool := NewAgentTool(
		"blocked_tool",
		"A tool that stops the turn",
		func(ctx context.Context, input BlockedInput, _ ToolCall) (ToolResponse, error) {
			resp := NewTextErrorResponse("permission denied")
			resp.StopTurn = true
			return resp, nil
		},
	)

	agent := NewAgent(mockModel, WithTools(blockedTool))

	result, err := agent.Stream(context.Background(), AgentStreamCall{
		Prompt: "test stop turn",
	})
	require.NoError(t, err)
	require.NotNil(t, result)

	// Only one step — StopTurn prevented the second model call.
	require.Len(t, result.Steps, 1)
	require.Equal(t, 1, stepCount)

	// Tool result should be present with StopTurn=true.
	toolResults := result.Steps[0].Content.ToolResults()
	require.Len(t, toolResults, 1)
	require.Equal(t, "blocked_tool", toolResults[0].ToolName)
	require.True(t, toolResults[0].StopTurn)

	// The final response also includes the stop-marked tool result.
	responseResults := result.Response.Content.ToolResults()
	require.Len(t, responseResults, 1)
	require.True(t, responseResults[0].StopTurn)
}

func TestStreamingAgent_SkipsToolDispatchWhenTruncated(t *testing.T) {
	t.Parallel()

	var toolExecuted bool
	mockModel := &mockLanguageModel{
		streamFunc: func(ctx context.Context, call Call) (StreamResponse, error) {
			return func(yield func(StreamPart) bool) {
				// Emit a tool call with truncated arguments
				if !yield(StreamPart{Type: StreamPartTypeToolInputStart, ID: "call-trunc", ToolCallName: "echo"}) {
					return
				}
				if !yield(StreamPart{Type: StreamPartTypeToolInputDelta, ID: "call-trunc", Delta: `{"message":"tr`}) {
					return
				}
				if !yield(StreamPart{Type: StreamPartTypeToolInputEnd, ID: "call-trunc"}) {
					return
				}
				if !yield(StreamPart{
					Type:          StreamPartTypeToolCall,
					ID:            "call-trunc",
					ToolCallName:  "echo",
					ToolCallInput: `{"message":"tr`,
				}) {
					return
				}
				// Finish with length instead of tool_calls
				yield(StreamPart{
					Type:         StreamPartTypeFinish,
					Usage:        Usage{InputTokens: 10, OutputTokens: 20, TotalTokens: 30},
					FinishReason: FinishReasonLength,
				})
			}, nil
		},
	}

	echoTool := &trackingEchoTool{onExecute: func() { toolExecuted = true }}
	agent := NewAgent(
		mockModel,
		WithTools(echoTool),
	)

	result, err := agent.Stream(context.Background(), AgentStreamCall{
		Prompt: "test",
	})
	require.NoError(t, err)
	require.Equal(t, FinishReasonLength, result.Response.FinishReason)
	require.False(t, toolExecuted, "tool must not be dispatched when finish_reason is length")
	require.Len(t, result.Steps, 1)
}

// trackingEchoTool wraps EchoTool to track whether Run was called.
type trackingEchoTool struct {
	onExecute       func()
	providerOptions ProviderOptions
}

func (t *trackingEchoTool) Info() ToolInfo {
	return (&EchoTool{}).Info()
}

func (t *trackingEchoTool) Run(ctx context.Context, params ToolCall) (ToolResponse, error) {
	if t.onExecute != nil {
		t.onExecute()
	}
	return (&EchoTool{}).Run(ctx, params)
}

func (t *trackingEchoTool) ProviderOptions() ProviderOptions {
	return t.providerOptions
}

func (t *trackingEchoTool) SetProviderOptions(opts ProviderOptions) {
	t.providerOptions = opts
}

// TestStreamingAgent_SkipsToolDispatchOnAbnormalFinish generalizes the
// truncation guard: any finish reason that is not an explicit tool-calls
// turn must not dispatch, even if ToolCall parts were emitted (a buggy or
// unusual provider path, or a finish reason the provider layer maps from
// an upstream failure such as content_filter or
// insufficient_system_resource). See CHARM-2020.
func TestStreamingAgent_SkipsToolDispatchOnAbnormalFinish(t *testing.T) {
	t.Parallel()

	for _, reason := range []FinishReason{FinishReasonError, FinishReasonContentFilter, FinishReasonUnknown} {
		t.Run(string(reason), func(t *testing.T) {
			t.Parallel()

			var toolExecuted bool
			mockModel := &mockLanguageModel{
				streamFunc: func(ctx context.Context, call Call) (StreamResponse, error) {
					return func(yield func(StreamPart) bool) {
						if !yield(StreamPart{Type: StreamPartTypeToolInputStart, ID: "call-x", ToolCallName: "echo"}) {
							return
						}
						if !yield(StreamPart{Type: StreamPartTypeToolInputDelta, ID: "call-x", Delta: `{"message":"hi"}`}) {
							return
						}
						if !yield(StreamPart{Type: StreamPartTypeToolInputEnd, ID: "call-x"}) {
							return
						}
						if !yield(StreamPart{Type: StreamPartTypeToolCall, ID: "call-x", ToolCallName: "echo", ToolCallInput: `{"message":"hi"}`}) {
							return
						}
						yield(StreamPart{
							Type:         StreamPartTypeFinish,
							Usage:        Usage{InputTokens: 10, OutputTokens: 20, TotalTokens: 30},
							FinishReason: reason,
						})
					}, nil
				},
			}

			echoTool := &trackingEchoTool{onExecute: func() { toolExecuted = true }}
			agent := NewAgent(mockModel, WithTools(echoTool))

			result, err := agent.Stream(context.Background(), AgentStreamCall{Prompt: "test"})
			require.NoError(t, err)
			require.False(t, toolExecuted, "tool must not be dispatched when finish_reason is %s", reason)
			require.Equal(t, reason, result.Response.FinishReason)
			require.Len(t, result.Steps, 1)
		})
	}
}

// TestStreamingAgent_ParallelToolResultsInCallOrder guards CHARM-2020 F6:
// two parallel tools where the second finishes first must still append
// results in the order the model called them. Providers that pair results
// with calls positionally (and any diff of step content) depend on it.
func TestStreamingAgent_ParallelToolResultsInCallOrder(t *testing.T) {
	t.Parallel()

	type input struct {
		N int `json:"n"`
	}
	slow := NewParallelAgentTool("slow", "finishes last",
		func(ctx context.Context, in input, call ToolCall) (ToolResponse, error) {
			time.Sleep(50 * time.Millisecond)
			return NewTextResponse("slow-result"), nil
		})
	fast := NewParallelAgentTool("fast", "finishes first",
		func(ctx context.Context, in input, call ToolCall) (ToolResponse, error) {
			return NewTextResponse("fast-result"), nil
		})

	mockModel := &mockLanguageModel{
		streamFunc: func(ctx context.Context, call Call) (StreamResponse, error) {
			for _, msg := range call.Prompt {
				if msg.Role == MessageRoleTool {
					return func(yield func(StreamPart) bool) {
						yield(StreamPart{Type: StreamPartTypeTextStart, ID: "t"})
						yield(StreamPart{Type: StreamPartTypeTextDelta, ID: "t", Delta: "done"})
						yield(StreamPart{Type: StreamPartTypeTextEnd, ID: "t"})
						yield(StreamPart{Type: StreamPartTypeFinish, FinishReason: FinishReasonStop, Usage: Usage{TotalTokens: 5}})
					}, nil
				}
			}
			return func(yield func(StreamPart) bool) {
				if !yield(StreamPart{Type: StreamPartTypeToolCall, ID: "call-slow", ToolCallName: "slow", ToolCallInput: `{"n":1}`}) {
					return
				}
				if !yield(StreamPart{Type: StreamPartTypeToolCall, ID: "call-fast", ToolCallName: "fast", ToolCallInput: `{"n":2}`}) {
					return
				}
				yield(StreamPart{Type: StreamPartTypeFinish, FinishReason: FinishReasonToolCalls, Usage: Usage{TotalTokens: 10}})
			}, nil
		},
	}

	agent := NewAgent(mockModel, WithTools(slow, fast))
	result, err := agent.Stream(context.Background(), AgentStreamCall{Prompt: "run both"})
	require.NoError(t, err)
	require.Len(t, result.Steps, 2)

	var order []string
	for _, c := range result.Steps[0].Content {
		if tr, ok := AsContentType[ToolResultContent](c); ok {
			order = append(order, tr.ToolCallID)
		}
	}
	require.Equal(t, []string{"call-slow", "call-fast"}, order,
		"results must follow call order, not completion order")
}

// TestStreamingAgent_NoRepairOrOnToolCallBeforeAbnormalFinish pins the
// CHARM-2020 "never repair" requirement for the stream path: a provider
// that emits a ToolCall part and only then finishes with length/error must
// not have the call repaired (an extra model call) or exposed to
// OnToolCall consumers.
func TestStreamingAgent_NoRepairOrOnToolCallBeforeAbnormalFinish(t *testing.T) {
	t.Parallel()

	for _, reason := range []FinishReason{FinishReasonLength, FinishReasonError, FinishReasonContentFilter, FinishReasonUnknown} {
		t.Run(string(reason), func(t *testing.T) {
			t.Parallel()

			var repaired, onToolCallFired, executed bool
			model := &mockLanguageModel{
				streamFunc: func(ctx context.Context, call Call) (StreamResponse, error) {
					return func(yield func(StreamPart) bool) {
						// ToolCall emitted BEFORE the finish part.
						if !yield(StreamPart{Type: StreamPartTypeToolCall, ID: "call-x", ToolCallName: "echo", ToolCallInput: `{"message":"tr`}) {
							return
						}
						yield(StreamPart{Type: StreamPartTypeFinish, FinishReason: reason, Usage: Usage{TotalTokens: 10}})
					}, nil
				},
			}

			agent := NewAgent(
				model,
				WithTools(&trackingEchoTool{onExecute: func() { executed = true }}),
				WithRepairToolCall(func(ctx context.Context, options ToolCallRepairOptions) (*ToolCallContent, error) {
					repaired = true
					c := options.OriginalToolCall
					c.Input = `{"message":"fixed"}`
					return &c, nil
				}),
			)

			result, err := agent.Stream(context.Background(), AgentStreamCall{
				Prompt: "test",
				OnToolCall: func(ToolCallContent) error {
					onToolCallFired = true
					return nil
				},
			})
			require.NoError(t, err)
			require.False(t, repaired, "repair must not run when finish is %s", reason)
			require.False(t, onToolCallFired, "OnToolCall must not fire when finish is %s", reason)
			require.False(t, executed, "tool must not execute when finish is %s", reason)
			require.Len(t, result.Steps, 1)
			// The raw call is still recorded for the step.
			var calls []ToolCallContent
			for _, c := range result.Steps[0].Content {
				if tc, ok := AsContentType[ToolCallContent](c); ok {
					calls = append(calls, tc)
				}
			}
			require.Len(t, calls, 1)
			require.Equal(t, `{"message":"tr`, calls[0].Input, "raw (unrepaired) input must be recorded")
		})
	}
}

// TestStreamingAgent_ProviderExecutedToolCallGatedOnFinish extends the
// late-binding guarantee to provider-executed calls: OnToolCall must not
// fire for them before the finish reason is known either.
func TestStreamingAgent_ProviderExecutedToolCallGatedOnFinish(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		reason       FinishReason
		wantNotified bool
	}{
		{FinishReasonToolCalls, true},
		{FinishReasonLength, false},
		{FinishReasonError, false},
	} {
		t.Run(string(tc.reason), func(t *testing.T) {
			t.Parallel()

			var onToolCallFired bool
			model := &mockLanguageModel{
				streamFunc: func(ctx context.Context, call Call) (StreamResponse, error) {
					return func(yield func(StreamPart) bool) {
						// Provider-executed tool call (e.g. web search)
						if !yield(StreamPart{Type: StreamPartTypeToolCall, ID: "ws-1", ToolCallName: "web_search", ToolCallInput: `{"q":"x"}`, ProviderExecuted: true}) {
							return
						}
						if !yield(StreamPart{Type: StreamPartTypeToolResult, ID: "ws-1", ToolCallName: "web_search", ProviderExecuted: true}) {
							return
						}
						yield(StreamPart{Type: StreamPartTypeFinish, FinishReason: tc.reason, Usage: Usage{TotalTokens: 10}})
					}, nil
				},
			}

			agent := NewAgent(model)
			result, err := agent.Stream(context.Background(), AgentStreamCall{
				Prompt: "test",
				OnToolCall: func(ToolCallContent) error {
					onToolCallFired = true
					return nil
				},
			})
			require.NoError(t, err)
			require.Equal(t, tc.wantNotified, onToolCallFired,
				"OnToolCall fired=%v for provider-executed call with finish %s", onToolCallFired, tc.reason)
			require.Len(t, result.Steps, 1)
		})
	}
}
