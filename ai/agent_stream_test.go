package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

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
