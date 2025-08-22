package ai

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/charmbracelet/crush/internal/llm/tools"
	"github.com/stretchr/testify/require"
)

// Mock tool for testing
type mockTool struct {
	name        string
	description string
	parameters  map[string]any
	required    []string
	executeFunc func(ctx context.Context, call tools.ToolCall) (tools.ToolResponse, error)
}

func (m *mockTool) Info() tools.ToolInfo {
	return tools.ToolInfo{
		Name:        m.name,
		Description: m.description,
		Parameters:  m.parameters,
		Required:    m.required,
	}
}

func (m *mockTool) Run(ctx context.Context, call tools.ToolCall) (tools.ToolResponse, error) {
	if m.executeFunc != nil {
		return m.executeFunc(ctx, call)
	}
	return tools.ToolResponse{Content: "mock result", IsError: false}, nil
}

// Mock language model for testing
type mockLanguageModel struct {
	generateFunc func(ctx context.Context, call Call) (*Response, error)
}

func (m *mockLanguageModel) Generate(ctx context.Context, call Call) (*Response, error) {
	if m.generateFunc != nil {
		return m.generateFunc(ctx, call)
	}
	return &Response{
		Content: []Content{
			TextContent{Text: "Hello, world!"},
		},
		Usage: Usage{
			InputTokens:  3,
			OutputTokens: 10,
			TotalTokens:  13,
		},
		FinishReason: FinishReasonStop,
	}, nil
}

func (m *mockLanguageModel) Stream(ctx context.Context, call Call) (StreamResponse, error) {
	panic("not implemented")
}

func (m *mockLanguageModel) Provider() string {
	return "mock-provider"
}

func (m *mockLanguageModel) Model() string {
	return "mock-model"
}

// Test result.content - comprehensive content types (matches TS test)
func TestAgent_Generate_ResultContent_AllTypes(t *testing.T) {
	t.Parallel()

	tool1 := &mockTool{
		name:        "tool1",
		description: "Test tool",
		parameters: map[string]any{
			"value": map[string]any{"type": "string"},
		},
		required: []string{"value"},
		executeFunc: func(ctx context.Context, call tools.ToolCall) (tools.ToolResponse, error) {
			var input map[string]any
			err := json.Unmarshal([]byte(call.Input), &input)
			require.NoError(t, err)
			require.Equal(t, "value", input["value"])
			return tools.ToolResponse{Content: "result1", IsError: false}, nil
		},
	}

	model := &mockLanguageModel{
		generateFunc: func(ctx context.Context, call Call) (*Response, error) {
			return &Response{
				Content: []Content{
					TextContent{Text: "Hello, world!"},
					SourceContent{
						ID:         "123",
						URL:        "https://example.com",
						Title:      "Example",
						SourceType: SourceTypeURL,
						ProviderMetadata: ProviderMetadata{
							"provider": map[string]any{"custom": "value"},
						},
					},
					FileContent{
						Data:      []byte{1, 2, 3},
						MediaType: "image/png",
					},
					ReasoningContent{
						Text: "I will open the conversation with witty banter.",
					},
					ToolCallContent{
						ToolCallID: "call-1",
						ToolName:   "tool1",
						Input:      `{"value":"value"}`,
					},
					TextContent{Text: "More text"},
				},
				Usage: Usage{
					InputTokens:  3,
					OutputTokens: 10,
					TotalTokens:  13,
				},
				FinishReason: FinishReasonStop, // Note: FinishReasonStop, not ToolCalls
			}, nil
		},
	}

	agent := NewAgent(model, WithTools(tool1))
	result, err := agent.Generate(context.Background(), AgentCall{
		Prompt: "prompt",
	})

	require.NoError(t, err)
	require.NotNil(t, result)
	require.Len(t, result.Steps, 1) // Single step like TypeScript

	// Check final response content includes tool result
	require.Len(t, result.Response.Content, 7) // original 6 + 1 tool result

	// Verify each content type in order
	textContent, ok := AsContentType[TextContent](result.Response.Content[0])
	require.True(t, ok)
	require.Equal(t, "Hello, world!", textContent.Text)

	sourceContent, ok := AsContentType[SourceContent](result.Response.Content[1])
	require.True(t, ok)
	require.Equal(t, "123", sourceContent.ID)

	fileContent, ok := AsContentType[FileContent](result.Response.Content[2])
	require.True(t, ok)
	require.Equal(t, []byte{1, 2, 3}, fileContent.Data)

	reasoningContent, ok := AsContentType[ReasoningContent](result.Response.Content[3])
	require.True(t, ok)
	require.Equal(t, "I will open the conversation with witty banter.", reasoningContent.Text)

	toolCallContent, ok := AsContentType[ToolCallContent](result.Response.Content[4])
	require.True(t, ok)
	require.Equal(t, "call-1", toolCallContent.ToolCallID)

	moreTextContent, ok := AsContentType[TextContent](result.Response.Content[5])
	require.True(t, ok)
	require.Equal(t, "More text", moreTextContent.Text)

	// Tool result should be appended
	toolResultContent, ok := AsContentType[ToolResultContent](result.Response.Content[6])
	require.True(t, ok)
	require.Equal(t, "call-1", toolResultContent.ToolCallID)
	require.Equal(t, "tool1", toolResultContent.ToolName)
}

// Test result.text extraction
func TestAgent_Generate_ResultText(t *testing.T) {
	t.Parallel()

	model := &mockLanguageModel{
		generateFunc: func(ctx context.Context, call Call) (*Response, error) {
			return &Response{
				Content: []Content{
					TextContent{Text: "Hello, world!"},
				},
				Usage: Usage{
					InputTokens:  3,
					OutputTokens: 10,
					TotalTokens:  13,
				},
				FinishReason: FinishReasonStop,
			}, nil
		},
	}

	agent := NewAgent(model)
	result, err := agent.Generate(context.Background(), AgentCall{
		Prompt: "prompt",
	})

	require.NoError(t, err)
	require.NotNil(t, result)

	// Test text extraction from content
	text := result.Response.Content.Text()
	require.Equal(t, "Hello, world!", text)
}

// Test result.toolCalls extraction (matches TS test exactly)
func TestAgent_Generate_ResultToolCalls(t *testing.T) {
	t.Parallel()

	tool1 := &mockTool{
		name:        "tool1",
		description: "Test tool 1",
		parameters: map[string]any{
			"value": map[string]any{"type": "string"},
		},
		required: []string{"value"},
	}

	tool2 := &mockTool{
		name:        "tool2",
		description: "Test tool 2",
		parameters: map[string]any{
			"somethingElse": map[string]any{"type": "string"},
		},
		required: []string{"somethingElse"},
	}

	model := &mockLanguageModel{
		generateFunc: func(ctx context.Context, call Call) (*Response, error) {
			// Verify tools are passed correctly
			require.Len(t, call.Tools, 2)
			require.Equal(t, ToolChoiceAuto, *call.ToolChoice) // Should be auto, not required

			// Verify prompt structure
			require.Len(t, call.Prompt, 1)
			require.Equal(t, MessageRoleUser, call.Prompt[0].Role)

			return &Response{
				Content: []Content{
					ToolCallContent{
						ToolCallID: "call-1",
						ToolName:   "tool1",
						Input:      `{"value":"value"}`,
					},
				},
				Usage: Usage{
					InputTokens:  3,
					OutputTokens: 10,
					TotalTokens:  13,
				},
				FinishReason: FinishReasonStop, // Note: Stop, not ToolCalls
			}, nil
		},
	}

	agent := NewAgent(model, WithTools(tool1, tool2))
	result, err := agent.Generate(context.Background(), AgentCall{
		Prompt: "test-input",
	})

	require.NoError(t, err)
	require.NotNil(t, result)
	require.Len(t, result.Steps, 1) // Single step

	// Extract tool calls from final response (should be empty since tools don't execute)
	var toolCalls []ToolCallContent
	for _, content := range result.Response.Content {
		if toolCall, ok := AsContentType[ToolCallContent](content); ok {
			toolCalls = append(toolCalls, toolCall)
		}
	}

	require.Len(t, toolCalls, 1)
	require.Equal(t, "call-1", toolCalls[0].ToolCallID)
	require.Equal(t, "tool1", toolCalls[0].ToolName)

	// Parse and verify input
	var input map[string]any
	err = json.Unmarshal([]byte(toolCalls[0].Input), &input)
	require.NoError(t, err)
	require.Equal(t, "value", input["value"])
}

// Test result.toolResults extraction (matches TS test exactly)
func TestAgent_Generate_ResultToolResults(t *testing.T) {
	t.Parallel()

	tool1 := &mockTool{
		name:        "tool1",
		description: "Test tool",
		parameters: map[string]any{
			"value": map[string]any{"type": "string"},
		},
		required: []string{"value"},
		executeFunc: func(ctx context.Context, call tools.ToolCall) (tools.ToolResponse, error) {
			var input map[string]any
			err := json.Unmarshal([]byte(call.Input), &input)
			require.NoError(t, err)
			require.Equal(t, "value", input["value"])
			return tools.ToolResponse{Content: "result1", IsError: false}, nil
		},
	}

	model := &mockLanguageModel{
		generateFunc: func(ctx context.Context, call Call) (*Response, error) {
			// Verify tools and tool choice
			require.Len(t, call.Tools, 1)
			require.Equal(t, ToolChoiceAuto, *call.ToolChoice)

			// Verify prompt
			require.Len(t, call.Prompt, 1)
			require.Equal(t, MessageRoleUser, call.Prompt[0].Role)

			return &Response{
				Content: []Content{
					ToolCallContent{
						ToolCallID: "call-1",
						ToolName:   "tool1",
						Input:      `{"value":"value"}`,
					},
				},
				Usage: Usage{
					InputTokens:  3,
					OutputTokens: 10,
					TotalTokens:  13,
				},
				FinishReason: FinishReasonStop, // Note: Stop, not ToolCalls
			}, nil
		},
	}

	agent := NewAgent(model, WithTools(tool1))
	result, err := agent.Generate(context.Background(), AgentCall{
		Prompt: "test-input",
	})

	require.NoError(t, err)
	require.NotNil(t, result)
	require.Len(t, result.Steps, 1) // Single step

	// Extract tool results from final response
	var toolResults []ToolResultContent
	for _, content := range result.Response.Content {
		if toolResult, ok := AsContentType[ToolResultContent](content); ok {
			toolResults = append(toolResults, toolResult)
		}
	}

	require.Len(t, toolResults, 1)
	require.Equal(t, "call-1", toolResults[0].ToolCallID)
	require.Equal(t, "tool1", toolResults[0].ToolName)

	// Verify result content
	textResult, ok := toolResults[0].Result.(ToolResultOutputContentText)
	require.True(t, ok)
	require.Equal(t, "result1", textResult.Text)
}

// Test multi-step scenario (matches TS "2 steps: initial, tool-result" test)
func TestAgent_Generate_MultipleSteps(t *testing.T) {
	t.Parallel()

	tool1 := &mockTool{
		name:        "tool1",
		description: "Test tool",
		parameters: map[string]any{
			"value": map[string]any{"type": "string"},
		},
		required: []string{"value"},
		executeFunc: func(ctx context.Context, call tools.ToolCall) (tools.ToolResponse, error) {
			var input map[string]any
			err := json.Unmarshal([]byte(call.Input), &input)
			require.NoError(t, err)
			require.Equal(t, "value", input["value"])
			return tools.ToolResponse{Content: "result1", IsError: false}, nil
		},
	}

	callCount := 0
	model := &mockLanguageModel{
		generateFunc: func(ctx context.Context, call Call) (*Response, error) {
			callCount++
			switch callCount {
			case 1:
				// First call - return tool call with FinishReasonToolCalls
				return &Response{
					Content: []Content{
						ToolCallContent{
							ToolCallID: "call-1",
							ToolName:   "tool1",
							Input:      `{"value":"value"}`,
						},
					},
					Usage: Usage{
						InputTokens:  10,
						OutputTokens: 5,
						TotalTokens:  15,
					},
					FinishReason: FinishReasonToolCalls, // This triggers multi-step
				}, nil
			case 2:
				// Second call - return final text
				return &Response{
					Content: []Content{
						TextContent{Text: "Hello, world!"},
					},
					Usage: Usage{
						InputTokens:  3,
						OutputTokens: 10,
						TotalTokens:  13,
					},
					FinishReason: FinishReasonStop,
				}, nil
			default:
				t.Fatalf("Unexpected call count: %d", callCount)
				return nil, nil
			}
		},
	}

	agent := NewAgent(model, WithTools(tool1))
	result, err := agent.Generate(context.Background(), AgentCall{
		Prompt: "test-input",
	})

	require.NoError(t, err)
	require.NotNil(t, result)
	require.Len(t, result.Steps, 2)

	// Check total usage sums both steps
	require.Equal(t, int64(13), result.TotalUsage.InputTokens)  // 10 + 3
	require.Equal(t, int64(15), result.TotalUsage.OutputTokens) // 5 + 10
	require.Equal(t, int64(28), result.TotalUsage.TotalTokens)  // 15 + 13

	// Final response should be from last step
	require.Len(t, result.Response.Content, 1)
	textContent, ok := AsContentType[TextContent](result.Response.Content[0])
	require.True(t, ok)
	require.Equal(t, "Hello, world!", textContent.Text)

	// result.toolCalls should be empty (from last step)
	var toolCalls []ToolCallContent
	for _, content := range result.Response.Content {
		if _, ok := AsContentType[ToolCallContent](content); ok {
			toolCalls = append(toolCalls, content.(ToolCallContent))
		}
	}
	require.Len(t, toolCalls, 0)

	// result.toolResults should be empty (from last step)
	var toolResults []ToolResultContent
	for _, content := range result.Response.Content {
		if _, ok := AsContentType[ToolResultContent](content); ok {
			toolResults = append(toolResults, content.(ToolResultContent))
		}
	}
	require.Len(t, toolResults, 0)
}

// Test basic text generation
func TestAgent_Generate_BasicText(t *testing.T) {
	t.Parallel()

	model := &mockLanguageModel{
		generateFunc: func(ctx context.Context, call Call) (*Response, error) {
			return &Response{
				Content: []Content{
					TextContent{Text: "Hello, world!"},
				},
				Usage: Usage{
					InputTokens:  3,
					OutputTokens: 10,
					TotalTokens:  13,
				},
				FinishReason: FinishReasonStop,
			}, nil
		},
	}

	agent := NewAgent(model)
	result, err := agent.Generate(context.Background(), AgentCall{
		Prompt: "test prompt",
	})

	require.NoError(t, err)
	require.NotNil(t, result)
	require.Len(t, result.Steps, 1)

	// Check final response
	require.Len(t, result.Response.Content, 1)
	textContent, ok := AsContentType[TextContent](result.Response.Content[0])
	require.True(t, ok)
	require.Equal(t, "Hello, world!", textContent.Text)

	// Check usage
	require.Equal(t, int64(3), result.Response.Usage.InputTokens)
	require.Equal(t, int64(10), result.Response.Usage.OutputTokens)
	require.Equal(t, int64(13), result.Response.Usage.TotalTokens)

	// Check total usage
	require.Equal(t, int64(3), result.TotalUsage.InputTokens)
	require.Equal(t, int64(10), result.TotalUsage.OutputTokens)
	require.Equal(t, int64(13), result.TotalUsage.TotalTokens)
}

// Test empty prompt error
func TestAgent_Generate_EmptyPrompt(t *testing.T) {
	t.Parallel()

	model := &mockLanguageModel{}
	agent := NewAgent(model)

	result, err := agent.Generate(context.Background(), AgentCall{
		Prompt: "", // Empty prompt should cause error
	})

	require.Error(t, err)
	require.Nil(t, result)
	require.Contains(t, err.Error(), "Prompt can't be empty")
}

// Test with system prompt
func TestAgent_Generate_WithSystemPrompt(t *testing.T) {
	t.Parallel()

	model := &mockLanguageModel{
		generateFunc: func(ctx context.Context, call Call) (*Response, error) {
			// Verify system message is included
			require.Len(t, call.Prompt, 2) // system + user
			require.Equal(t, MessageRoleSystem, call.Prompt[0].Role)
			require.Equal(t, MessageRoleUser, call.Prompt[1].Role)

			systemPart, ok := call.Prompt[0].Content[0].(TextPart)
			require.True(t, ok)
			require.Equal(t, "You are a helpful assistant", systemPart.Text)

			return &Response{
				Content: []Content{
					TextContent{Text: "Hello, world!"},
				},
				Usage: Usage{
					InputTokens:  3,
					OutputTokens: 10,
					TotalTokens:  13,
				},
				FinishReason: FinishReasonStop,
			}, nil
		},
	}

	agent := NewAgent(model, WithSystemPrompt("You are a helpful assistant"))
	result, err := agent.Generate(context.Background(), AgentCall{
		Prompt: "test prompt",
	})

	require.NoError(t, err)
	require.NotNil(t, result)
}

// Test options.headers
func TestAgent_Generate_OptionsHeaders(t *testing.T) {
	t.Parallel()

	model := &mockLanguageModel{
		generateFunc: func(ctx context.Context, call Call) (*Response, error) {
			// Verify headers are passed
			require.Equal(t, map[string]string{
				"custom-request-header": "request-header-value",
			}, call.Headers)

			return &Response{
				Content: []Content{
					TextContent{Text: "Hello, world!"},
				},
				Usage: Usage{
					InputTokens:  3,
					OutputTokens: 10,
					TotalTokens:  13,
				},
				FinishReason: FinishReasonStop,
			}, nil
		},
	}

	agent := NewAgent(model)
	result, err := agent.Generate(context.Background(), AgentCall{
		Prompt:  "test-input",
		Headers: map[string]string{"custom-request-header": "request-header-value"},
	})

	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "Hello, world!", result.Response.Content.Text())
}

// Test options.activeTools filtering
func TestAgent_Generate_OptionsActiveTools(t *testing.T) {
	t.Parallel()

	tool1 := &mockTool{
		name:        "tool1",
		description: "Test tool 1",
		parameters: map[string]any{
			"value": map[string]any{"type": "string"},
		},
		required: []string{"value"},
	}

	tool2 := &mockTool{
		name:        "tool2",
		description: "Test tool 2",
		parameters: map[string]any{
			"value": map[string]any{"type": "string"},
		},
		required: []string{"value"},
	}

	model := &mockLanguageModel{
		generateFunc: func(ctx context.Context, call Call) (*Response, error) {
			// Verify only tool1 is available
			require.Len(t, call.Tools, 1)
			functionTool, ok := call.Tools[0].(FunctionTool)
			require.True(t, ok)
			require.Equal(t, "tool1", functionTool.Name)

			return &Response{
				Content: []Content{
					TextContent{Text: "Hello, world!"},
				},
				Usage: Usage{
					InputTokens:  3,
					OutputTokens: 10,
					TotalTokens:  13,
				},
				FinishReason: FinishReasonStop,
			}, nil
		},
	}

	agent := NewAgent(model, WithTools(tool1, tool2))
	result, err := agent.Generate(context.Background(), AgentCall{
		Prompt:      "test-input",
		ActiveTools: []string{"tool1"}, // Only tool1 should be active
	})

	require.NoError(t, err)
	require.NotNil(t, result)
}