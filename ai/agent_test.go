package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

// Mock tool for testing
type mockTool struct {
	name            string
	providerOptions ProviderOptions
	description     string
	parameters      map[string]any
	required        []string
	executeFunc     func(ctx context.Context, call ToolCall) (ToolResponse, error)
}

func (m *mockTool) SetProviderOptions(opts ProviderOptions) {
	m.providerOptions = opts
}

func (m *mockTool) ProviderOptions() ProviderOptions {
	return m.providerOptions
}

func (m *mockTool) Info() ToolInfo {
	return ToolInfo{
		Name:        m.name,
		Description: m.description,
		Parameters:  m.parameters,
		Required:    m.required,
	}
}

func (m *mockTool) Run(ctx context.Context, call ToolCall) (ToolResponse, error) {
	if m.executeFunc != nil {
		return m.executeFunc(ctx, call)
	}
	return ToolResponse{Content: "mock result", IsError: false}, nil
}

// Mock language model for testing
type mockLanguageModel struct {
	generateFunc func(ctx context.Context, call Call) (*Response, error)
	streamFunc   func(ctx context.Context, call Call) (StreamResponse, error)
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
	if m.streamFunc != nil {
		return m.streamFunc(ctx, call)
	}
	return nil, fmt.Errorf("mock stream not implemented")
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

	// Create a type-safe tool using the new API
	type TestInput struct {
		Value string `json:"value" description:"Test value"`
	}

	tool1 := NewAgentTool(
		"tool1",
		"Test tool",
		func(ctx context.Context, input TestInput, _ ToolCall) (ToolResponse, error) {
			require.Equal(t, "value", input.Value)
			return ToolResponse{Content: "result1", IsError: false}, nil
		},
	)

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

	// Create type-safe tools using the new API
	type Tool1Input struct {
		Value string `json:"value" description:"Test value"`
	}

	type Tool2Input struct {
		SomethingElse string `json:"somethingElse" description:"Another test value"`
	}

	tool1 := NewAgentTool(
		"tool1",
		"Test tool 1",
		func(ctx context.Context, input Tool1Input, _ ToolCall) (ToolResponse, error) {
			return ToolResponse{Content: "result1", IsError: false}, nil
		},
	)

	tool2 := NewAgentTool(
		"tool2",
		"Test tool 2",
		func(ctx context.Context, input Tool2Input, _ ToolCall) (ToolResponse, error) {
			return ToolResponse{Content: "result2", IsError: false}, nil
		},
	)

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

	// Create type-safe tool using the new API
	type TestInput struct {
		Value string `json:"value" description:"Test value"`
	}

	tool1 := NewAgentTool(
		"tool1",
		"Test tool",
		func(ctx context.Context, input TestInput, _ ToolCall) (ToolResponse, error) {
			require.Equal(t, "value", input.Value)
			return ToolResponse{Content: "result1", IsError: false}, nil
		},
	)

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

	// Create type-safe tool using the new API
	type TestInput struct {
		Value string `json:"value" description:"Test value"`
	}

	tool1 := NewAgentTool(
		"tool1",
		"Test tool",
		func(ctx context.Context, input TestInput, _ ToolCall) (ToolResponse, error) {
			require.Equal(t, "value", input.Value)
			return ToolResponse{Content: "result1", IsError: false}, nil
		},
	)

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

func TestResponseContent_Getters(t *testing.T) {
	t.Parallel()

	// Create test content with all types
	content := ResponseContent{
		TextContent{Text: "Hello world"},
		ReasoningContent{Text: "Let me think..."},
		FileContent{Data: []byte("file data"), MediaType: "text/plain"},
		SourceContent{SourceType: SourceTypeURL, URL: "https://example.com", Title: "Example"},
		ToolCallContent{ToolCallID: "call1", ToolName: "test_tool", Input: `{"arg": "value"}`},
		ToolResultContent{ToolCallID: "call1", ToolName: "test_tool", Result: ToolResultOutputContentText{Text: "result"}},
	}

	// Test Text()
	require.Equal(t, "Hello world", content.Text())

	// Test Reasoning()
	reasoning := content.Reasoning()
	require.Len(t, reasoning, 1)
	require.Equal(t, "Let me think...", reasoning[0].Text)

	// Test ReasoningText()
	require.Equal(t, "Let me think...", content.ReasoningText())

	// Test Files()
	files := content.Files()
	require.Len(t, files, 1)
	require.Equal(t, "text/plain", files[0].MediaType)
	require.Equal(t, []byte("file data"), files[0].Data)

	// Test Sources()
	sources := content.Sources()
	require.Len(t, sources, 1)
	require.Equal(t, SourceTypeURL, sources[0].SourceType)
	require.Equal(t, "https://example.com", sources[0].URL)
	require.Equal(t, "Example", sources[0].Title)

	// Test ToolCalls()
	toolCalls := content.ToolCalls()
	require.Len(t, toolCalls, 1)
	require.Equal(t, "call1", toolCalls[0].ToolCallID)
	require.Equal(t, "test_tool", toolCalls[0].ToolName)
	require.Equal(t, `{"arg": "value"}`, toolCalls[0].Input)

	// Test ToolResults()
	toolResults := content.ToolResults()
	require.Len(t, toolResults, 1)
	require.Equal(t, "call1", toolResults[0].ToolCallID)
	require.Equal(t, "test_tool", toolResults[0].ToolName)
	result, ok := AsToolResultOutputType[ToolResultOutputContentText](toolResults[0].Result)
	require.True(t, ok)
	require.Equal(t, "result", result.Text)
}

func TestResponseContent_Getters_Empty(t *testing.T) {
	t.Parallel()

	// Test with empty content
	content := ResponseContent{}

	require.Equal(t, "", content.Text())
	require.Equal(t, "", content.ReasoningText())
	require.Empty(t, content.Reasoning())
	require.Empty(t, content.Files())
	require.Empty(t, content.Sources())
	require.Empty(t, content.ToolCalls())
	require.Empty(t, content.ToolResults())
}

func TestResponseContent_Getters_MultipleItems(t *testing.T) {
	t.Parallel()

	// Test with multiple items of same type
	content := ResponseContent{
		ReasoningContent{Text: "First thought"},
		ReasoningContent{Text: "Second thought"},
		FileContent{Data: []byte("file1"), MediaType: "text/plain"},
		FileContent{Data: []byte("file2"), MediaType: "image/png"},
	}

	// Test multiple reasoning
	reasoning := content.Reasoning()
	require.Len(t, reasoning, 2)
	require.Equal(t, "First thought", reasoning[0].Text)
	require.Equal(t, "Second thought", reasoning[1].Text)

	// Test concatenated reasoning text
	require.Equal(t, "First thoughtSecond thought", content.ReasoningText())

	// Test multiple files
	files := content.Files()
	require.Len(t, files, 2)
	require.Equal(t, "text/plain", files[0].MediaType)
	require.Equal(t, "image/png", files[1].MediaType)
}

func TestStopConditions(t *testing.T) {
	t.Parallel()

	// Create test steps
	step1 := StepResult{
		Response: Response{
			Content: ResponseContent{
				TextContent{Text: "Hello"},
			},
			FinishReason: FinishReasonToolCalls,
			Usage:        Usage{TotalTokens: 10},
		},
	}

	step2 := StepResult{
		Response: Response{
			Content: ResponseContent{
				TextContent{Text: "World"},
				ToolCallContent{ToolCallID: "call1", ToolName: "search", Input: `{"query": "test"}`},
			},
			FinishReason: FinishReasonStop,
			Usage:        Usage{TotalTokens: 15},
		},
	}

	step3 := StepResult{
		Response: Response{
			Content: ResponseContent{
				ReasoningContent{Text: "Let me think..."},
				FileContent{Data: []byte("data"), MediaType: "text/plain"},
			},
			FinishReason: FinishReasonLength,
			Usage:        Usage{TotalTokens: 20},
		},
	}

	t.Run("StepCountIs", func(t *testing.T) {
		t.Parallel()
		condition := StepCountIs(2)

		// Should not stop with 1 step
		require.False(t, condition([]StepResult{step1}))

		// Should stop with 2 steps
		require.True(t, condition([]StepResult{step1, step2}))

		// Should stop with more than 2 steps
		require.True(t, condition([]StepResult{step1, step2, step3}))

		// Should not stop with empty steps
		require.False(t, condition([]StepResult{}))
	})

	t.Run("HasToolCall", func(t *testing.T) {
		t.Parallel()
		condition := HasToolCall("search")

		// Should not stop when tool not called
		require.False(t, condition([]StepResult{step1}))

		// Should stop when tool is called in last step
		require.True(t, condition([]StepResult{step1, step2}))

		// Should not stop when tool called in earlier step but not last
		require.False(t, condition([]StepResult{step1, step2, step3}))

		// Should not stop with empty steps
		require.False(t, condition([]StepResult{}))

		// Should not stop when different tool is called
		differentToolCondition := HasToolCall("different_tool")
		require.False(t, differentToolCondition([]StepResult{step1, step2}))
	})

	t.Run("HasContent", func(t *testing.T) {
		t.Parallel()
		reasoningCondition := HasContent(ContentTypeReasoning)
		fileCondition := HasContent(ContentTypeFile)

		// Should not stop when content type not present
		require.False(t, reasoningCondition([]StepResult{step1, step2}))

		// Should stop when content type is present in last step
		require.True(t, reasoningCondition([]StepResult{step1, step2, step3}))
		require.True(t, fileCondition([]StepResult{step1, step2, step3}))

		// Should not stop with empty steps
		require.False(t, reasoningCondition([]StepResult{}))
	})

	t.Run("FinishReasonIs", func(t *testing.T) {
		t.Parallel()
		stopCondition := FinishReasonIs(FinishReasonStop)
		lengthCondition := FinishReasonIs(FinishReasonLength)

		// Should not stop when finish reason doesn't match
		require.False(t, stopCondition([]StepResult{step1}))

		// Should stop when finish reason matches in last step
		require.True(t, stopCondition([]StepResult{step1, step2}))
		require.True(t, lengthCondition([]StepResult{step1, step2, step3}))

		// Should not stop with empty steps
		require.False(t, stopCondition([]StepResult{}))
	})

	t.Run("MaxTokensUsed", func(t *testing.T) {
		condition := MaxTokensUsed(30)

		// Should not stop when under limit
		require.False(t, condition([]StepResult{step1}))        // 10 tokens
		require.False(t, condition([]StepResult{step1, step2})) // 25 tokens

		// Should stop when at or over limit
		require.True(t, condition([]StepResult{step1, step2, step3})) // 45 tokens

		// Should not stop with empty steps
		require.False(t, condition([]StepResult{}))
	})
}

func TestStopConditions_Integration(t *testing.T) {
	t.Parallel()

	t.Run("StepCountIs integration", func(t *testing.T) {
		t.Parallel()
		model := &mockLanguageModel{
			generateFunc: func(ctx context.Context, call Call) (*Response, error) {
				return &Response{
					Content: ResponseContent{
						TextContent{Text: "Mock response"},
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

		agent := NewAgent(model, WithStopConditions(StepCountIs(1)))

		result, err := agent.Generate(context.Background(), AgentCall{
			Prompt: "test prompt",
		})

		require.NoError(t, err)
		require.NotNil(t, result)
		require.Len(t, result.Steps, 1) // Should stop after 1 step
	})

	t.Run("Multiple stop conditions", func(t *testing.T) {
		t.Parallel()
		model := &mockLanguageModel{
			generateFunc: func(ctx context.Context, call Call) (*Response, error) {
				return &Response{
					Content: ResponseContent{
						TextContent{Text: "Mock response"},
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

		agent := NewAgent(model, WithStopConditions(
			StepCountIs(5),                   // Stop after 5 steps
			FinishReasonIs(FinishReasonStop), // Or stop on finish reason
		))

		result, err := agent.Generate(context.Background(), AgentCall{
			Prompt: "test prompt",
		})

		require.NoError(t, err)
		require.NotNil(t, result)
		// Should stop on first condition met (finish reason stop)
		require.Equal(t, FinishReasonStop, result.Response.FinishReason)
	})
}

func TestPrepareStep(t *testing.T) {
	t.Parallel()

	t.Run("System prompt modification", func(t *testing.T) {
		t.Parallel()
		var capturedSystemPrompt string
		model := &mockLanguageModel{
			generateFunc: func(ctx context.Context, call Call) (*Response, error) {
				// Capture the system message to verify it was modified
				if len(call.Prompt) > 0 && call.Prompt[0].Role == MessageRoleSystem {
					if len(call.Prompt[0].Content) > 0 {
						if textPart, ok := AsContentType[TextPart](call.Prompt[0].Content[0]); ok {
							capturedSystemPrompt = textPart.Text
						}
					}
				}
				return &Response{
					Content: ResponseContent{
						TextContent{Text: "Response"},
					},
					Usage:        Usage{TotalTokens: 10},
					FinishReason: FinishReasonStop,
				}, nil
			},
		}

		prepareStepFunc := func(ctx context.Context, options PrepareStepFunctionOptions) (context.Context, PrepareStepResult, error) {
			newSystem := "Modified system prompt for step " + fmt.Sprintf("%d", options.StepNumber)
			return ctx, PrepareStepResult{
				Model:    options.Model,
				Messages: options.Messages,
				System:   &newSystem,
			}, nil
		}

		agent := NewAgent(model, WithSystemPrompt("Original system prompt"))

		result, err := agent.Generate(context.Background(), AgentCall{
			Prompt:      "test prompt",
			PrepareStep: prepareStepFunc,
		})

		require.NoError(t, err)
		require.NotNil(t, result)
		require.Equal(t, "Modified system prompt for step 0", capturedSystemPrompt)
	})

	t.Run("Tool choice modification", func(t *testing.T) {
		t.Parallel()
		var capturedToolChoice *ToolChoice
		model := &mockLanguageModel{
			generateFunc: func(ctx context.Context, call Call) (*Response, error) {
				capturedToolChoice = call.ToolChoice
				return &Response{
					Content: ResponseContent{
						TextContent{Text: "Response"},
					},
					Usage:        Usage{TotalTokens: 10},
					FinishReason: FinishReasonStop,
				}, nil
			},
		}

		prepareStepFunc := func(ctx context.Context, options PrepareStepFunctionOptions) (context.Context, PrepareStepResult, error) {
			toolChoice := ToolChoiceNone
			return ctx, PrepareStepResult{
				Model:      options.Model,
				Messages:   options.Messages,
				ToolChoice: &toolChoice,
			}, nil
		}

		agent := NewAgent(model)

		result, err := agent.Generate(context.Background(), AgentCall{
			Prompt:      "test prompt",
			PrepareStep: prepareStepFunc,
		})

		require.NoError(t, err)
		require.NotNil(t, result)
		require.NotNil(t, capturedToolChoice)
		require.Equal(t, ToolChoiceNone, *capturedToolChoice)
	})

	t.Run("Active tools modification", func(t *testing.T) {
		t.Parallel()
		var capturedToolNames []string
		model := &mockLanguageModel{
			generateFunc: func(ctx context.Context, call Call) (*Response, error) {
				// Capture tool names to verify active tools were modified
				for _, tool := range call.Tools {
					capturedToolNames = append(capturedToolNames, tool.GetName())
				}
				return &Response{
					Content: ResponseContent{
						TextContent{Text: "Response"},
					},
					Usage:        Usage{TotalTokens: 10},
					FinishReason: FinishReasonStop,
				}, nil
			},
		}

		tool1 := &mockTool{name: "tool1", description: "Tool 1"}
		tool2 := &mockTool{name: "tool2", description: "Tool 2"}
		tool3 := &mockTool{name: "tool3", description: "Tool 3"}

		prepareStepFunc := func(ctx context.Context, options PrepareStepFunctionOptions) (context.Context, PrepareStepResult, error) {
			activeTools := []string{"tool2"} // Only tool2 should be active
			return ctx, PrepareStepResult{
				Model:       options.Model,
				Messages:    options.Messages,
				ActiveTools: activeTools,
			}, nil
		}

		agent := NewAgent(model, WithTools(tool1, tool2, tool3))

		result, err := agent.Generate(context.Background(), AgentCall{
			Prompt:      "test prompt",
			PrepareStep: prepareStepFunc,
		})

		require.NoError(t, err)
		require.NotNil(t, result)
		require.Len(t, capturedToolNames, 1)
		require.Equal(t, "tool2", capturedToolNames[0])
	})

	t.Run("No tools when DisableAllTools is true", func(t *testing.T) {
		t.Parallel()
		var capturedToolCount int
		model := &mockLanguageModel{
			generateFunc: func(ctx context.Context, call Call) (*Response, error) {
				capturedToolCount = len(call.Tools)
				return &Response{
					Content: ResponseContent{
						TextContent{Text: "Response"},
					},
					Usage:        Usage{TotalTokens: 10},
					FinishReason: FinishReasonStop,
				}, nil
			},
		}

		tool1 := &mockTool{name: "tool1", description: "Tool 1"}

		prepareStepFunc := func(ctx context.Context, options PrepareStepFunctionOptions) (context.Context, PrepareStepResult, error) {
			return ctx, PrepareStepResult{
				Model:           options.Model,
				Messages:        options.Messages,
				DisableAllTools: true, // Disable all tools for this step
			}, nil
		}

		agent := NewAgent(model, WithTools(tool1))

		result, err := agent.Generate(context.Background(), AgentCall{
			Prompt:      "test prompt",
			PrepareStep: prepareStepFunc,
		})

		require.NoError(t, err)
		require.NotNil(t, result)
		require.Equal(t, 0, capturedToolCount) // No tools should be passed
	})

	t.Run("All fields modified together", func(t *testing.T) {
		t.Parallel()
		var capturedSystemPrompt string
		var capturedToolChoice *ToolChoice
		var capturedToolNames []string

		model := &mockLanguageModel{
			generateFunc: func(ctx context.Context, call Call) (*Response, error) {
				// Capture system prompt
				if len(call.Prompt) > 0 && call.Prompt[0].Role == MessageRoleSystem {
					if len(call.Prompt[0].Content) > 0 {
						if textPart, ok := AsContentType[TextPart](call.Prompt[0].Content[0]); ok {
							capturedSystemPrompt = textPart.Text
						}
					}
				}
				// Capture tool choice
				capturedToolChoice = call.ToolChoice
				// Capture tool names
				for _, tool := range call.Tools {
					capturedToolNames = append(capturedToolNames, tool.GetName())
				}
				return &Response{
					Content: ResponseContent{
						TextContent{Text: "Response"},
					},
					Usage:        Usage{TotalTokens: 10},
					FinishReason: FinishReasonStop,
				}, nil
			},
		}

		tool1 := &mockTool{name: "tool1", description: "Tool 1"}
		tool2 := &mockTool{name: "tool2", description: "Tool 2"}

		prepareStepFunc := func(ctx context.Context, options PrepareStepFunctionOptions) (context.Context, PrepareStepResult, error) {
			newSystem := "Step-specific system"
			toolChoice := SpecificToolChoice("tool1")
			activeTools := []string{"tool1"}
			return ctx, PrepareStepResult{
				Model:       options.Model,
				Messages:    options.Messages,
				System:      &newSystem,
				ToolChoice:  &toolChoice,
				ActiveTools: activeTools,
			}, nil
		}

		agent := NewAgent(model, WithSystemPrompt("Original system"), WithTools(tool1, tool2))

		result, err := agent.Generate(context.Background(), AgentCall{
			Prompt:      "test prompt",
			PrepareStep: prepareStepFunc,
		})

		require.NoError(t, err)
		require.NotNil(t, result)
		require.Equal(t, "Step-specific system", capturedSystemPrompt)
		require.NotNil(t, capturedToolChoice)
		require.Equal(t, SpecificToolChoice("tool1"), *capturedToolChoice)
		require.Len(t, capturedToolNames, 1)
		require.Equal(t, "tool1", capturedToolNames[0])
	})

	t.Run("Nil fields use parent values", func(t *testing.T) {
		t.Parallel()
		var capturedSystemPrompt string
		var capturedToolChoice *ToolChoice
		var capturedToolNames []string

		model := &mockLanguageModel{
			generateFunc: func(ctx context.Context, call Call) (*Response, error) {
				// Capture system prompt
				if len(call.Prompt) > 0 && call.Prompt[0].Role == MessageRoleSystem {
					if len(call.Prompt[0].Content) > 0 {
						if textPart, ok := AsContentType[TextPart](call.Prompt[0].Content[0]); ok {
							capturedSystemPrompt = textPart.Text
						}
					}
				}
				// Capture tool choice
				capturedToolChoice = call.ToolChoice
				// Capture tool names
				for _, tool := range call.Tools {
					capturedToolNames = append(capturedToolNames, tool.GetName())
				}
				return &Response{
					Content: ResponseContent{
						TextContent{Text: "Response"},
					},
					Usage:        Usage{TotalTokens: 10},
					FinishReason: FinishReasonStop,
				}, nil
			},
		}

		tool1 := &mockTool{name: "tool1", description: "Tool 1"}

		prepareStepFunc := func(ctx context.Context, options PrepareStepFunctionOptions) (context.Context, PrepareStepResult, error) {
			// All optional fields are nil, should use parent values
			return ctx, PrepareStepResult{
				Model:       options.Model,
				Messages:    options.Messages,
				System:      nil, // Use parent
				ToolChoice:  nil, // Use parent (auto)
				ActiveTools: nil, // Use parent (all tools)
			}, nil
		}

		agent := NewAgent(model, WithSystemPrompt("Parent system"), WithTools(tool1))

		result, err := agent.Generate(context.Background(), AgentCall{
			Prompt:      "test prompt",
			PrepareStep: prepareStepFunc,
		})

		require.NoError(t, err)
		require.NotNil(t, result)
		require.Equal(t, "Parent system", capturedSystemPrompt)
		require.NotNil(t, capturedToolChoice)
		require.Equal(t, ToolChoiceAuto, *capturedToolChoice) // Default
		require.Len(t, capturedToolNames, 1)
		require.Equal(t, "tool1", capturedToolNames[0])
	})

	t.Run("Empty ActiveTools means all tools", func(t *testing.T) {
		t.Parallel()
		var capturedToolNames []string
		model := &mockLanguageModel{
			generateFunc: func(ctx context.Context, call Call) (*Response, error) {
				// Capture tool names to verify all tools are included
				for _, tool := range call.Tools {
					capturedToolNames = append(capturedToolNames, tool.GetName())
				}
				return &Response{
					Content: ResponseContent{
						TextContent{Text: "Response"},
					},
					Usage:        Usage{TotalTokens: 10},
					FinishReason: FinishReasonStop,
				}, nil
			},
		}

		tool1 := &mockTool{name: "tool1", description: "Tool 1"}
		tool2 := &mockTool{name: "tool2", description: "Tool 2"}

		prepareStepFunc := func(ctx context.Context, options PrepareStepFunctionOptions) (context.Context, PrepareStepResult, error) {
			return ctx, PrepareStepResult{
				Model:       options.Model,
				Messages:    options.Messages,
				ActiveTools: []string{}, // Empty slice means all tools
			}, nil
		}

		agent := NewAgent(model, WithTools(tool1, tool2))

		result, err := agent.Generate(context.Background(), AgentCall{
			Prompt:      "test prompt",
			PrepareStep: prepareStepFunc,
		})

		require.NoError(t, err)
		require.NotNil(t, result)
		require.Len(t, capturedToolNames, 2) // All tools should be included
		require.Contains(t, capturedToolNames, "tool1")
		require.Contains(t, capturedToolNames, "tool2")
	})
}

func TestToolCallRepair(t *testing.T) {
	t.Parallel()

	t.Run("Valid tool call passes validation", func(t *testing.T) {
		t.Parallel()
		model := &mockLanguageModel{
			generateFunc: func(ctx context.Context, call Call) (*Response, error) {
				return &Response{
					Content: ResponseContent{
						TextContent{Text: "Response"},
						ToolCallContent{
							ToolCallID: "call1",
							ToolName:   "test_tool",
							Input:      `{"value": "test"}`, // Valid JSON with required field
						},
					},
					Usage:        Usage{TotalTokens: 10},
					FinishReason: FinishReasonStop, // Changed to stop to avoid infinite loop
				}, nil
			},
		}

		tool := &mockTool{
			name:        "test_tool",
			description: "Test tool",
			parameters: map[string]any{
				"value": map[string]any{"type": "string"},
			},
			required: []string{"value"},
			executeFunc: func(ctx context.Context, call ToolCall) (ToolResponse, error) {
				return ToolResponse{Content: "success", IsError: false}, nil
			},
		}

		agent := NewAgent(model, WithTools(tool), WithStopConditions(StepCountIs(2))) // Limit steps

		result, err := agent.Generate(context.Background(), AgentCall{
			Prompt: "test prompt",
		})

		require.NoError(t, err)
		require.NotNil(t, result)
		require.Len(t, result.Steps, 1) // Only one step since FinishReason is stop

		// Check that tool call was executed successfully
		toolCalls := result.Steps[0].Content.ToolCalls()
		require.Len(t, toolCalls, 1)
		require.False(t, toolCalls[0].Invalid) // Should be valid
	})

	t.Run("Invalid tool call without repair function", func(t *testing.T) {
		t.Parallel()
		model := &mockLanguageModel{
			generateFunc: func(ctx context.Context, call Call) (*Response, error) {
				return &Response{
					Content: ResponseContent{
						TextContent{Text: "Response"},
						ToolCallContent{
							ToolCallID: "call1",
							ToolName:   "test_tool",
							Input:      `{"wrong_field": "test"}`, // Missing required field
						},
					},
					Usage:        Usage{TotalTokens: 10},
					FinishReason: FinishReasonStop, // Changed to stop to avoid infinite loop
				}, nil
			},
		}

		tool := &mockTool{
			name:        "test_tool",
			description: "Test tool",
			parameters: map[string]any{
				"value": map[string]any{"type": "string"},
			},
			required: []string{"value"},
		}

		agent := NewAgent(model, WithTools(tool), WithStopConditions(StepCountIs(2))) // Limit steps

		result, err := agent.Generate(context.Background(), AgentCall{
			Prompt: "test prompt",
		})

		require.NoError(t, err)
		require.NotNil(t, result)
		require.Len(t, result.Steps, 1) // Only one step

		// Check that tool call was marked as invalid
		toolCalls := result.Steps[0].Content.ToolCalls()
		require.Len(t, toolCalls, 1)
		require.True(t, toolCalls[0].Invalid) // Should be invalid
		require.Contains(t, toolCalls[0].ValidationError.Error(), "missing required parameter: value")
	})

	t.Run("Invalid tool call with successful repair", func(t *testing.T) {
		t.Parallel()
		model := &mockLanguageModel{
			generateFunc: func(ctx context.Context, call Call) (*Response, error) {
				return &Response{
					Content: ResponseContent{
						TextContent{Text: "Response"},
						ToolCallContent{
							ToolCallID: "call1",
							ToolName:   "test_tool",
							Input:      `{"wrong_field": "test"}`, // Missing required field
						},
					},
					Usage:        Usage{TotalTokens: 10},
					FinishReason: FinishReasonStop, // Changed to stop
				}, nil
			},
		}

		tool := &mockTool{
			name:        "test_tool",
			description: "Test tool",
			parameters: map[string]any{
				"value": map[string]any{"type": "string"},
			},
			required: []string{"value"},
			executeFunc: func(ctx context.Context, call ToolCall) (ToolResponse, error) {
				return ToolResponse{Content: "repaired_success", IsError: false}, nil
			},
		}

		repairFunc := func(ctx context.Context, options ToolCallRepairOptions) (*ToolCallContent, error) {
			// Simple repair: add the missing required field
			repairedToolCall := options.OriginalToolCall
			repairedToolCall.Input = `{"value": "repaired"}`
			return &repairedToolCall, nil
		}

		agent := NewAgent(model, WithTools(tool), WithRepairToolCall(repairFunc), WithStopConditions(StepCountIs(2)))

		result, err := agent.Generate(context.Background(), AgentCall{
			Prompt: "test prompt",
		})

		require.NoError(t, err)
		require.NotNil(t, result)
		require.Len(t, result.Steps, 1) // Only one step

		// Check that tool call was repaired and is now valid
		toolCalls := result.Steps[0].Content.ToolCalls()
		require.Len(t, toolCalls, 1)
		require.False(t, toolCalls[0].Invalid)                        // Should be valid after repair
		require.Equal(t, `{"value": "repaired"}`, toolCalls[0].Input) // Should have repaired input
	})

	t.Run("Invalid tool call with failed repair", func(t *testing.T) {
		t.Parallel()
		model := &mockLanguageModel{
			generateFunc: func(ctx context.Context, call Call) (*Response, error) {
				return &Response{
					Content: ResponseContent{
						TextContent{Text: "Response"},
						ToolCallContent{
							ToolCallID: "call1",
							ToolName:   "test_tool",
							Input:      `{"wrong_field": "test"}`, // Missing required field
						},
					},
					Usage:        Usage{TotalTokens: 10},
					FinishReason: FinishReasonStop, // Changed to stop
				}, nil
			},
		}

		tool := &mockTool{
			name:        "test_tool",
			description: "Test tool",
			parameters: map[string]any{
				"value": map[string]any{"type": "string"},
			},
			required: []string{"value"},
		}

		repairFunc := func(ctx context.Context, options ToolCallRepairOptions) (*ToolCallContent, error) {
			// Repair function fails
			return nil, errors.New("repair failed")
		}

		agent := NewAgent(model, WithTools(tool), WithRepairToolCall(repairFunc), WithStopConditions(StepCountIs(2)))

		result, err := agent.Generate(context.Background(), AgentCall{
			Prompt: "test prompt",
		})

		require.NoError(t, err)
		require.NotNil(t, result)
		require.Len(t, result.Steps, 1) // Only one step

		// Check that tool call was marked as invalid since repair failed
		toolCalls := result.Steps[0].Content.ToolCalls()
		require.Len(t, toolCalls, 1)
		require.True(t, toolCalls[0].Invalid) // Should be invalid
		require.Contains(t, toolCalls[0].ValidationError.Error(), "missing required parameter: value")
	})

	t.Run("Nonexistent tool call", func(t *testing.T) {
		t.Parallel()
		model := &mockLanguageModel{
			generateFunc: func(ctx context.Context, call Call) (*Response, error) {
				return &Response{
					Content: ResponseContent{
						TextContent{Text: "Response"},
						ToolCallContent{
							ToolCallID: "call1",
							ToolName:   "nonexistent_tool",
							Input:      `{"value": "test"}`,
						},
					},
					Usage:        Usage{TotalTokens: 10},
					FinishReason: FinishReasonStop, // Changed to stop
				}, nil
			},
		}

		tool := &mockTool{name: "test_tool", description: "Test tool"}

		agent := NewAgent(model, WithTools(tool), WithStopConditions(StepCountIs(2)))

		result, err := agent.Generate(context.Background(), AgentCall{
			Prompt: "test prompt",
		})

		require.NoError(t, err)
		require.NotNil(t, result)
		require.Len(t, result.Steps, 1) // Only one step

		// Check that tool call was marked as invalid due to nonexistent tool
		toolCalls := result.Steps[0].Content.ToolCalls()
		require.Len(t, toolCalls, 1)
		require.True(t, toolCalls[0].Invalid) // Should be invalid
		require.Contains(t, toolCalls[0].ValidationError.Error(), "tool not found: nonexistent_tool")
	})

	t.Run("Invalid JSON in tool call", func(t *testing.T) {
		t.Parallel()
		model := &mockLanguageModel{
			generateFunc: func(ctx context.Context, call Call) (*Response, error) {
				return &Response{
					Content: ResponseContent{
						TextContent{Text: "Response"},
						ToolCallContent{
							ToolCallID: "call1",
							ToolName:   "test_tool",
							Input:      `{invalid json}`, // Invalid JSON
						},
					},
					Usage:        Usage{TotalTokens: 10},
					FinishReason: FinishReasonStop, // Changed to stop
				}, nil
			},
		}

		tool := &mockTool{
			name:        "test_tool",
			description: "Test tool",
			parameters: map[string]any{
				"value": map[string]any{"type": "string"},
			},
			required: []string{"value"},
		}

		agent := NewAgent(model, WithTools(tool), WithStopConditions(StepCountIs(2)))

		result, err := agent.Generate(context.Background(), AgentCall{
			Prompt: "test prompt",
		})

		require.NoError(t, err)
		require.NotNil(t, result)
		require.Len(t, result.Steps, 1) // Only one step

		// Check that tool call was marked as invalid due to invalid JSON
		toolCalls := result.Steps[0].Content.ToolCalls()
		require.Len(t, toolCalls, 1)
		require.True(t, toolCalls[0].Invalid) // Should be invalid
		require.Contains(t, toolCalls[0].ValidationError.Error(), "invalid JSON input")
	})
}
