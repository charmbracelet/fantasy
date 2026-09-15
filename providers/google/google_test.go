package google

import (
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"
)

func TestToGooglePrompt_ThoughtSignatures(t *testing.T) {
	t.Parallel()

	t.Run("single tool call with thought signature", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Fetch weather"},
				},
			},
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.ReasoningPart{
						Text: "Looking up weather in SF...",
						ProviderOptions: fantasy.ProviderOptions{
							Name: &ReasoningMetadata{
								Signature: "sig_weather_1",
								ToolID:    "call_weather_1",
							},
						},
					},
					fantasy.ToolCallPart{
						ToolCallID: "call_weather_1",
						ToolName:   "get_weather",
						Input:      `{"location":"San Francisco"}`,
					},
				},
			},
		}

		_, content, _ := toGooglePrompt(prompt, false)
		require.Len(t, content, 2)

		modelContent := content[1]
		require.Equal(t, genai.RoleModel, modelContent.Role)
		require.Len(t, modelContent.Parts, 1)

		part := modelContent.Parts[0]
		require.NotNil(t, part.FunctionCall)
		assert.Equal(t, "call_weather_1", part.FunctionCall.ID)
		assert.Equal(t, "get_weather", part.FunctionCall.Name)
		assert.Equal(t, []byte("sig_weather_1"), part.ThoughtSignature)
	})

	t.Run("multiple sequential tool calls with distinct signatures", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.ReasoningPart{
						Text: "First thinking step",
						ProviderOptions: fantasy.ProviderOptions{
							Name: &ReasoningMetadata{
								Signature: "sig_call_1",
								ToolID:    "call_1",
							},
						},
					},
					fantasy.ReasoningPart{
						Text: "Second thinking step",
						ProviderOptions: fantasy.ProviderOptions{
							Name: &ReasoningMetadata{
								Signature: "sig_call_2",
								ToolID:    "call_2",
							},
						},
					},
					fantasy.ToolCallPart{
						ToolCallID: "call_1",
						ToolName:   "tool_one",
						Input:      `{"step":1}`,
					},
					fantasy.ToolCallPart{
						ToolCallID: "call_2",
						ToolName:   "tool_two",
						Input:      `{"step":2}`,
					},
				},
			},
		}

		_, content, _ := toGooglePrompt(prompt, false)
		require.Len(t, content, 1)

		modelContent := content[0]
		require.Equal(t, genai.RoleModel, modelContent.Role)
		require.Len(t, modelContent.Parts, 2)

		part1 := modelContent.Parts[0]
		require.NotNil(t, part1.FunctionCall)
		assert.Equal(t, "call_1", part1.FunctionCall.ID)
		assert.Equal(t, []byte("sig_call_1"), part1.ThoughtSignature)

		part2 := modelContent.Parts[1]
		require.NotNil(t, part2.FunctionCall)
		assert.Equal(t, "call_2", part2.FunctionCall.ID)
		assert.Equal(t, []byte("sig_call_2"), part2.ThoughtSignature)
	})

	t.Run("parallel tool calls where only first has signature", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.ReasoningPart{
						Text: "Thinking for parallel tools",
						ProviderOptions: fantasy.ProviderOptions{
							Name: &ReasoningMetadata{
								Signature: "sig_parallel_first",
								ToolID:    "call_p1",
							},
						},
					},
					fantasy.ToolCallPart{
						ToolCallID: "call_p1",
						ToolName:   "fetch_file",
						Input:      `{"path":"a.go"}`,
					},
					fantasy.ToolCallPart{
						ToolCallID: "call_p2",
						ToolName:   "fetch_file",
						Input:      `{"path":"b.go"}`,
					},
				},
			},
		}

		_, content, _ := toGooglePrompt(prompt, false)
		require.Len(t, content, 1)

		modelContent := content[0]
		require.Equal(t, genai.RoleModel, modelContent.Role)
		require.Len(t, modelContent.Parts, 2)

		part1 := modelContent.Parts[0]
		require.NotNil(t, part1.FunctionCall)
		assert.Equal(t, "call_p1", part1.FunctionCall.ID)
		assert.Equal(t, []byte("sig_parallel_first"), part1.ThoughtSignature)

		part2 := modelContent.Parts[1]
		require.NotNil(t, part2.FunctionCall)
		assert.Equal(t, "call_p2", part2.FunctionCall.ID)
		assert.Nil(t, part2.ThoughtSignature)
	})

	t.Run("turn reasoning without tool ID attaches to single tool call", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.ReasoningPart{
						Text: "General reasoning",
						ProviderOptions: fantasy.ProviderOptions{
							Name: &ReasoningMetadata{
								Signature: "sig_general",
								ToolID:    "",
							},
						},
					},
					fantasy.ToolCallPart{
						ToolCallID: "call_single",
						ToolName:   "exec_cmd",
						Input:      `{"cmd":"ls"}`,
					},
				},
			},
		}

		_, content, _ := toGooglePrompt(prompt, false)
		require.Len(t, content, 1)

		modelContent := content[0]
		require.Len(t, modelContent.Parts, 1)
		assert.Equal(t, []byte("sig_general"), modelContent.Parts[0].ThoughtSignature)
	})

	t.Run("text response with thought signature", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.ReasoningPart{
						Text: "Deep thinking about answer",
						ProviderOptions: fantasy.ProviderOptions{
							Name: &ReasoningMetadata{
								Signature: "sig_text_1",
							},
						},
					},
					fantasy.TextPart{
						Text: "Here is the answer.",
					},
				},
			},
		}

		_, content, _ := toGooglePrompt(prompt, false)
		require.Len(t, content, 1)

		modelContent := content[0]
		require.Len(t, modelContent.Parts, 1)
		assert.Equal(t, "Here is the answer.", modelContent.Parts[0].Text)
		assert.Equal(t, []byte("sig_text_1"), modelContent.Parts[0].ThoughtSignature)
	})

	t.Run("non-thinking assistant turn has no signatures", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Plain text without thinking"},
					fantasy.ToolCallPart{
						ToolCallID: "call_plain",
						ToolName:   "read_file",
						Input:      `{"file":"main.go"}`,
					},
				},
			},
		}

		_, content, _ := toGooglePrompt(prompt, false)
		require.Len(t, content, 1)

		modelContent := content[0]
		require.Len(t, modelContent.Parts, 2)
		assert.Nil(t, modelContent.Parts[0].ThoughtSignature)
		assert.Nil(t, modelContent.Parts[1].ThoughtSignature)
	})

	t.Run("vertex AI preserves thought signature and clears function call ID", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.ReasoningPart{
						Text: "Vertex reasoning",
						ProviderOptions: fantasy.ProviderOptions{
							Name: &ReasoningMetadata{
								Signature: "sig_vertex",
								ToolID:    "call_v1",
							},
						},
					},
					fantasy.ToolCallPart{
						ToolCallID: "call_v1",
						ToolName:   "vertex_tool",
						Input:      `{"key":"val"}`,
					},
				},
			},
		}

		_, content, _ := toGooglePrompt(prompt, true)
		require.Len(t, content, 1)

		modelContent := content[0]
		require.Len(t, modelContent.Parts, 1)
		assert.Equal(t, "", modelContent.Parts[0].FunctionCall.ID)
		assert.Equal(t, []byte("sig_vertex"), modelContent.Parts[0].ThoughtSignature)
	})
}

func TestMapResponse_MultipleFunctionCalls_ThoughtSignatures(t *testing.T) {
	t.Parallel()

	lm := languageModel{
		providerOptions: options{
			toolCallIDFunc: func() string { return "generated_id" },
		},
	}

	response := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{
							Thought:          true,
							Text:             "Reasoning for first call",
							ThoughtSignature: []byte("sig_func_1"),
						},
						{
							FunctionCall: &genai.FunctionCall{
								ID:   "func_1",
								Name: "first_tool",
								Args: map[string]any{"arg": "1"},
							},
							ThoughtSignature: []byte("sig_func_1"),
						},
						{
							Thought:          true,
							Text:             "Reasoning for second call",
							ThoughtSignature: []byte("sig_func_2"),
						},
						{
							FunctionCall: &genai.FunctionCall{
								ID:   "func_2",
								Name: "second_tool",
								Args: map[string]any{"arg": "2"},
							},
							ThoughtSignature: []byte("sig_func_2"),
						},
					},
				},
				FinishReason: genai.FinishReasonStop,
			},
		},
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     10,
			CandidatesTokenCount: 20,
			TotalTokenCount:      30,
			ThoughtsTokenCount:   15,
		},
	}

	mapped, err := lm.mapResponse(response, nil)
	require.NoError(t, err)
	require.NotNil(t, mapped)
	require.Len(t, mapped.Content, 4)

	// Verify first reasoning block
	r1, ok := fantasy.AsContentType[fantasy.ReasoningContent](mapped.Content[0])
	require.True(t, ok)
	assert.Equal(t, "Reasoning for first call", r1.Text)
	rm1 := GetReasoningMetadata(fantasy.ProviderOptions(r1.ProviderMetadata))
	require.NotNil(t, rm1)
	assert.Equal(t, "sig_func_1", rm1.Signature)
	assert.Equal(t, "func_1", rm1.ToolID)

	// Verify first tool call
	tc1, ok := fantasy.AsContentType[fantasy.ToolCallContent](mapped.Content[1])
	require.True(t, ok)
	assert.Equal(t, "func_1", tc1.ToolCallID)
	assert.Equal(t, "first_tool", tc1.ToolName)

	// Verify second reasoning block
	r2, ok := fantasy.AsContentType[fantasy.ReasoningContent](mapped.Content[2])
	require.True(t, ok)
	assert.Equal(t, "Reasoning for second call", r2.Text)
	rm2 := GetReasoningMetadata(fantasy.ProviderOptions(r2.ProviderMetadata))
	require.NotNil(t, rm2)
	assert.Equal(t, "sig_func_2", rm2.Signature)
	assert.Equal(t, "func_2", rm2.ToolID)

	// Verify second tool call
	tc2, ok := fantasy.AsContentType[fantasy.ToolCallContent](mapped.Content[3])
	require.True(t, ok)
	assert.Equal(t, "func_2", tc2.ToolCallID)
	assert.Equal(t, "second_tool", tc2.ToolName)
}

func TestGetReasoningMetadata(t *testing.T) {
	t.Parallel()

	rm := &ReasoningMetadata{
		Signature: "test_sig",
		ToolID:    "test_tool",
	}

	t.Run("valid options", func(t *testing.T) {
		t.Parallel()
		opts := fantasy.ProviderOptions{
			Name: rm,
		}
		extracted := GetReasoningMetadata(opts)
		require.NotNil(t, extracted)
		assert.Equal(t, "test_sig", extracted.Signature)
		assert.Equal(t, "test_tool", extracted.ToolID)
	})

	t.Run("nil or empty options", func(t *testing.T) {
		t.Parallel()
		assert.Nil(t, GetReasoningMetadata(nil))
		assert.Nil(t, GetReasoningMetadata(fantasy.ProviderOptions{}))
		assert.Nil(t, GetReasoningMetadata(fantasy.ProviderOptions{"other": rm}))
	})
}
