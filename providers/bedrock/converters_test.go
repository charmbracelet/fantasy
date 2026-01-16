package bedrock

import (
	"context"
	"encoding/json"
	"testing"

	"charm.land/fantasy"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestConvertTextMessage tests conversion of text messages.
func TestConvertTextMessage(t *testing.T) {
	model := createTestModel(t)

	call := fantasy.Call{
		Prompt: fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Hello, world!"},
				},
			},
		},
	}

	request, _, err := model.prepareConverseRequest(call)
	require.NoError(t, err)
	require.NotNil(t, request)

	// Verify message structure
	assert.Len(t, request.Messages, 1)
	assert.Equal(t, types.ConversationRoleUser, request.Messages[0].Role)
	assert.Len(t, request.Messages[0].Content, 1)

	// Verify text content
	textBlock, ok := request.Messages[0].Content[0].(*types.ContentBlockMemberText)
	require.True(t, ok, "Expected text content block")
	assert.Equal(t, "Hello, world!", textBlock.Value)
}

// TestConvertImageAttachment tests conversion of image attachments.
func TestConvertImageAttachment(t *testing.T) {
	model := createTestModel(t)

	imageData := []byte{0xFF, 0xD8, 0xFF, 0xE0} // JPEG header

	call := fantasy.Call{
		Prompt: fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Check this image:"},
					fantasy.FilePart{
						Data:      imageData,
						MediaType: "image/jpeg",
					},
				},
			},
		},
	}

	request, _, err := model.prepareConverseRequest(call)
	require.NoError(t, err)
	require.NotNil(t, request)

	// Verify message structure
	assert.Len(t, request.Messages, 1)
	assert.Len(t, request.Messages[0].Content, 2)

	// Verify text content
	textBlock, ok := request.Messages[0].Content[0].(*types.ContentBlockMemberText)
	require.True(t, ok, "Expected text content block")
	assert.Equal(t, "Check this image:", textBlock.Value)

	// Verify image content
	imageBlock, ok := request.Messages[0].Content[1].(*types.ContentBlockMemberImage)
	require.True(t, ok, "Expected image content block")
	assert.Equal(t, types.ImageFormatJpeg, imageBlock.Value.Format)

	// Verify image data
	imageSource, ok := imageBlock.Value.Source.(*types.ImageSourceMemberBytes)
	require.True(t, ok, "Expected bytes image source")
	assert.Equal(t, imageData, imageSource.Value)
}

// TestConvertToolCall tests conversion of tool calls.
func TestConvertToolCall(t *testing.T) {
	model := createTestModel(t)

	toolInput := map[string]interface{}{
		"query": "test query",
		"limit": 10,
	}
	toolInputJSON, err := json.Marshal(toolInput)
	require.NoError(t, err)

	call := fantasy.Call{
		Prompt: fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Search for something"},
				},
			},
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.ToolCallPart{
						ToolCallID: "call_123",
						ToolName:   "search",
						Input:      string(toolInputJSON),
					},
				},
			},
		},
	}

	request, _, err := model.prepareConverseRequest(call)
	require.NoError(t, err)
	require.NotNil(t, request)

	// Verify message structure
	assert.Len(t, request.Messages, 2)

	// Verify assistant message with tool call
	assert.Equal(t, types.ConversationRoleAssistant, request.Messages[1].Role)
	assert.Len(t, request.Messages[1].Content, 1)

	// Verify tool use content
	toolUseBlock, ok := request.Messages[1].Content[0].(*types.ContentBlockMemberToolUse)
	require.True(t, ok, "Expected tool use content block")
	assert.Equal(t, "call_123", *toolUseBlock.Value.ToolUseId)
	assert.Equal(t, "search", *toolUseBlock.Value.Name)
	assert.NotNil(t, toolUseBlock.Value.Input)
}

// TestConvertToolResult tests conversion of tool results.
func TestConvertToolResult(t *testing.T) {
	model := createTestModel(t)

	call := fantasy.Call{
		Prompt: fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Search for something"},
				},
			},
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.ToolCallPart{
						ToolCallID: "call_123",
						ToolName:   "search",
						Input:      `{"query":"test"}`,
					},
				},
			},
			{
				Role: fantasy.MessageRoleTool,
				Content: []fantasy.MessagePart{
					fantasy.ToolResultPart{
						ToolCallID: "call_123",
						Output: fantasy.ToolResultOutputContentText{
							Text: "Search results: found 5 items",
						},
					},
				},
			},
		},
	}

	request, _, err := model.prepareConverseRequest(call)
	require.NoError(t, err)
	require.NotNil(t, request)

	// Verify message structure (tool results are sent as user messages)
	assert.Len(t, request.Messages, 3)

	// Verify tool result message
	assert.Equal(t, types.ConversationRoleUser, request.Messages[2].Role)
	assert.Len(t, request.Messages[2].Content, 1)

	// Verify tool result content
	toolResultBlock, ok := request.Messages[2].Content[0].(*types.ContentBlockMemberToolResult)
	require.True(t, ok, "Expected tool result content block")
	assert.Equal(t, "call_123", *toolResultBlock.Value.ToolUseId)
	assert.Len(t, toolResultBlock.Value.Content, 1)

	// Verify result text
	resultText, ok := toolResultBlock.Value.Content[0].(*types.ToolResultContentBlockMemberText)
	require.True(t, ok, "Expected text result content")
	assert.Equal(t, "Search results: found 5 items", resultText.Value)
}

// TestConvertToolResultError tests conversion of tool result errors.
func TestConvertToolResultError(t *testing.T) {
	model := createTestModel(t)

	call := fantasy.Call{
		Prompt: fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Search for something"},
				},
			},
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.ToolCallPart{
						ToolCallID: "call_123",
						ToolName:   "search",
						Input:      `{"query":"test"}`,
					},
				},
			},
			{
				Role: fantasy.MessageRoleTool,
				Content: []fantasy.MessagePart{
					fantasy.ToolResultPart{
						ToolCallID: "call_123",
						Output: fantasy.ToolResultOutputContentError{
							Error: assert.AnError,
						},
					},
				},
			},
		},
	}

	request, _, err := model.prepareConverseRequest(call)
	require.NoError(t, err)
	require.NotNil(t, request)

	// Verify tool result message
	assert.Len(t, request.Messages, 3)
	toolResultBlock, ok := request.Messages[2].Content[0].(*types.ContentBlockMemberToolResult)
	require.True(t, ok, "Expected tool result content block")

	// Verify error is converted to text
	resultText, ok := toolResultBlock.Value.Content[0].(*types.ToolResultContentBlockMemberText)
	require.True(t, ok, "Expected text result content")
	assert.Contains(t, resultText.Value, "assert.AnError")
}

// TestConvertSystemMessage tests conversion of system messages.
func TestConvertSystemMessage(t *testing.T) {
	model := createTestModel(t)

	call := fantasy.Call{
		Prompt: fantasy.Prompt{
			{
				Role: fantasy.MessageRoleSystem,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "You are a helpful assistant."},
				},
			},
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Hello!"},
				},
			},
		},
	}

	request, _, err := model.prepareConverseRequest(call)
	require.NoError(t, err)
	require.NotNil(t, request)

	// Verify system blocks
	assert.Len(t, request.System, 1)
	systemBlock, ok := request.System[0].(*types.SystemContentBlockMemberText)
	require.True(t, ok, "Expected text system block")
	assert.Equal(t, "You are a helpful assistant.", systemBlock.Value)

	// Verify user message
	assert.Len(t, request.Messages, 1)
	assert.Equal(t, types.ConversationRoleUser, request.Messages[0].Role)
}

// TestConvertMultiTurnConversation tests conversion of multi-turn conversations.
func TestConvertMultiTurnConversation(t *testing.T) {
	model := createTestModel(t)

	call := fantasy.Call{
		Prompt: fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "What is 2+2?"},
				},
			},
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "2+2 equals 4."},
				},
			},
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "What about 3+3?"},
				},
			},
		},
	}

	request, _, err := model.prepareConverseRequest(call)
	require.NoError(t, err)
	require.NotNil(t, request)

	// Verify message structure
	assert.Len(t, request.Messages, 3)
	assert.Equal(t, types.ConversationRoleUser, request.Messages[0].Role)
	assert.Equal(t, types.ConversationRoleAssistant, request.Messages[1].Role)
	assert.Equal(t, types.ConversationRoleUser, request.Messages[2].Role)

	// Verify content
	textBlock0, ok := request.Messages[0].Content[0].(*types.ContentBlockMemberText)
	require.True(t, ok)
	assert.Equal(t, "What is 2+2?", textBlock0.Value)

	textBlock1, ok := request.Messages[1].Content[0].(*types.ContentBlockMemberText)
	require.True(t, ok)
	assert.Equal(t, "2+2 equals 4.", textBlock1.Value)

	textBlock2, ok := request.Messages[2].Content[0].(*types.ContentBlockMemberText)
	require.True(t, ok)
	assert.Equal(t, "What about 3+3?", textBlock2.Value)
}

// createTestModel creates a test nova language model instance.
func createTestModel(t *testing.T) *novaLanguageModel {
	cfg, err := config.LoadDefaultConfig(context.Background())
	if err != nil {
		t.Skip("AWS configuration not available")
	}

	client := bedrockruntime.NewFromConfig(cfg)
	return &novaLanguageModel{
		modelID:  "amazon.nova-pro-v1:0",
		provider: Name,
		client:   client,
		options:  options{},
	}
}
