package openai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"charm.land/fantasy"
	"github.com/charmbracelet/openai-go/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestToOpenAiPrompt_SystemMessages(t *testing.T) {
	t.Parallel()

	t.Run("should forward system messages", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleSystem,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "You are a helpful assistant."},
				},
			},
		}

		messages, warnings := DefaultToPrompt(prompt, "openai", "gpt-5")

		require.Empty(t, warnings)
		require.Len(t, messages, 1)

		systemMsg := messages[0].OfSystem
		require.NotNil(t, systemMsg)
		require.Equal(t, "You are a helpful assistant.", systemMsg.Content.OfString.Value)
	})

	t.Run("should handle empty system messages", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role:    fantasy.MessageRoleSystem,
				Content: []fantasy.MessagePart{},
			},
		}

		messages, warnings := DefaultToPrompt(prompt, "openai", "gpt-5")

		require.Len(t, warnings, 1)
		require.Contains(t, warnings[0].Message, "system prompt has no text parts")
		require.Empty(t, messages)
	})

	t.Run("should join multiple system text parts", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleSystem,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "You are a helpful assistant."},
					fantasy.TextPart{Text: "Be concise."},
				},
			},
		}

		messages, warnings := DefaultToPrompt(prompt, "openai", "gpt-5")

		require.Empty(t, warnings)
		require.Len(t, messages, 1)

		systemMsg := messages[0].OfSystem
		require.NotNil(t, systemMsg)
		require.Equal(t, "You are a helpful assistant.\nBe concise.", systemMsg.Content.OfString.Value)
	})
}

func TestToOpenAiPrompt_UserMessages(t *testing.T) {
	t.Parallel()

	t.Run("should convert messages with only a text part to a string content", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Hello"},
				},
			},
		}

		messages, warnings := DefaultToPrompt(prompt, "openai", "gpt-5")

		require.Empty(t, warnings)
		require.Len(t, messages, 1)

		userMsg := messages[0].OfUser
		require.NotNil(t, userMsg)
		require.Equal(t, "Hello", userMsg.Content.OfString.Value)
	})

	t.Run("should convert messages with image parts", func(t *testing.T) {
		t.Parallel()

		imageData := []byte{0, 1, 2, 3}
		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Hello"},
					fantasy.FilePart{
						MediaType: "image/png",
						Data:      imageData,
					},
				},
			},
		}

		messages, warnings := DefaultToPrompt(prompt, "openai", "gpt-5")

		require.Empty(t, warnings)
		require.Len(t, messages, 1)

		userMsg := messages[0].OfUser
		require.NotNil(t, userMsg)

		content := userMsg.Content.OfArrayOfContentParts
		require.Len(t, content, 2)

		// Check text part
		textPart := content[0].OfText
		require.NotNil(t, textPart)
		require.Equal(t, "Hello", textPart.Text)

		// Check image part
		imagePart := content[1].OfImageURL
		require.NotNil(t, imagePart)
		expectedURL := "data:image/png;base64," + base64.StdEncoding.EncodeToString(imageData)
		require.Equal(t, expectedURL, imagePart.ImageURL.URL)
	})

	t.Run("should add image detail when specified through provider options", func(t *testing.T) {
		t.Parallel()

		imageData := []byte{0, 1, 2, 3}
		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.FilePart{
						MediaType: "image/png",
						Data:      imageData,
						ProviderOptions: NewProviderFileOptions(&ProviderFileOptions{
							ImageDetail: "low",
						}),
					},
				},
			},
		}

		messages, warnings := DefaultToPrompt(prompt, "openai", "gpt-5")

		require.Empty(t, warnings)
		require.Len(t, messages, 1)

		userMsg := messages[0].OfUser
		require.NotNil(t, userMsg)

		content := userMsg.Content.OfArrayOfContentParts
		require.Len(t, content, 1)

		imagePart := content[0].OfImageURL
		require.NotNil(t, imagePart)
		require.Equal(t, "low", imagePart.ImageURL.Detail)
	})
}

func TestToOpenAiPrompt_FileParts(t *testing.T) {
	t.Parallel()

	t.Run("should throw for unsupported mime types", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.FilePart{
						MediaType: "application/something",
						Data:      []byte("test"),
					},
				},
			},
		}

		messages, warnings := DefaultToPrompt(prompt, "openai", "gpt-5")

		require.Len(t, warnings, 2) // unsupported type + empty message
		require.Contains(t, warnings[0].Message, "file part media type application/something not supported")
		require.Contains(t, warnings[1].Message, "dropping empty user message")
		require.Empty(t, messages) // Message is now dropped because it's empty
	})

	t.Run("should add audio content for audio/wav file parts", func(t *testing.T) {
		t.Parallel()

		audioData := []byte{0, 1, 2, 3}
		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.FilePart{
						MediaType: "audio/wav",
						Data:      audioData,
					},
				},
			},
		}

		messages, warnings := DefaultToPrompt(prompt, "openai", "gpt-5")

		require.Empty(t, warnings)
		require.Len(t, messages, 1)

		userMsg := messages[0].OfUser
		require.NotNil(t, userMsg)

		content := userMsg.Content.OfArrayOfContentParts
		require.Len(t, content, 1)

		audioPart := content[0].OfInputAudio
		require.NotNil(t, audioPart)
		require.Equal(t, base64.StdEncoding.EncodeToString(audioData), audioPart.InputAudio.Data)
		require.Equal(t, "wav", audioPart.InputAudio.Format)
	})

	t.Run("should add audio content for audio/mpeg file parts", func(t *testing.T) {
		t.Parallel()

		audioData := []byte{0, 1, 2, 3}
		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.FilePart{
						MediaType: "audio/mpeg",
						Data:      audioData,
					},
				},
			},
		}

		messages, warnings := DefaultToPrompt(prompt, "openai", "gpt-5")

		require.Empty(t, warnings)
		require.Len(t, messages, 1)

		userMsg := messages[0].OfUser
		content := userMsg.Content.OfArrayOfContentParts
		audioPart := content[0].OfInputAudio
		require.NotNil(t, audioPart)
		require.Equal(t, "mp3", audioPart.InputAudio.Format)
	})

	t.Run("should add audio content for audio/mp3 file parts", func(t *testing.T) {
		t.Parallel()

		audioData := []byte{0, 1, 2, 3}
		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.FilePart{
						MediaType: "audio/mp3",
						Data:      audioData,
					},
				},
			},
		}

		messages, warnings := DefaultToPrompt(prompt, "openai", "gpt-5")

		require.Empty(t, warnings)
		require.Len(t, messages, 1)

		userMsg := messages[0].OfUser
		content := userMsg.Content.OfArrayOfContentParts
		audioPart := content[0].OfInputAudio
		require.NotNil(t, audioPart)
		require.Equal(t, "mp3", audioPart.InputAudio.Format)
	})

	t.Run("should convert messages with PDF file parts", func(t *testing.T) {
		t.Parallel()

		pdfData := []byte{1, 2, 3, 4, 5}
		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.FilePart{
						MediaType: "application/pdf",
						Data:      pdfData,
						Filename:  "document.pdf",
					},
				},
			},
		}

		messages, warnings := DefaultToPrompt(prompt, "openai", "gpt-5")

		require.Empty(t, warnings)
		require.Len(t, messages, 1)

		userMsg := messages[0].OfUser
		content := userMsg.Content.OfArrayOfContentParts
		require.Len(t, content, 1)

		filePart := content[0].OfFile
		require.NotNil(t, filePart)
		require.Equal(t, "document.pdf", filePart.File.Filename.Value)

		expectedData := "data:application/pdf;base64," + base64.StdEncoding.EncodeToString(pdfData)
		require.Equal(t, expectedData, filePart.File.FileData.Value)
	})

	t.Run("should convert messages with binary PDF file parts", func(t *testing.T) {
		t.Parallel()

		pdfData := []byte{1, 2, 3, 4, 5}
		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.FilePart{
						MediaType: "application/pdf",
						Data:      pdfData,
						Filename:  "document.pdf",
					},
				},
			},
		}

		messages, warnings := DefaultToPrompt(prompt, "openai", "gpt-5")

		require.Empty(t, warnings)
		require.Len(t, messages, 1)

		userMsg := messages[0].OfUser
		content := userMsg.Content.OfArrayOfContentParts
		filePart := content[0].OfFile
		require.NotNil(t, filePart)

		expectedData := "data:application/pdf;base64," + base64.StdEncoding.EncodeToString(pdfData)
		require.Equal(t, expectedData, filePart.File.FileData.Value)
	})

	t.Run("should convert messages with PDF file parts using file_id", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.FilePart{
						MediaType: "application/pdf",
						Data:      []byte("file-pdf-12345"),
					},
				},
			},
		}

		messages, warnings := DefaultToPrompt(prompt, "openai", "gpt-5")

		require.Empty(t, warnings)
		require.Len(t, messages, 1)

		userMsg := messages[0].OfUser
		content := userMsg.Content.OfArrayOfContentParts
		filePart := content[0].OfFile
		require.NotNil(t, filePart)
		require.Equal(t, "file-pdf-12345", filePart.File.FileID.Value)
		require.True(t, param.IsOmitted(filePart.File.FileData))
		require.True(t, param.IsOmitted(filePart.File.Filename))
	})

	t.Run("should use default filename for PDF file parts when not provided", func(t *testing.T) {
		t.Parallel()

		pdfData := []byte{1, 2, 3, 4, 5}
		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.FilePart{
						MediaType: "application/pdf",
						Data:      pdfData,
					},
				},
			},
		}

		messages, warnings := DefaultToPrompt(prompt, "openai", "gpt-5")

		require.Empty(t, warnings)
		require.Len(t, messages, 1)

		userMsg := messages[0].OfUser
		content := userMsg.Content.OfArrayOfContentParts
		filePart := content[0].OfFile
		require.NotNil(t, filePart)
		require.Equal(t, "part-0.pdf", filePart.File.Filename.Value)
	})
}

func TestToOpenAiPrompt_ToolCalls(t *testing.T) {
	t.Parallel()

	t.Run("should stringify arguments to tool calls", func(t *testing.T) {
		t.Parallel()

		inputArgs := map[string]any{"foo": "bar123"}
		inputJSON, _ := json.Marshal(inputArgs)

		outputResult := map[string]any{"oof": "321rab"}
		outputJSON, _ := json.Marshal(outputResult)

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.ToolCallPart{
						ToolCallID: "quux",
						ToolName:   "thwomp",
						Input:      string(inputJSON),
					},
				},
			},
			{
				Role: fantasy.MessageRoleTool,
				Content: []fantasy.MessagePart{
					fantasy.ToolResultPart{
						ToolCallID: "quux",
						Output: fantasy.ToolResultOutputContentText{
							Text: string(outputJSON),
						},
					},
				},
			},
		}

		messages, warnings := DefaultToPrompt(prompt, "openai", "gpt-5")

		require.Empty(t, warnings)
		require.Len(t, messages, 2)

		// Check assistant message with tool call
		assistantMsg := messages[0].OfAssistant
		require.NotNil(t, assistantMsg)
		require.Equal(t, "", assistantMsg.Content.OfString.Value)
		require.Len(t, assistantMsg.ToolCalls, 1)

		toolCall := assistantMsg.ToolCalls[0].OfFunction
		require.NotNil(t, toolCall)
		require.Equal(t, "quux", toolCall.ID)
		require.Equal(t, "thwomp", toolCall.Function.Name)
		require.Equal(t, string(inputJSON), toolCall.Function.Arguments)

		// Check tool message
		toolMsg := messages[1].OfTool
		require.NotNil(t, toolMsg)
		require.Equal(t, string(outputJSON), toolMsg.Content.OfString.Value)
		require.Equal(t, "quux", toolMsg.ToolCallID)
	})

	t.Run("should handle different tool output types", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleTool,
				Content: []fantasy.MessagePart{
					fantasy.ToolResultPart{
						ToolCallID: "text-tool",
						Output: fantasy.ToolResultOutputContentText{
							Text: "Hello world",
						},
					},
					fantasy.ToolResultPart{
						ToolCallID: "error-tool",
						Output: fantasy.ToolResultOutputContentError{
							Error: errors.New("Something went wrong"),
						},
					},
				},
			},
		}

		messages, warnings := DefaultToPrompt(prompt, "openai", "gpt-5")

		require.Empty(t, warnings)
		require.Len(t, messages, 2)

		// Check first tool message (text)
		textToolMsg := messages[0].OfTool
		require.NotNil(t, textToolMsg)
		require.Equal(t, "Hello world", textToolMsg.Content.OfString.Value)
		require.Equal(t, "text-tool", textToolMsg.ToolCallID)

		// Check second tool message (error)
		errorToolMsg := messages[1].OfTool
		require.NotNil(t, errorToolMsg)
		require.Equal(t, "Something went wrong", errorToolMsg.Content.OfString.Value)
		require.Equal(t, "error-tool", errorToolMsg.ToolCallID)
	})
}

func TestToOpenAiPrompt_AssistantMessages(t *testing.T) {
	t.Parallel()

	t.Run("should handle simple text assistant messages", func(t *testing.T) {
		t.Parallel()

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Hello, how can I help you?"},
				},
			},
		}

		messages, warnings := DefaultToPrompt(prompt, "openai", "gpt-5")

		require.Empty(t, warnings)
		require.Len(t, messages, 1)

		assistantMsg := messages[0].OfAssistant
		require.NotNil(t, assistantMsg)
		require.Equal(t, "Hello, how can I help you?", assistantMsg.Content.OfString.Value)
	})

	t.Run("should handle assistant messages with mixed content", func(t *testing.T) {
		t.Parallel()

		inputArgs := map[string]any{"query": "test"}
		inputJSON, _ := json.Marshal(inputArgs)

		prompt := fantasy.Prompt{
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Let me search for that."},
					fantasy.ToolCallPart{
						ToolCallID: "call-123",
						ToolName:   "search",
						Input:      string(inputJSON),
					},
				},
			},
		}

		messages, warnings := DefaultToPrompt(prompt, "openai", "gpt-5")

		require.Empty(t, warnings)
		require.Len(t, messages, 1)

		assistantMsg := messages[0].OfAssistant
		require.NotNil(t, assistantMsg)
		require.Equal(t, "Let me search for that.", assistantMsg.Content.OfString.Value)
		require.Len(t, assistantMsg.ToolCalls, 1)

		toolCall := assistantMsg.ToolCalls[0].OfFunction
		require.Equal(t, "call-123", toolCall.ID)
		require.Equal(t, "search", toolCall.Function.Name)
		require.Equal(t, string(inputJSON), toolCall.Function.Arguments)
	})
}

var testPrompt = fantasy.Prompt{
	{
		Role: fantasy.MessageRoleUser,
		Content: []fantasy.MessagePart{
			fantasy.TextPart{Text: "Hello"},
		},
	},
}

var testLogprobs = map[string]any{
	"content": []map[string]any{
		{
			"token":   "Hello",
			"logprob": -0.0009994634,
			"top_logprobs": []map[string]any{
				{
					"token":   "Hello",
					"logprob": -0.0009994634,
				},
			},
		},
		{
			"token":   "!",
			"logprob": -0.13410144,
			"top_logprobs": []map[string]any{
				{
					"token":   "!",
					"logprob": -0.13410144,
				},
			},
		},
		{
			"token":   " How",
			"logprob": -0.0009250381,
			"top_logprobs": []map[string]any{
				{
					"token":   " How",
					"logprob": -0.0009250381,
				},
			},
		},
		{
			"token":   " can",
			"logprob": -0.047709424,
			"top_logprobs": []map[string]any{
				{
					"token":   " can",
					"logprob": -0.047709424,
				},
			},
		},
		{
			"token":   " I",
			"logprob": -0.000009014684,
			"top_logprobs": []map[string]any{
				{
					"token":   " I",
					"logprob": -0.000009014684,
				},
			},
		},
		{
			"token":   " assist",
			"logprob": -0.009125131,
			"top_logprobs": []map[string]any{
				{
					"token":   " assist",
					"logprob": -0.009125131,
				},
			},
		},
		{
			"token":   " you",
			"logprob": -0.0000066306106,
			"top_logprobs": []map[string]any{
				{
					"token":   " you",
					"logprob": -0.0000066306106,
				},
			},
		},
		{
			"token":   " today",
			"logprob": -0.00011093382,
			"top_logprobs": []map[string]any{
				{
					"token":   " today",
					"logprob": -0.00011093382,
				},
			},
		},
		{
			"token":   "?",
			"logprob": -0.00004596782,
			"top_logprobs": []map[string]any{
				{
					"token":   "?",
					"logprob": -0.00004596782,
				},
			},
		},
	},
}

type mockServer struct {
	server   *httptest.Server
	response map[string]any
	calls    []mockCall
}

type mockCall struct {
	method  string
	path    string
	headers map[string]string
	body    map[string]any
}

func newMockServer() *mockServer {
	ms := &mockServer{
		calls: make([]mockCall, 0),
	}

	ms.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Record the call
		call := mockCall{
			method:  r.Method,
			path:    r.URL.Path,
			headers: make(map[string]string),
		}

		for k, v := range r.Header {
			if len(v) > 0 {
				call.headers[k] = v[0]
			}
		}

		// Parse request body
		if r.Body != nil {
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body)
			call.body = body
		}

		ms.calls = append(ms.calls, call)

		// Return mock response
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ms.response)
	}))

	return ms
}

func (ms *mockServer) close() {
	ms.server.Close()
}

func (ms *mockServer) prepareJSONResponse(opts map[string]any) {
	// Default values
	response := map[string]any{
		"id":      "chatcmpl-95ZTZkhr0mHNKqerQfiwkuox3PHAd",
		"object":  "chat.completion",
		"created": 1711115037,
		"model":   "gpt-3.5-turbo-0125",
		"choices": []map[string]any{
			{
				"index": 0,
				"message": map[string]any{
					"role":    "assistant",
					"content": "",
				},
				"finish_reason": "stop",
			},
		},
		"usage": map[string]any{
			"prompt_tokens":     4,
			"total_tokens":      34,
			"completion_tokens": 30,
		},
		"system_fingerprint": "fp_3bc1b5746c",
	}

	// Override with provided options
	for k, v := range opts {
		switch k {
		case "content":
			response["choices"].([]map[string]any)[0]["message"].(map[string]any)["content"] = v
		case "tool_calls":
			response["choices"].([]map[string]any)[0]["message"].(map[string]any)["tool_calls"] = v
		case "function_call":
			response["choices"].([]map[string]any)[0]["message"].(map[string]any)["function_call"] = v
		case "annotations":
			response["choices"].([]map[string]any)[0]["message"].(map[string]any)["annotations"] = v
		case "usage":
			response["usage"] = v
		case "finish_reason":
			response["choices"].([]map[string]any)[0]["finish_reason"] = v
		case "id":
			response["id"] = v
		case "created":
			response["created"] = v
		case "model":
			response["model"] = v
		case "logprobs":
			if v != nil {
				response["choices"].([]map[string]any)[0]["logprobs"] = v
			}
		}
	}

	ms.response = response
}

func TestDoGenerate(t *testing.T) {
	t.Parallel()

	t.Run("should extract text response", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{
			"content": "Hello, World!",
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")

		result, err := model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
		})

		require.NoError(t, err)
		require.Len(t, result.Content, 1)

		textContent, ok := result.Content[0].(fantasy.TextContent)
		require.True(t, ok)
		require.Equal(t, "Hello, World!", textContent.Text)
	})

	t.Run("should extract usage", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{
			"usage": map[string]any{
				"prompt_tokens":     20,
				"total_tokens":      25,
				"completion_tokens": 5,
			},
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")

		result, err := model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
		})

		require.NoError(t, err)
		require.Equal(t, int64(20), result.Usage.InputTokens)
		require.Equal(t, int64(5), result.Usage.OutputTokens)
		require.Equal(t, int64(25), result.Usage.TotalTokens)
	})

	t.Run("should send request body", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")

		_, err = model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
		})

		require.NoError(t, err)
		require.Len(t, server.calls, 1)

		call := server.calls[0]
		require.Equal(t, "POST", call.method)
		require.Equal(t, "/chat/completions", call.path)
		require.Equal(t, "gpt-3.5-turbo", call.body["model"])

		messages, ok := call.body["messages"].([]any)
		require.True(t, ok)
		require.Len(t, messages, 1)

		message := messages[0].(map[string]any)
		require.Equal(t, "user", message["role"])
		require.Equal(t, "Hello", message["content"])
	})

	t.Run("should support partial usage", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{
			"usage": map[string]any{
				"prompt_tokens": 20,
				"total_tokens":  20,
			},
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")

		result, err := model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
		})

		require.NoError(t, err)
		require.Equal(t, int64(20), result.Usage.InputTokens)
		require.Equal(t, int64(0), result.Usage.OutputTokens)
		require.Equal(t, int64(20), result.Usage.TotalTokens)
	})

	t.Run("should extract logprobs", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{
			"logprobs": testLogprobs,
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")

		result, err := model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
			ProviderOptions: NewProviderOptions(&ProviderOptions{
				LogProbs: fantasy.Opt(true),
			}),
		})

		require.NoError(t, err)
		require.NotNil(t, result.ProviderMetadata)

		openaiMeta, ok := result.ProviderMetadata["openai"].(*ProviderMetadata)
		require.True(t, ok)

		logprobs := openaiMeta.Logprobs
		require.True(t, ok)
		require.NotNil(t, logprobs)
	})

	t.Run("should extract finish reason", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{
			"finish_reason": "stop",
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")

		result, err := model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
		})

		require.NoError(t, err)
		require.Equal(t, fantasy.FinishReasonStop, result.FinishReason)
	})

	t.Run("should support unknown finish reason", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{
			"finish_reason": "eos",
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")

		result, err := model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
		})

		require.NoError(t, err)
		require.Equal(t, fantasy.FinishReasonUnknown, result.FinishReason)
	})

	t.Run("should pass the model and the messages", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{
			"content": "",
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")

		_, err = model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
		})

		require.NoError(t, err)
		require.Len(t, server.calls, 1)

		call := server.calls[0]
		require.Equal(t, "gpt-3.5-turbo", call.body["model"])

		messages := call.body["messages"].([]any)
		require.Len(t, messages, 1)

		message := messages[0].(map[string]any)
		require.Equal(t, "user", message["role"])
		require.Equal(t, "Hello", message["content"])
	})

	t.Run("should pass settings", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")

		_, err = model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
			ProviderOptions: NewProviderOptions(&ProviderOptions{
				LogitBias: map[string]int64{
					"50256": -100,
				},
				ParallelToolCalls: fantasy.Opt(false),
				User:              fantasy.Opt("test-user-id"),
			}),
		})

		require.NoError(t, err)
		require.Len(t, server.calls, 1)

		call := server.calls[0]
		require.Equal(t, "gpt-3.5-turbo", call.body["model"])

		messages := call.body["messages"].([]any)
		require.Len(t, messages, 1)

		logitBias := call.body["logit_bias"].(map[string]any)
		require.Equal(t, float64(-100), logitBias["50256"])
		require.Equal(t, false, call.body["parallel_tool_calls"])
		require.Equal(t, "test-user-id", call.body["user"])
	})

	t.Run("should pass reasoningEffort setting", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{
			"content": "",
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "o1-mini")

		_, err = model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
			ProviderOptions: NewProviderOptions(
				&ProviderOptions{
					ReasoningEffort: ReasoningEffortOption(ReasoningEffortLow),
				},
			),
		})

		require.NoError(t, err)
		require.Len(t, server.calls, 1)

		call := server.calls[0]
		require.Equal(t, "o1-mini", call.body["model"])
		require.Equal(t, "low", call.body["reasoning_effort"])

		messages := call.body["messages"].([]any)
		require.Len(t, messages, 1)

		message := messages[0].(map[string]any)
		require.Equal(t, "user", message["role"])
		require.Equal(t, "Hello", message["content"])
	})

	t.Run("should pass textVerbosity setting", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{
			"content": "",
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-4o")

		_, err = model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
			ProviderOptions: NewProviderOptions(&ProviderOptions{
				TextVerbosity: fantasy.Opt("low"),
			}),
		})

		require.NoError(t, err)
		require.Len(t, server.calls, 1)

		call := server.calls[0]
		require.Equal(t, "gpt-4o", call.body["model"])
		require.Equal(t, "low", call.body["verbosity"])

		messages := call.body["messages"].([]any)
		require.Len(t, messages, 1)

		message := messages[0].(map[string]any)
		require.Equal(t, "user", message["role"])
		require.Equal(t, "Hello", message["content"])
	})

	t.Run("should pass tools and toolChoice", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{
			"content": "",
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")

		_, err = model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
			Tools: []fantasy.Tool{
				fantasy.FunctionTool{
					Name: "test-tool",
					InputSchema: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"value": map[string]any{
								"type": "string",
							},
						},
						"required":             []string{"value"},
						"additionalProperties": false,
						"$schema":              "http://json-schema.org/draft-07/schema#",
					},
				},
			},
			ToolChoice: &[]fantasy.ToolChoice{fantasy.ToolChoice("test-tool")}[0],
		})

		require.NoError(t, err)
		require.Len(t, server.calls, 1)

		call := server.calls[0]
		require.Equal(t, "gpt-3.5-turbo", call.body["model"])

		messages := call.body["messages"].([]any)
		require.Len(t, messages, 1)

		tools := call.body["tools"].([]any)
		require.Len(t, tools, 1)

		tool := tools[0].(map[string]any)
		require.Equal(t, "function", tool["type"])

		function := tool["function"].(map[string]any)
		require.Equal(t, "test-tool", function["name"])
		require.Equal(t, false, function["strict"])

		toolChoice := call.body["tool_choice"].(map[string]any)
		require.Equal(t, "function", toolChoice["type"])

		toolChoiceFunction := toolChoice["function"].(map[string]any)
		require.Equal(t, "test-tool", toolChoiceFunction["name"])
	})

	t.Run("should parse tool results", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{
			"tool_calls": []map[string]any{
				{
					"id":   "call_O17Uplv4lJvD6DVdIvFFeRMw",
					"type": "function",
					"function": map[string]any{
						"name":      "test-tool",
						"arguments": `{"value":"Spark"}`,
					},
				},
			},
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")

		result, err := model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
			Tools: []fantasy.Tool{
				fantasy.FunctionTool{
					Name: "test-tool",
					InputSchema: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"value": map[string]any{
								"type": "string",
							},
						},
						"required":             []string{"value"},
						"additionalProperties": false,
						"$schema":              "http://json-schema.org/draft-07/schema#",
					},
				},
			},
			ToolChoice: &[]fantasy.ToolChoice{fantasy.ToolChoice("test-tool")}[0],
		})

		require.NoError(t, err)
		require.Len(t, result.Content, 1)

		toolCall, ok := result.Content[0].(fantasy.ToolCallContent)
		require.True(t, ok)
		require.Equal(t, "call_O17Uplv4lJvD6DVdIvFFeRMw", toolCall.ToolCallID)
		require.Equal(t, "test-tool", toolCall.ToolName)
		require.Equal(t, `{"value":"Spark"}`, toolCall.Input)
	})

	t.Run("should handle ToolChoiceRequired", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{
			"content": "",
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")

		_, err = model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
			Tools: []fantasy.Tool{
				fantasy.FunctionTool{
					Name: "test-tool",
					InputSchema: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"value": map[string]any{
								"type": "string",
							},
						},
						"required":             []string{"value"},
						"additionalProperties": false,
						"$schema":              "http://json-schema.org/draft-07/schema#",
					},
				},
			},
			ToolChoice: &[]fantasy.ToolChoice{fantasy.ToolChoiceRequired}[0],
		})

		require.NoError(t, err)
		require.Len(t, server.calls, 1)

		call := server.calls[0]
		require.Equal(t, "gpt-3.5-turbo", call.body["model"])

		// Verify tool is present
		tools := call.body["tools"].([]any)
		require.Len(t, tools, 1)

		tool := tools[0].(map[string]any)
		require.Equal(t, "function", tool["type"])

		function := tool["function"].(map[string]any)
		require.Equal(t, "test-tool", function["name"])

		// Verify tool_choice is set to "required" (not a function name)
		toolChoice := call.body["tool_choice"]
		require.Equal(t, "required", toolChoice)
	})

	t.Run("should parse annotations/citations", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{
			"content": "Based on the search results [doc1], I found information.",
			"annotations": []map[string]any{
				{
					"type": "url_citation",
					"url_citation": map[string]any{
						"start_index": 24,
						"end_index":   29,
						"url":         "https://example.com/doc1.pdf",
						"title":       "Document 1",
					},
				},
			},
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")

		result, err := model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
		})

		require.NoError(t, err)
		require.Len(t, result.Content, 2)

		textContent, ok := result.Content[0].(fantasy.TextContent)
		require.True(t, ok)
		require.Equal(t, "Based on the search results [doc1], I found information.", textContent.Text)

		sourceContent, ok := result.Content[1].(fantasy.SourceContent)
		require.True(t, ok)
		require.Equal(t, fantasy.SourceTypeURL, sourceContent.SourceType)
		require.Equal(t, "https://example.com/doc1.pdf", sourceContent.URL)
		require.Equal(t, "Document 1", sourceContent.Title)
		require.NotEmpty(t, sourceContent.ID)
	})

	t.Run("should return cached_tokens in prompt_details_tokens", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{
			"usage": map[string]any{
				"prompt_tokens":     15,
				"completion_tokens": 20,
				"total_tokens":      35,
				"prompt_tokens_details": map[string]any{
					"cached_tokens": 1152,
				},
			},
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-4o-mini")

		result, err := model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
		})

		require.NoError(t, err)
		require.Equal(t, int64(1152), result.Usage.CacheReadTokens)
		// InputTokens = prompt_tokens - cached_tokens = 15 - 1152 = -1137 → clamped to 0
		require.Equal(t, int64(0), result.Usage.InputTokens)
		require.Equal(t, int64(20), result.Usage.OutputTokens)
		require.Equal(t, int64(35), result.Usage.TotalTokens)
	})

	t.Run("should return accepted_prediction_tokens and rejected_prediction_tokens", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{
			"usage": map[string]any{
				"prompt_tokens":     15,
				"completion_tokens": 20,
				"total_tokens":      35,
				"completion_tokens_details": map[string]any{
					"accepted_prediction_tokens": 123,
					"rejected_prediction_tokens": 456,
				},
			},
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-4o-mini")

		result, err := model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
		})

		require.NoError(t, err)
		require.NotNil(t, result.ProviderMetadata)

		openaiMeta, ok := result.ProviderMetadata["openai"].(*ProviderMetadata)

		require.True(t, ok)
		require.Equal(t, int64(123), openaiMeta.AcceptedPredictionTokens)
		require.Equal(t, int64(456), openaiMeta.RejectedPredictionTokens)
	})

	t.Run("should clear out temperature, top_p, frequency_penalty, presence_penalty for reasoning models", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "o1-preview")

		result, err := model.Generate(context.Background(), fantasy.Call{
			Prompt:           testPrompt,
			Temperature:      &[]float64{0.5}[0],
			TopP:             &[]float64{0.7}[0],
			FrequencyPenalty: &[]float64{0.2}[0],
			PresencePenalty:  &[]float64{0.3}[0],
		})

		require.NoError(t, err)
		require.Len(t, server.calls, 1)

		call := server.calls[0]
		require.Equal(t, "o1-preview", call.body["model"])

		messages := call.body["messages"].([]any)
		require.Len(t, messages, 1)

		message := messages[0].(map[string]any)
		require.Equal(t, "user", message["role"])
		require.Equal(t, "Hello", message["content"])

		// These should not be present
		require.Nil(t, call.body["temperature"])
		require.Nil(t, call.body["top_p"])
		require.Nil(t, call.body["frequency_penalty"])
		require.Nil(t, call.body["presence_penalty"])

		// Should have warnings
		require.Len(t, result.Warnings, 4)
		require.Equal(t, fantasy.CallWarningTypeUnsupportedSetting, result.Warnings[0].Type)
		require.Equal(t, "temperature", result.Warnings[0].Setting)
		require.Contains(t, result.Warnings[0].Details, "temperature is not supported for reasoning models")
	})

	t.Run("should convert maxOutputTokens to max_completion_tokens for reasoning models", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "o1-preview")

		_, err = model.Generate(context.Background(), fantasy.Call{
			Prompt:          testPrompt,
			MaxOutputTokens: &[]int64{1000}[0],
		})

		require.NoError(t, err)
		require.Len(t, server.calls, 1)

		call := server.calls[0]
		require.Equal(t, "o1-preview", call.body["model"])
		require.Equal(t, float64(1000), call.body["max_completion_tokens"])
		require.Nil(t, call.body["max_tokens"])

		messages := call.body["messages"].([]any)
		require.Len(t, messages, 1)

		message := messages[0].(map[string]any)
		require.Equal(t, "user", message["role"])
		require.Equal(t, "Hello", message["content"])
	})

	t.Run("should return reasoning tokens", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{
			"usage": map[string]any{
				"prompt_tokens":     15,
				"completion_tokens": 20,
				"total_tokens":      35,
				"completion_tokens_details": map[string]any{
					"reasoning_tokens": 10,
				},
			},
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "o1-preview")

		result, err := model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
		})

		require.NoError(t, err)
		require.Equal(t, int64(15), result.Usage.InputTokens)
		require.Equal(t, int64(20), result.Usage.OutputTokens)
		require.Equal(t, int64(35), result.Usage.TotalTokens)
		require.Equal(t, int64(10), result.Usage.ReasoningTokens)
	})

	t.Run("should send max_completion_tokens extension setting", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{
			"model": "o1-preview",
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "o1-preview")

		_, err = model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
			ProviderOptions: NewProviderOptions(&ProviderOptions{
				MaxCompletionTokens: fantasy.Opt(int64(255)),
			}),
		})

		require.NoError(t, err)
		require.Len(t, server.calls, 1)

		call := server.calls[0]
		require.Equal(t, "o1-preview", call.body["model"])
		require.Equal(t, float64(255), call.body["max_completion_tokens"])

		messages := call.body["messages"].([]any)
		require.Len(t, messages, 1)

		message := messages[0].(map[string]any)
		require.Equal(t, "user", message["role"])
		require.Equal(t, "Hello", message["content"])
	})

	t.Run("should send prediction extension setting", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{
			"content": "",
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")

		_, err = model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
			ProviderOptions: NewProviderOptions(&ProviderOptions{
				Prediction: map[string]any{
					"type":    "content",
					"content": "Hello, World!",
				},
			}),
		})

		require.NoError(t, err)
		require.Len(t, server.calls, 1)

		call := server.calls[0]
		require.Equal(t, "gpt-3.5-turbo", call.body["model"])

		prediction := call.body["prediction"].(map[string]any)
		require.Equal(t, "content", prediction["type"])
		require.Equal(t, "Hello, World!", prediction["content"])

		messages := call.body["messages"].([]any)
		require.Len(t, messages, 1)

		message := messages[0].(map[string]any)
		require.Equal(t, "user", message["role"])
		require.Equal(t, "Hello", message["content"])
	})

	t.Run("should send store extension setting", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{
			"content": "",
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")

		_, err = model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
			ProviderOptions: NewProviderOptions(&ProviderOptions{
				Store: fantasy.Opt(true),
			}),
		})

		require.NoError(t, err)
		require.Len(t, server.calls, 1)

		call := server.calls[0]
		require.Equal(t, "gpt-3.5-turbo", call.body["model"])
		require.Equal(t, true, call.body["store"])

		messages := call.body["messages"].([]any)
		require.Len(t, messages, 1)

		message := messages[0].(map[string]any)
		require.Equal(t, "user", message["role"])
		require.Equal(t, "Hello", message["content"])
	})

	t.Run("should send metadata extension values", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{
			"content": "",
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")

		_, err = model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
			ProviderOptions: NewProviderOptions(&ProviderOptions{
				Metadata: map[string]any{
					"custom": "value",
				},
			}),
		})

		require.NoError(t, err)
		require.Len(t, server.calls, 1)

		call := server.calls[0]
		require.Equal(t, "gpt-3.5-turbo", call.body["model"])

		metadata := call.body["metadata"].(map[string]any)
		require.Equal(t, "value", metadata["custom"])

		messages := call.body["messages"].([]any)
		require.Len(t, messages, 1)

		message := messages[0].(map[string]any)
		require.Equal(t, "user", message["role"])
		require.Equal(t, "Hello", message["content"])
	})

	t.Run("should send promptCacheKey extension value", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{
			"content": "",
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")

		_, err = model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
			ProviderOptions: NewProviderOptions(&ProviderOptions{
				PromptCacheKey: fantasy.Opt("test-cache-key-123"),
			}),
		})

		require.NoError(t, err)
		require.Len(t, server.calls, 1)

		call := server.calls[0]
		require.Equal(t, "gpt-3.5-turbo", call.body["model"])
		require.Equal(t, "test-cache-key-123", call.body["prompt_cache_key"])

		messages := call.body["messages"].([]any)
		require.Len(t, messages, 1)

		message := messages[0].(map[string]any)
		require.Equal(t, "user", message["role"])
		require.Equal(t, "Hello", message["content"])
	})

	t.Run("should send safety_identifier extension value", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{
			"content": "",
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")

		_, err = model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
			ProviderOptions: NewProviderOptions(&ProviderOptions{
				SafetyIdentifier: fantasy.Opt("test-safety-identifier-123"),
			}),
		})

		require.NoError(t, err)
		require.Len(t, server.calls, 1)

		call := server.calls[0]
		require.Equal(t, "gpt-3.5-turbo", call.body["model"])
		require.Equal(t, "test-safety-identifier-123", call.body["safety_identifier"])

		messages := call.body["messages"].([]any)
		require.Len(t, messages, 1)

		message := messages[0].(map[string]any)
		require.Equal(t, "user", message["role"])
		require.Equal(t, "Hello", message["content"])
	})

	t.Run("should remove temperature setting for search preview models", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-4o-search-preview")

		result, err := model.Generate(context.Background(), fantasy.Call{
			Prompt:      testPrompt,
			Temperature: &[]float64{0.7}[0],
		})

		require.NoError(t, err)
		require.Len(t, server.calls, 1)

		call := server.calls[0]
		require.Equal(t, "gpt-4o-search-preview", call.body["model"])
		require.Nil(t, call.body["temperature"])

		require.Len(t, result.Warnings, 1)
		require.Equal(t, fantasy.CallWarningTypeUnsupportedSetting, result.Warnings[0].Type)
		require.Equal(t, "temperature", result.Warnings[0].Setting)
		require.Contains(t, result.Warnings[0].Details, "search preview models")
	})

	t.Run("should send ServiceTier flex processing setting", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{
			"content": "",
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "o3-mini")

		_, err = model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
			ProviderOptions: NewProviderOptions(&ProviderOptions{
				ServiceTier: fantasy.Opt("flex"),
			}),
		})

		require.NoError(t, err)
		require.Len(t, server.calls, 1)

		call := server.calls[0]
		require.Equal(t, "o3-mini", call.body["model"])
		require.Equal(t, "flex", call.body["service_tier"])

		messages := call.body["messages"].([]any)
		require.Len(t, messages, 1)

		message := messages[0].(map[string]any)
		require.Equal(t, "user", message["role"])
		require.Equal(t, "Hello", message["content"])
	})

	t.Run("should show warning when using flex processing with unsupported model", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-4o-mini")

		result, err := model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
			ProviderOptions: NewProviderOptions(&ProviderOptions{
				ServiceTier: fantasy.Opt("flex"),
			}),
		})

		require.NoError(t, err)
		require.Len(t, server.calls, 1)

		call := server.calls[0]
		require.Nil(t, call.body["service_tier"])

		require.Len(t, result.Warnings, 1)
		require.Equal(t, fantasy.CallWarningTypeUnsupportedSetting, result.Warnings[0].Type)
		require.Equal(t, "ServiceTier", result.Warnings[0].Setting)
		require.Contains(t, result.Warnings[0].Details, "flex processing is only available")
	})

	t.Run("should send serviceTier priority processing setting", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-4o-mini")

		_, err = model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
			ProviderOptions: NewProviderOptions(&ProviderOptions{
				ServiceTier: fantasy.Opt("priority"),
			}),
		})

		require.NoError(t, err)
		require.Len(t, server.calls, 1)

		call := server.calls[0]
		require.Equal(t, "gpt-4o-mini", call.body["model"])
		require.Equal(t, "priority", call.body["service_tier"])

		messages := call.body["messages"].([]any)
		require.Len(t, messages, 1)

		message := messages[0].(map[string]any)
		require.Equal(t, "user", message["role"])
		require.Equal(t, "Hello", message["content"])
	})

	t.Run("should show warning when using priority processing with unsupported model", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()

		server.prepareJSONResponse(map[string]any{})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")

		result, err := model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
			ProviderOptions: NewProviderOptions(&ProviderOptions{
				ServiceTier: fantasy.Opt("priority"),
			}),
		})

		require.NoError(t, err)
		require.Len(t, server.calls, 1)

		call := server.calls[0]
		require.Nil(t, call.body["service_tier"])

		require.Len(t, result.Warnings, 1)
		require.Equal(t, fantasy.CallWarningTypeUnsupportedSetting, result.Warnings[0].Type)
		require.Equal(t, "ServiceTier", result.Warnings[0].Setting)
		require.Contains(t, result.Warnings[0].Details, "priority processing is only available")
	})
}

type streamingMockServer struct {
	server *httptest.Server
	chunks []string
	calls  []mockCall
}

func newStreamingMockServer() *streamingMockServer {
	sms := &streamingMockServer{
		calls: make([]mockCall, 0),
	}

	sms.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Record the call
		call := mockCall{
			method:  r.Method,
			path:    r.URL.Path,
			headers: make(map[string]string),
		}

		for k, v := range r.Header {
			if len(v) > 0 {
				call.headers[k] = v[0]
			}
		}

		// Parse request body
		if r.Body != nil {
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body)
			call.body = body
		}

		sms.calls = append(sms.calls, call)

		// Set streaming headers
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		// Add custom headers if any
		for _, chunk := range sms.chunks {
			if strings.HasPrefix(chunk, "HEADER:") {
				parts := strings.SplitN(chunk[7:], ":", 2)
				if len(parts) == 2 {
					w.Header().Set(parts[0], parts[1])
				}
				continue
			}
		}

		w.WriteHeader(http.StatusOK)

		// Write chunks
		for _, chunk := range sms.chunks {
			if strings.HasPrefix(chunk, "HEADER:") {
				continue
			}
			w.Write([]byte(chunk))
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
	}))

	return sms
}

func (sms *streamingMockServer) close() {
	sms.server.Close()
}

func (sms *streamingMockServer) prepareStreamResponse(opts map[string]any) {
	content := []string{}
	if c, ok := opts["content"].([]string); ok {
		content = c
	}

	usage := map[string]any{
		"prompt_tokens":     17,
		"total_tokens":      244,
		"completion_tokens": 227,
	}
	if u, ok := opts["usage"].(map[string]any); ok {
		usage = u
	}

	logprobs := map[string]any{}
	if l, ok := opts["logprobs"].(map[string]any); ok {
		logprobs = l
	}

	finishReason := "stop"
	if fr, ok := opts["finish_reason"].(string); ok {
		finishReason = fr
	}

	model := "gpt-3.5-turbo-0613"
	if m, ok := opts["model"].(string); ok {
		model = m
	}

	headers := map[string]string{}
	if h, ok := opts["headers"].(map[string]string); ok {
		headers = h
	}

	chunks := []string{}

	// Add custom headers
	for k, v := range headers {
		chunks = append(chunks, "HEADER:"+k+":"+v)
	}

	// Initial chunk with role
	initialChunk := map[string]any{
		"id":                 "chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP",
		"object":             "chat.completion.chunk",
		"created":            1702657020,
		"model":              model,
		"system_fingerprint": nil,
		"choices": []map[string]any{
			{
				"index": 0,
				"delta": map[string]any{
					"role":    "assistant",
					"content": "",
				},
				"finish_reason": nil,
			},
		},
	}
	initialData, _ := json.Marshal(initialChunk)
	chunks = append(chunks, "data: "+string(initialData)+"\n\n")

	// Content chunks
	for i, text := range content {
		contentChunk := map[string]any{
			"id":                 "chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP",
			"object":             "chat.completion.chunk",
			"created":            1702657020,
			"model":              model,
			"system_fingerprint": nil,
			"choices": []map[string]any{
				{
					"index": 1,
					"delta": map[string]any{
						"content": text,
					},
					"finish_reason": nil,
				},
			},
		}
		contentData, _ := json.Marshal(contentChunk)
		chunks = append(chunks, "data: "+string(contentData)+"\n\n")

		// Add annotations if this is the last content chunk and we have annotations
		if i == len(content)-1 {
			if annotations, ok := opts["annotations"].([]map[string]any); ok {
				annotationChunk := map[string]any{
					"id":                 "chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP",
					"object":             "chat.completion.chunk",
					"created":            1702657020,
					"model":              model,
					"system_fingerprint": nil,
					"choices": []map[string]any{
						{
							"index": 1,
							"delta": map[string]any{
								"annotations": annotations,
							},
							"finish_reason": nil,
						},
					},
				}
				annotationData, _ := json.Marshal(annotationChunk)
				chunks = append(chunks, "data: "+string(annotationData)+"\n\n")
			}
		}
	}

	// Finish chunk
	finishChunk := map[string]any{
		"id":                 "chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP",
		"object":             "chat.completion.chunk",
		"created":            1702657020,
		"model":              model,
		"system_fingerprint": nil,
		"choices": []map[string]any{
			{
				"index":         0,
				"delta":         map[string]any{},
				"finish_reason": finishReason,
			},
		},
	}

	if len(logprobs) > 0 {
		finishChunk["choices"].([]map[string]any)[0]["logprobs"] = logprobs
	}

	finishData, _ := json.Marshal(finishChunk)
	chunks = append(chunks, "data: "+string(finishData)+"\n\n")

	// Usage chunk
	usageChunk := map[string]any{
		"id":                 "chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP",
		"object":             "chat.completion.chunk",
		"created":            1702657020,
		"model":              model,
		"system_fingerprint": "fp_3bc1b5746c",
		"choices":            []map[string]any{},
		"usage":              usage,
	}
	usageData, _ := json.Marshal(usageChunk)
	chunks = append(chunks, "data: "+string(usageData)+"\n\n")

	// Done
	chunks = append(chunks, "data: [DONE]\n\n")

	sms.chunks = chunks
}

func (sms *streamingMockServer) prepareToolStreamResponse() {
	chunks := []string{
		`data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1711357598,"model":"gpt-3.5-turbo-0125","system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_O17Uplv4lJvD6DVdIvFFeRMw","type":"function","function":{"name":"test-tool","arguments":""}}]},"logprobs":null,"finish_reason":null}]}` + "\n\n",
		`data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1711357598,"model":"gpt-3.5-turbo-0125","system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\""}}]},"logprobs":null,"finish_reason":null}]}` + "\n\n",
		`data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1711357598,"model":"gpt-3.5-turbo-0125","system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"value"}}]},"logprobs":null,"finish_reason":null}]}` + "\n\n",
		`data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1711357598,"model":"gpt-3.5-turbo-0125","system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\":\""}}]},"logprobs":null,"finish_reason":null}]}` + "\n\n",
		`data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1711357598,"model":"gpt-3.5-turbo-0125","system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Spark"}}]},"logprobs":null,"finish_reason":null}]}` + "\n\n",
		`data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1711357598,"model":"gpt-3.5-turbo-0125","system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"le"}}]},"logprobs":null,"finish_reason":null}]}` + "\n\n",
		`data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1711357598,"model":"gpt-3.5-turbo-0125","system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":" Day"}}]},"logprobs":null,"finish_reason":null}]}` + "\n\n",
		`data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1711357598,"model":"gpt-3.5-turbo-0125","system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"}"}}]},"logprobs":null,"finish_reason":null}]}` + "\n\n",
		`data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1711357598,"model":"gpt-3.5-turbo-0125","system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"tool_calls"}]}` + "\n\n",
		`data: {"id":"chatcmpl-96aZqmeDpA9IPD6tACY8djkMsJCMP","object":"chat.completion.chunk","created":1711357598,"model":"gpt-3.5-turbo-0125","system_fingerprint":"fp_3bc1b5746c","choices":[],"usage":{"prompt_tokens":53,"completion_tokens":17,"total_tokens":70}}` + "\n\n",
		"data: [DONE]\n\n",
	}
	sms.chunks = chunks
}

func (sms *streamingMockServer) prepareErrorStreamResponse() {
	chunks := []string{
		`data: {"error":{"message": "The server had an error processing your request. Sorry about that! You can retry your request, or contact us through our help center at help.openai.com if you keep seeing this error.","type":"server_error","param":null,"code":null}}` + "\n\n",
		"data: [DONE]\n\n",
	}
	sms.chunks = chunks
}

func (sms *streamingMockServer) prepareToolStreamResponseWithEmptyArgs() {
	chunks := []string{
		// Tool call start with empty arguments (like Copilot sometimes does)
		`data: {"id":"chatcmpl-emptyargs","object":"chat.completion.chunk","created":1711357598,"model":"gpt-3.5-turbo-0125","system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_empty_args","type":"function","function":{"name":"test-tool","arguments":""}}]},"logprobs":null,"finish_reason":null}]}` + "\n\n",
		// Finish without any argument deltas
		`data: {"id":"chatcmpl-emptyargs","object":"chat.completion.chunk","created":1711357598,"model":"gpt-3.5-turbo-0125","system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"tool_calls"}]}` + "\n\n",
		`data: {"id":"chatcmpl-emptyargs","object":"chat.completion.chunk","created":1711357598,"model":"gpt-3.5-turbo-0125","system_fingerprint":"fp_3bc1b5746c","choices":[],"usage":{"prompt_tokens":53,"completion_tokens":17,"total_tokens":70}}` + "\n\n",
		"data: [DONE]\n\n",
	}
	sms.chunks = chunks
}

func collectStreamParts(stream fantasy.StreamResponse) ([]fantasy.StreamPart, error) {
	var parts []fantasy.StreamPart
	for part := range stream {
		parts = append(parts, part)
		if part.Type == fantasy.StreamPartTypeError {
			break
		}
		if part.Type == fantasy.StreamPartTypeFinish {
			break
		}
	}
	return parts, nil
}

func TestDoStream(t *testing.T) {
	t.Parallel()

	t.Run("should stream text deltas", func(t *testing.T) {
		t.Parallel()

		server := newStreamingMockServer()
		defer server.close()

		server.prepareStreamResponse(map[string]any{
			"content":       []string{"Hello", ", ", "World!"},
			"finish_reason": "stop",
			"usage": map[string]any{
				"prompt_tokens":     17,
				"total_tokens":      244,
				"completion_tokens": 227,
			},
			"logprobs": testLogprobs,
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")

		stream, err := model.Stream(context.Background(), fantasy.Call{
			Prompt: testPrompt,
		})

		require.NoError(t, err)

		parts, err := collectStreamParts(stream)
		require.NoError(t, err)

		// Verify stream structure
		require.True(t, len(parts) >= 4) // text-start, deltas, text-end, finish

		// Find text parts
		textStart, textEnd, finish := -1, -1, -1
		var deltas []string

		for i, part := range parts {
			switch part.Type {
			case fantasy.StreamPartTypeTextStart:
				textStart = i
			case fantasy.StreamPartTypeTextDelta:
				deltas = append(deltas, part.Delta)
			case fantasy.StreamPartTypeTextEnd:
				textEnd = i
			case fantasy.StreamPartTypeFinish:
				finish = i
			}
		}

		require.NotEqual(t, -1, textStart)
		require.NotEqual(t, -1, textEnd)
		require.NotEqual(t, -1, finish)
		require.Equal(t, []string{"Hello", ", ", "World!"}, deltas)

		// Check finish part
		finishPart := parts[finish]
		require.Equal(t, fantasy.FinishReasonStop, finishPart.FinishReason)
		require.Equal(t, int64(17), finishPart.Usage.InputTokens)
		require.Equal(t, int64(227), finishPart.Usage.OutputTokens)
		require.Equal(t, int64(244), finishPart.Usage.TotalTokens)
	})

	t.Run("should stream tool deltas", func(t *testing.T) {
		t.Parallel()

		server := newStreamingMockServer()
		defer server.close()

		server.prepareToolStreamResponse()

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")

		stream, err := model.Stream(context.Background(), fantasy.Call{
			Prompt: testPrompt,
			Tools: []fantasy.Tool{
				fantasy.FunctionTool{
					Name: "test-tool",
					InputSchema: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"value": map[string]any{
								"type": "string",
							},
						},
						"required":             []string{"value"},
						"additionalProperties": false,
						"$schema":              "http://json-schema.org/draft-07/schema#",
					},
				},
			},
		})

		require.NoError(t, err)

		parts, err := collectStreamParts(stream)
		require.NoError(t, err)

		// Find tool-related parts
		toolInputStart, toolInputEnd, toolCall := -1, -1, -1
		var toolDeltas []string

		for i, part := range parts {
			switch part.Type {
			case fantasy.StreamPartTypeToolInputStart:
				toolInputStart = i
				require.Equal(t, "call_O17Uplv4lJvD6DVdIvFFeRMw", part.ID)
				require.Equal(t, "test-tool", part.ToolCallName)
			case fantasy.StreamPartTypeToolInputDelta:
				toolDeltas = append(toolDeltas, part.Delta)
			case fantasy.StreamPartTypeToolInputEnd:
				toolInputEnd = i
			case fantasy.StreamPartTypeToolCall:
				toolCall = i
				require.Equal(t, "call_O17Uplv4lJvD6DVdIvFFeRMw", part.ID)
				require.Equal(t, "test-tool", part.ToolCallName)
				require.Equal(t, `{"value":"Sparkle Day"}`, part.ToolCallInput)
			}
		}

		require.NotEqual(t, -1, toolInputStart)
		require.NotEqual(t, -1, toolInputEnd)
		require.NotEqual(t, -1, toolCall)

		// Verify tool deltas combine to form the complete input
		var fullInput strings.Builder
		for _, delta := range toolDeltas {
			fullInput.WriteString(delta)
		}
		require.Equal(t, `{"value":"Sparkle Day"}`, fullInput.String())
	})

	t.Run("should handle tool calls with empty arguments", func(t *testing.T) {
		t.Parallel()

		server := newStreamingMockServer()
		defer server.close()

		server.prepareToolStreamResponseWithEmptyArgs()

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")

		stream, err := model.Stream(context.Background(), fantasy.Call{
			Prompt: testPrompt,
			Tools: []fantasy.Tool{
				fantasy.FunctionTool{
					Name: "test-tool",
					InputSchema: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"value": map[string]any{
								"type": "string",
							},
						},
						"required":             []string{"value"},
						"additionalProperties": false,
						"$schema":              "http://json-schema.org/draft-07/schema#",
					},
				},
			},
		})

		require.NoError(t, err)

		parts, err := collectStreamParts(stream)
		require.NoError(t, err)

		// Find tool-related parts
		toolInputStart, toolInputEnd, toolCall := -1, -1, -1

		for i, part := range parts {
			switch part.Type {
			case fantasy.StreamPartTypeToolInputStart:
				toolInputStart = i
				require.Equal(t, "call_empty_args", part.ID)
				require.Equal(t, "test-tool", part.ToolCallName)
			case fantasy.StreamPartTypeToolInputEnd:
				toolInputEnd = i
				require.Equal(t, "call_empty_args", part.ID)
			case fantasy.StreamPartTypeToolCall:
				toolCall = i
				require.Equal(t, "call_empty_args", part.ID)
				require.Equal(t, "test-tool", part.ToolCallName)
				// Empty arguments should be normalized to "{}"
				require.Equal(t, "{}", part.ToolCallInput)
			}
		}

		require.NotEqual(t, -1, toolInputStart, "expected ToolInputStart part")
		require.NotEqual(t, -1, toolInputEnd, "expected ToolInputEnd part")
		require.NotEqual(t, -1, toolCall, "expected ToolCall part")
	})

	t.Run("should stream annotations/citations", func(t *testing.T) {
		t.Parallel()

		server := newStreamingMockServer()
		defer server.close()

		server.prepareStreamResponse(map[string]any{
			"content": []string{"Based on search results"},
			"annotations": []map[string]any{
				{
					"type": "url_citation",
					"url_citation": map[string]any{
						"start_index": 24,
						"end_index":   29,
						"url":         "https://example.com/doc1.pdf",
						"title":       "Document 1",
					},
				},
			},
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")

		stream, err := model.Stream(context.Background(), fantasy.Call{
			Prompt: testPrompt,
		})

		require.NoError(t, err)

		parts, err := collectStreamParts(stream)
		require.NoError(t, err)

		// Find source part
		var sourcePart *fantasy.StreamPart
		for _, part := range parts {
			if part.Type == fantasy.StreamPartTypeSource {
				sourcePart = &part
				break
			}
		}

		require.NotNil(t, sourcePart)
		require.Equal(t, fantasy.SourceTypeURL, sourcePart.SourceType)
		require.Equal(t, "https://example.com/doc1.pdf", sourcePart.URL)
		require.Equal(t, "Document 1", sourcePart.Title)
		require.NotEmpty(t, sourcePart.ID)
	})

	t.Run("should handle error stream parts", func(t *testing.T) {
		t.Parallel()

		server := newStreamingMockServer()
		defer server.close()

		server.prepareErrorStreamResponse()

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")

		stream, err := model.Stream(context.Background(), fantasy.Call{
			Prompt: testPrompt,
		})

		require.NoError(t, err)

		parts, err := collectStreamParts(stream)
		require.NoError(t, err)

		// Should have error and finish parts
		require.True(t, len(parts) >= 1)

		// Find error part
		var errorPart *fantasy.StreamPart
		for _, part := range parts {
			if part.Type == fantasy.StreamPartTypeError {
				errorPart = &part
				break
			}
		}

		require.NotNil(t, errorPart)
		require.NotNil(t, errorPart.Error)
	})

	t.Run("should send request body", func(t *testing.T) {
		t.Parallel()

		server := newStreamingMockServer()
		defer server.close()

		server.prepareStreamResponse(map[string]any{
			"content": []string{},
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")

		_, err = model.Stream(context.Background(), fantasy.Call{
			Prompt: testPrompt,
		})

		require.NoError(t, err)
		require.Len(t, server.calls, 1)

		call := server.calls[0]
		require.Equal(t, "POST", call.method)
		require.Equal(t, "/chat/completions", call.path)
		require.Equal(t, "gpt-3.5-turbo", call.body["model"])
		require.Equal(t, true, call.body["stream"])

		streamOptions := call.body["stream_options"].(map[string]any)
		require.Equal(t, true, streamOptions["include_usage"])

		messages := call.body["messages"].([]any)
		require.Len(t, messages, 1)

		message := messages[0].(map[string]any)
		require.Equal(t, "user", message["role"])
		require.Equal(t, "Hello", message["content"])
	})

	t.Run("should return cached tokens in providerMetadata", func(t *testing.T) {
		t.Parallel()

		server := newStreamingMockServer()
		defer server.close()

		server.prepareStreamResponse(map[string]any{
			"content": []string{},
			"usage": map[string]any{
				"prompt_tokens":     15,
				"completion_tokens": 20,
				"total_tokens":      35,
				"prompt_tokens_details": map[string]any{
					"cached_tokens": 1152,
				},
			},
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")

		stream, err := model.Stream(context.Background(), fantasy.Call{
			Prompt: testPrompt,
		})

		require.NoError(t, err)

		parts, err := collectStreamParts(stream)
		require.NoError(t, err)

		// Find finish part
		var finishPart *fantasy.StreamPart
		for _, part := range parts {
			if part.Type == fantasy.StreamPartTypeFinish {
				finishPart = &part
				break
			}
		}

		require.NotNil(t, finishPart)
		require.Equal(t, int64(1152), finishPart.Usage.CacheReadTokens)
		// InputTokens = prompt_tokens - cached_tokens = 15 - 1152 = -1137 → clamped to 0
		require.Equal(t, int64(0), finishPart.Usage.InputTokens)
		require.Equal(t, int64(20), finishPart.Usage.OutputTokens)
		require.Equal(t, int64(35), finishPart.Usage.TotalTokens)
	})

	t.Run("should return accepted_prediction_tokens and rejected_prediction_tokens", func(t *testing.T) {
		t.Parallel()

		server := newStreamingMockServer()
		defer server.close()

		server.prepareStreamResponse(map[string]any{
			"content": []string{},
			"usage": map[string]any{
				"prompt_tokens":     15,
				"completion_tokens": 20,
				"total_tokens":      35,
				"completion_tokens_details": map[string]any{
					"accepted_prediction_tokens": 123,
					"rejected_prediction_tokens": 456,
				},
			},
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")

		stream, err := model.Stream(context.Background(), fantasy.Call{
			Prompt: testPrompt,
		})

		require.NoError(t, err)

		parts, err := collectStreamParts(stream)
		require.NoError(t, err)

		// Find finish part
		var finishPart *fantasy.StreamPart
		for _, part := range parts {
			if part.Type == fantasy.StreamPartTypeFinish {
				finishPart = &part
				break
			}
		}

		require.NotNil(t, finishPart)
		require.NotNil(t, finishPart.ProviderMetadata)

		openaiMeta, ok := finishPart.ProviderMetadata["openai"].(*ProviderMetadata)
		require.True(t, ok)
		require.Equal(t, int64(123), openaiMeta.AcceptedPredictionTokens)
		require.Equal(t, int64(456), openaiMeta.RejectedPredictionTokens)
	})

	t.Run("should send store extension setting", func(t *testing.T) {
		t.Parallel()

		server := newStreamingMockServer()
		defer server.close()

		server.prepareStreamResponse(map[string]any{
			"content": []string{},
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")

		_, err = model.Stream(context.Background(), fantasy.Call{
			Prompt: testPrompt,
			ProviderOptions: NewProviderOptions(&ProviderOptions{
				Store: fantasy.Opt(true),
			}),
		})

		require.NoError(t, err)
		require.Len(t, server.calls, 1)

		call := server.calls[0]
		require.Equal(t, "gpt-3.5-turbo", call.body["model"])
		require.Equal(t, true, call.body["stream"])
		require.Equal(t, true, call.body["store"])

		streamOptions := call.body["stream_options"].(map[string]any)
		require.Equal(t, true, streamOptions["include_usage"])

		messages := call.body["messages"].([]any)
		require.Len(t, messages, 1)

		message := messages[0].(map[string]any)
		require.Equal(t, "user", message["role"])
		require.Equal(t, "Hello", message["content"])
	})

	t.Run("should send metadata extension values", func(t *testing.T) {
		t.Parallel()

		server := newStreamingMockServer()
		defer server.close()

		server.prepareStreamResponse(map[string]any{
			"content": []string{},
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")

		_, err = model.Stream(context.Background(), fantasy.Call{
			Prompt: testPrompt,
			ProviderOptions: NewProviderOptions(&ProviderOptions{
				Metadata: map[string]any{
					"custom": "value",
				},
			}),
		})

		require.NoError(t, err)
		require.Len(t, server.calls, 1)

		call := server.calls[0]
		require.Equal(t, "gpt-3.5-turbo", call.body["model"])
		require.Equal(t, true, call.body["stream"])

		metadata := call.body["metadata"].(map[string]any)
		require.Equal(t, "value", metadata["custom"])

		streamOptions := call.body["stream_options"].(map[string]any)
		require.Equal(t, true, streamOptions["include_usage"])

		messages := call.body["messages"].([]any)
		require.Len(t, messages, 1)

		message := messages[0].(map[string]any)
		require.Equal(t, "user", message["role"])
		require.Equal(t, "Hello", message["content"])
	})

	t.Run("should send serviceTier flex processing setting in streaming", func(t *testing.T) {
		t.Parallel()

		server := newStreamingMockServer()
		defer server.close()

		server.prepareStreamResponse(map[string]any{
			"content": []string{},
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "o3-mini")

		_, err = model.Stream(context.Background(), fantasy.Call{
			Prompt: testPrompt,
			ProviderOptions: NewProviderOptions(&ProviderOptions{
				ServiceTier: fantasy.Opt("flex"),
			}),
		})

		require.NoError(t, err)
		require.Len(t, server.calls, 1)

		call := server.calls[0]
		require.Equal(t, "o3-mini", call.body["model"])
		require.Equal(t, "flex", call.body["service_tier"])
		require.Equal(t, true, call.body["stream"])

		streamOptions := call.body["stream_options"].(map[string]any)
		require.Equal(t, true, streamOptions["include_usage"])

		messages := call.body["messages"].([]any)
		require.Len(t, messages, 1)

		message := messages[0].(map[string]any)
		require.Equal(t, "user", message["role"])
		require.Equal(t, "Hello", message["content"])
	})

	t.Run("should send serviceTier priority processing setting in streaming", func(t *testing.T) {
		t.Parallel()

		server := newStreamingMockServer()
		defer server.close()

		server.prepareStreamResponse(map[string]any{
			"content": []string{},
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "gpt-4o-mini")

		_, err = model.Stream(context.Background(), fantasy.Call{
			Prompt: testPrompt,
			ProviderOptions: NewProviderOptions(&ProviderOptions{
				ServiceTier: fantasy.Opt("priority"),
			}),
		})

		require.NoError(t, err)
		require.Len(t, server.calls, 1)

		call := server.calls[0]
		require.Equal(t, "gpt-4o-mini", call.body["model"])
		require.Equal(t, "priority", call.body["service_tier"])
		require.Equal(t, true, call.body["stream"])

		streamOptions := call.body["stream_options"].(map[string]any)
		require.Equal(t, true, streamOptions["include_usage"])

		messages := call.body["messages"].([]any)
		require.Len(t, messages, 1)

		message := messages[0].(map[string]any)
		require.Equal(t, "user", message["role"])
		require.Equal(t, "Hello", message["content"])
	})

	t.Run("should stream text delta for reasoning models", func(t *testing.T) {
		t.Parallel()

		server := newStreamingMockServer()
		defer server.close()

		server.prepareStreamResponse(map[string]any{
			"content": []string{"Hello, World!"},
			"model":   "o1-preview",
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "o1-preview")

		stream, err := model.Stream(context.Background(), fantasy.Call{
			Prompt: testPrompt,
		})

		require.NoError(t, err)

		parts, err := collectStreamParts(stream)
		require.NoError(t, err)

		// Find text parts
		var textDeltas []string
		for _, part := range parts {
			if part.Type == fantasy.StreamPartTypeTextDelta {
				textDeltas = append(textDeltas, part.Delta)
			}
		}

		// Should contain the text content (without empty delta)
		require.Equal(t, []string{"Hello, World!"}, textDeltas)
	})

	t.Run("should send reasoning tokens", func(t *testing.T) {
		t.Parallel()

		server := newStreamingMockServer()
		defer server.close()

		server.prepareStreamResponse(map[string]any{
			"content": []string{"Hello, World!"},
			"model":   "o1-preview",
			"usage": map[string]any{
				"prompt_tokens":     15,
				"completion_tokens": 20,
				"total_tokens":      35,
				"completion_tokens_details": map[string]any{
					"reasoning_tokens": 10,
				},
			},
		})

		provider, err := New(
			WithAPIKey("test-api-key"),
			WithBaseURL(server.server.URL),
		)
		require.NoError(t, err)
		model, _ := provider.LanguageModel(t.Context(), "o1-preview")

		stream, err := model.Stream(context.Background(), fantasy.Call{
			Prompt: testPrompt,
		})

		require.NoError(t, err)

		parts, err := collectStreamParts(stream)
		require.NoError(t, err)

		// Find finish part
		var finishPart *fantasy.StreamPart
		for _, part := range parts {
			if part.Type == fantasy.StreamPartTypeFinish {
				finishPart = &part
				break
			}
		}

		require.NotNil(t, finishPart)
		require.Equal(t, int64(15), finishPart.Usage.InputTokens)
		require.Equal(t, int64(20), finishPart.Usage.OutputTokens)
		require.Equal(t, int64(35), finishPart.Usage.TotalTokens)
		require.Equal(t, int64(10), finishPart.Usage.ReasoningTokens)
	})
}

func TestDefaultToPrompt_DropsEmptyMessages(t *testing.T) {
	t.Parallel()

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

		messages, warnings := DefaultToPrompt(prompt, "openai", "gpt-4")

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

		messages, warnings := DefaultToPrompt(prompt, "openai", "gpt-4")

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

		messages, warnings := DefaultToPrompt(prompt, "openai", "gpt-4")

		require.Len(t, messages, 2, "should have both user and assistant messages")
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
						MediaType: "application/unknown",
					},
				},
			},
		}

		messages, warnings := DefaultToPrompt(prompt, "openai", "gpt-4")

		require.Empty(t, messages)
		require.Len(t, warnings, 2) // One for unsupported type, one for empty message
		require.Contains(t, warnings[1].Message, "dropping empty user message")
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

		messages, warnings := DefaultToPrompt(prompt, "openai", "gpt-4")

		require.Len(t, messages, 1)
		require.Empty(t, warnings)
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

		messages, warnings := DefaultToPrompt(prompt, "openai", "gpt-4")

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

		messages, warnings := DefaultToPrompt(prompt, "openai", "gpt-4")

		require.Len(t, messages, 1)
		require.Empty(t, warnings)
	})
}

func TestResponsesToPrompt_DropsEmptyMessages(t *testing.T) {
	t.Parallel()

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

		input, warnings := toResponsesPrompt(prompt, "system", false)

		require.Len(t, input, 1, "should only have user message")
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

		input, warnings := toResponsesPrompt(prompt, "system", false)

		require.Len(t, input, 2, "should have both user and assistant messages")
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

		input, warnings := toResponsesPrompt(prompt, "system", false)

		require.Len(t, input, 2, "should have both user and assistant messages")
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
						MediaType: "application/unknown",
					},
				},
			},
		}

		input, warnings := toResponsesPrompt(prompt, "system", false)

		require.Empty(t, input)
		require.Len(t, warnings, 2) // One for unsupported type, one for empty message
		require.Contains(t, warnings[1].Message, "dropping empty user message")
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

		input, warnings := toResponsesPrompt(prompt, "system", false)

		require.Len(t, input, 1)
		require.Empty(t, warnings)
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

		input, warnings := toResponsesPrompt(prompt, "system", false)

		require.Len(t, input, 1)
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

		input, warnings := toResponsesPrompt(prompt, "system", false)

		require.Len(t, input, 1)
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
			name:     "matches openai format with resulted in",
			message:  "This model's maximum context length is 128000 tokens. However, your messages resulted in 150000 tokens.",
			wantErr:  true,
			wantUsed: 150000,
			wantMax:  128000,
		},
		{
			name:     "matches openai format with requested",
			message:  "maximum context length is 8192 tokens, however you requested 10000 tokens",
			wantErr:  true,
			wantUsed: 10000,
			wantMax:  8192,
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
				if tt.wantUsed > 0 {
					require.Equal(t, tt.wantUsed, providerErr.ContextUsedTokens)
					require.Equal(t, tt.wantMax, providerErr.ContextMaxTokens)
				}
			} else {
				require.False(t, providerErr.IsContextTooLarge())
			}
		})
	}
}

func TestUserAgent(t *testing.T) {
	t.Parallel()

	t.Run("default UA applied", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()
		server.prepareJSONResponse(map[string]any{})

		p, err := New(WithAPIKey("k"), WithBaseURL(server.server.URL))
		require.NoError(t, err)
		model, _ := p.LanguageModel(t.Context(), "gpt-4")
		_, _ = model.Generate(t.Context(), fantasy.Call{Prompt: testPrompt})

		require.Len(t, server.calls, 1)
		assert.Equal(t, "Charm-Fantasy/"+fantasy.Version+" (https://charm.land/fantasy)", server.calls[0].headers["User-Agent"])
	})

	t.Run("WithHeaders User-Agent wins over default", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()
		server.prepareJSONResponse(map[string]any{})

		p, err := New(WithAPIKey("k"), WithBaseURL(server.server.URL), WithHeaders(map[string]string{"User-Agent": "custom-from-headers"}))
		require.NoError(t, err)
		model, _ := p.LanguageModel(t.Context(), "gpt-4")
		_, _ = model.Generate(t.Context(), fantasy.Call{Prompt: testPrompt})

		require.Len(t, server.calls, 1)
		assert.Equal(t, "custom-from-headers", server.calls[0].headers["User-Agent"])
	})

	t.Run("WithUserAgent wins over both", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()
		server.prepareJSONResponse(map[string]any{})

		p, err := New(
			WithAPIKey("k"),
			WithBaseURL(server.server.URL),
			WithHeaders(map[string]string{"User-Agent": "from-headers"}),
			WithUserAgent("explicit-ua"),
		)
		require.NoError(t, err)
		model, _ := p.LanguageModel(t.Context(), "gpt-4")
		_, _ = model.Generate(t.Context(), fantasy.Call{Prompt: testPrompt})

		require.Len(t, server.calls, 1)
		assert.Equal(t, "explicit-ua", server.calls[0].headers["User-Agent"])
	})

	t.Run("Call.UserAgent overrides provider WithHeaders UA", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()
		server.prepareJSONResponse(map[string]any{})

		p, err := New(
			WithAPIKey("k"),
			WithBaseURL(server.server.URL),
			WithHeaders(map[string]string{"User-Agent": "header-ua"}),
		)
		require.NoError(t, err)
		model, _ := p.LanguageModel(t.Context(), "gpt-4")
		_, _ = model.Generate(t.Context(), fantasy.Call{
			Prompt:    testPrompt,
			UserAgent: "call-level-ua",
		})

		require.Len(t, server.calls, 1)
		assert.Equal(t, "call-level-ua", server.calls[0].headers["User-Agent"])
	})

	t.Run("no Call UA falls through to provider UA", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()
		server.prepareJSONResponse(map[string]any{})

		p, err := New(
			WithAPIKey("k"),
			WithBaseURL(server.server.URL),
			WithUserAgent("provider-ua"),
		)
		require.NoError(t, err)
		model, _ := p.LanguageModel(t.Context(), "gpt-4")
		_, _ = model.Generate(t.Context(), fantasy.Call{Prompt: testPrompt})

		require.Len(t, server.calls, 1)
		assert.Equal(t, "provider-ua", server.calls[0].headers["User-Agent"])
	})

	t.Run("agent WithUserAgent overrides provider UA end-to-end", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()
		server.prepareJSONResponse(map[string]any{})

		p, err := New(
			WithAPIKey("k"),
			WithBaseURL(server.server.URL),
			WithUserAgent("provider-ua"),
		)
		require.NoError(t, err)
		model, _ := p.LanguageModel(t.Context(), "gpt-4")

		agent := fantasy.NewAgent(model, fantasy.WithUserAgent("agent-ua"))
		_, _ = agent.Generate(t.Context(), fantasy.AgentCall{Prompt: "hi"})

		require.Len(t, server.calls, 1)
		assert.Equal(t, "agent-ua", server.calls[0].headers["User-Agent"])
	})

	t.Run("agent without UA falls through to provider UA end-to-end", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()
		server.prepareJSONResponse(map[string]any{})

		p, err := New(
			WithAPIKey("k"),
			WithBaseURL(server.server.URL),
			WithUserAgent("provider-ua"),
		)
		require.NoError(t, err)
		model, _ := p.LanguageModel(t.Context(), "gpt-4")

		agent := fantasy.NewAgent(model)
		_, _ = agent.Generate(t.Context(), fantasy.AgentCall{Prompt: "hi"})

		require.Len(t, server.calls, 1)
		assert.Equal(t, "provider-ua", server.calls[0].headers["User-Agent"])
	})
}

// --- OpenAI Responses API Web Search Tests ---

// mockResponsesWebSearchResponse returns a Responses API response
// containing a web_search_call output item followed by a message
// with url_citation annotations.
func mockResponsesWebSearchResponse() map[string]any {
	return map[string]any{
		"id":     "resp_01WebSearch",
		"object": "response",
		"model":  "gpt-4.1",
		"output": []any{
			map[string]any{
				"type":   "web_search_call",
				"id":     "ws_01",
				"status": "completed",
				"action": map[string]any{
					"type":  "search",
					"query": "latest AI news",
				},
			},
			map[string]any{
				"type":   "message",
				"id":     "msg_01",
				"role":   "assistant",
				"status": "completed",
				"content": []any{
					map[string]any{
						"type": "output_text",
						"text": "Based on recent search results, here is the latest AI news.",
						"annotations": []any{
							map[string]any{
								"type":        "url_citation",
								"url":         "https://example.com/ai-news",
								"title":       "Latest AI News",
								"start_index": 0,
								"end_index":   50,
							},
							map[string]any{
								"type":        "url_citation",
								"url":         "https://example.com/ml-update",
								"title":       "ML Update",
								"start_index": 51,
								"end_index":   60,
							},
						},
					},
				},
			},
		},
		"status": "completed",
		"usage": map[string]any{
			"input_tokens":  100,
			"output_tokens": 50,
			"total_tokens":  150,
		},
	}
}

func newResponsesProvider(t *testing.T, serverURL string) fantasy.LanguageModel {
	t.Helper()
	provider, err := New(
		WithAPIKey("test-api-key"),
		WithBaseURL(serverURL),
		WithUseResponsesAPI(),
	)
	require.NoError(t, err)
	model, err := provider.LanguageModel(context.Background(), "gpt-4.1")
	require.NoError(t, err)
	return model
}

func TestResponsesGenerate_WebSearchResponse(t *testing.T) {
	t.Parallel()

	server := newMockServer()
	defer server.close()
	server.response = mockResponsesWebSearchResponse()

	model := newResponsesProvider(t, server.server.URL)

	resp, err := model.Generate(context.Background(), fantasy.Call{
		Prompt: testPrompt,
		Tools:  []fantasy.Tool{WebSearchTool(nil)},
	})
	require.NoError(t, err)

	require.Equal(t, "POST", server.calls[0].method)
	require.Equal(t, "/responses", server.calls[0].path)

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
	require.Equal(t, "ws_01", toolCalls[0].ToolCallID)

	// SourceContent entries from url_citation annotations.
	require.Len(t, sources, 2)
	require.Equal(t, "https://example.com/ai-news", sources[0].URL)
	require.Equal(t, "Latest AI News", sources[0].Title)
	require.Equal(t, fantasy.SourceTypeURL, sources[0].SourceType)
	require.Equal(t, "https://example.com/ml-update", sources[1].URL)
	require.Equal(t, "ML Update", sources[1].Title)

	// ToolResultContent with provider metadata.
	require.Len(t, toolResults, 1)
	require.True(t, toolResults[0].ProviderExecuted)
	require.Equal(t, "web_search", toolResults[0].ToolName)
	require.Equal(t, "ws_01", toolResults[0].ToolCallID)

	metaVal, ok := toolResults[0].ProviderMetadata[Name]
	require.True(t, ok, "providerMetadata should contain openai key")
	wsMeta, ok := metaVal.(*WebSearchCallMetadata)
	require.True(t, ok, "metadata should be *WebSearchCallMetadata")
	require.Equal(t, "ws_01", wsMeta.ItemID)
	require.NotNil(t, wsMeta.Action)
	require.Equal(t, "search", wsMeta.Action.Type)
	require.Equal(t, "latest AI news", wsMeta.Action.Query)

	// TextContent with the final answer.
	require.Len(t, texts, 1)
	require.Equal(t,
		"Based on recent search results, here is the latest AI news.",
		texts[0].Text,
	)
}

func TestResponsesGenerate_StoreOption(t *testing.T) {
	t.Parallel()

	server := newMockServer()
	defer server.close()
	server.response = mockResponsesWebSearchResponse()

	model := newResponsesProvider(t, server.server.URL)

	_, err := model.Generate(context.Background(), fantasy.Call{
		Prompt: testPrompt,
		ProviderOptions: fantasy.ProviderOptions{
			Name: &ResponsesProviderOptions{
				Store: fantasy.Opt(true),
			},
		},
	})
	require.NoError(t, err)

	require.Equal(t, "POST", server.calls[0].method)
	require.Equal(t, "/responses", server.calls[0].path)
	require.Equal(t, true, server.calls[0].body["store"])
}

func TestResponsesGenerate_PreviousResponseIDOption(t *testing.T) {
	t.Parallel()

	server := newMockServer()
	defer server.close()
	server.response = mockResponsesWebSearchResponse()

	model := newResponsesProvider(t, server.server.URL)

	_, err := model.Generate(context.Background(), fantasy.Call{
		Prompt: testPrompt,
		ProviderOptions: fantasy.ProviderOptions{
			Name: &ResponsesProviderOptions{
				PreviousResponseID: fantasy.Opt("resp_prev_123"),
				Store:              fantasy.Opt(true),
			},
		},
	})
	require.NoError(t, err)

	require.Equal(t, "POST", server.calls[0].method)
	require.Equal(t, "/responses", server.calls[0].path)
	require.Equal(t, "resp_prev_123", server.calls[0].body["previous_response_id"])
}

func TestResponsesGenerate_StateChainingAcrossTurns(t *testing.T) {
	t.Parallel()

	server := newMockServer()
	defer server.close()
	server.response = map[string]any{
		"id":     "resp_turn_1",
		"object": "response",
		"model":  "gpt-4.1",
		"output": []any{
			map[string]any{
				"type":   "message",
				"id":     "msg_1",
				"role":   "assistant",
				"status": "completed",
				"content": []any{
					map[string]any{
						"type": "output_text",
						"text": "First turn",
					},
				},
			},
		},
		"status": "completed",
		"usage": map[string]any{
			"input_tokens":  10,
			"output_tokens": 5,
			"total_tokens":  15,
		},
	}

	model := newResponsesProvider(t, server.server.URL)

	first, err := model.Generate(context.Background(), fantasy.Call{
		Prompt: testPrompt,
		ProviderOptions: fantasy.ProviderOptions{
			Name: &ResponsesProviderOptions{Store: fantasy.Opt(true)},
		},
	})
	require.NoError(t, err)

	meta, ok := first.ProviderMetadata[Name].(*ResponsesProviderMetadata)
	require.True(t, ok)
	require.Equal(t, "resp_turn_1", meta.ResponseID)

	server.response = map[string]any{
		"id":     "resp_turn_2",
		"object": "response",
		"model":  "gpt-4.1",
		"output": []any{
			map[string]any{
				"type":   "message",
				"id":     "msg_2",
				"role":   "assistant",
				"status": "completed",
				"content": []any{
					map[string]any{
						"type": "output_text",
						"text": "Second turn",
					},
				},
			},
		},
		"status": "completed",
		"usage": map[string]any{
			"input_tokens":  8,
			"output_tokens": 4,
			"total_tokens":  12,
		},
	}

	_, err = model.Generate(context.Background(), fantasy.Call{
		Prompt: fantasy.Prompt{
			fantasy.NewUserMessage("follow-up only"),
		},
		ProviderOptions: fantasy.ProviderOptions{
			Name: &ResponsesProviderOptions{
				Store:              fantasy.Opt(true),
				PreviousResponseID: &meta.ResponseID,
			},
		},
	})
	require.NoError(t, err)
	require.Len(t, server.calls, 2)

	firstCall := server.calls[0]
	require.Equal(t, true, firstCall.body["store"])

	secondCall := server.calls[1]
	require.Equal(t, "resp_turn_1", secondCall.body["previous_response_id"])
	require.Equal(t, true, secondCall.body["store"])

	input, ok := secondCall.body["input"].([]any)
	require.True(t, ok)
	require.Len(t, input, 1)

	inputMessage, ok := input[0].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "user", inputMessage["role"])
}

func TestResponsesGenerate_WebSearchToolInRequest(t *testing.T) {
	t.Parallel()

	t.Run("basic web_search tool", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()
		server.response = mockResponsesWebSearchResponse()

		model := newResponsesProvider(t, server.server.URL)

		_, err := model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
			Tools:  []fantasy.Tool{WebSearchTool(nil)},
		})
		require.NoError(t, err)

		tools, ok := server.calls[0].body["tools"].([]any)
		require.True(t, ok, "request body should have tools array")
		require.Len(t, tools, 1)

		tool, ok := tools[0].(map[string]any)
		require.True(t, ok)
		require.Equal(t, "web_search", tool["type"])
	})

	t.Run("with search_context_size and allowed_domains", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()
		server.response = mockResponsesWebSearchResponse()

		model := newResponsesProvider(t, server.server.URL)

		_, err := model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
			Tools: []fantasy.Tool{
				WebSearchTool(&WebSearchToolOptions{
					SearchContextSize: SearchContextSizeHigh,
					AllowedDomains:    []string{"example.com", "test.com"},
				}),
			},
		})
		require.NoError(t, err)

		tools, ok := server.calls[0].body["tools"].([]any)
		require.True(t, ok)
		require.Len(t, tools, 1)

		tool, ok := tools[0].(map[string]any)
		require.True(t, ok)
		require.Equal(t, "web_search", tool["type"])
		require.Equal(t, "high", tool["search_context_size"])

		filters, ok := tool["filters"].(map[string]any)
		require.True(t, ok, "tool should have filters")
		domains, ok := filters["allowed_domains"].([]any)
		require.True(t, ok, "filters should have allowed_domains")
		require.Len(t, domains, 2)
		require.Equal(t, "example.com", domains[0])
		require.Equal(t, "test.com", domains[1])
	})

	t.Run("with user_location", func(t *testing.T) {
		t.Parallel()

		server := newMockServer()
		defer server.close()
		server.response = mockResponsesWebSearchResponse()

		model := newResponsesProvider(t, server.server.URL)

		_, err := model.Generate(context.Background(), fantasy.Call{
			Prompt: testPrompt,
			Tools: []fantasy.Tool{
				WebSearchTool(&WebSearchToolOptions{
					UserLocation: &WebSearchUserLocation{
						City:    "San Francisco",
						Country: "US",
					},
				}),
			},
		})
		require.NoError(t, err)

		tools, ok := server.calls[0].body["tools"].([]any)
		require.True(t, ok)
		require.Len(t, tools, 1)

		tool, ok := tools[0].(map[string]any)
		require.True(t, ok)
		require.Equal(t, "web_search", tool["type"])

		userLoc, ok := tool["user_location"].(map[string]any)
		require.True(t, ok, "tool should have user_location")
		require.Equal(t, "San Francisco", userLoc["city"])
		require.Equal(t, "US", userLoc["country"])
	})
}

func TestResponsesToPrompt_WebSearchProviderExecutedToolResults(t *testing.T) {
	t.Parallel()

	prompt := fantasy.Prompt{
		{
			Role: fantasy.MessageRoleUser,
			Content: []fantasy.MessagePart{
				fantasy.TextPart{Text: "Search for the latest AI news"},
			},
		},
		{
			Role: fantasy.MessageRoleAssistant,
			Content: []fantasy.MessagePart{
				fantasy.ToolCallPart{
					ToolCallID:       "ws_01",
					ToolName:         "web_search",
					ProviderExecuted: true,
				},
				fantasy.ToolResultPart{
					ToolCallID:       "ws_01",
					ProviderExecuted: true,
				},
				fantasy.TextPart{Text: "Here is what I found."},
			},
		},
	}

	t.Run("store false skips item reference", func(t *testing.T) {
		t.Parallel()

		input, warnings := toResponsesPrompt(prompt, "system instructions", false)

		require.Empty(t, warnings)
		require.Len(t, input, 2,
			"expected user + assistant text when store=false")
		require.Nil(t, input[0].OfItemReference)
		require.Nil(t, input[1].OfItemReference)
	})

	t.Run("store true uses item reference", func(t *testing.T) {
		t.Parallel()

		input, warnings := toResponsesPrompt(prompt, "system instructions", true)

		require.Empty(t, warnings)
		require.Len(t, input, 3,
			"expected user + item_reference + assistant text when store=true")
		require.NotNil(t, input[1].OfItemReference)
		require.Equal(t, "ws_01", input[1].OfItemReference.ID)
	})
}

func TestResponsesToPrompt_ReasoningWithStore(t *testing.T) {
	t.Parallel()

	encryptedContent := "gAAAAABpvAwtDPh5dSXW86hwbwoTo4DJHANQ"
	reasoningItemID := "rs_08d030b87966238b0069bc095b7e5c81"

	reasoningPart := fantasy.ReasoningPart{
		Text: "Let me think about this...",
		ProviderOptions: fantasy.ProviderOptions{
			Name: &ResponsesReasoningMetadata{
				ItemID:           reasoningItemID,
				EncryptedContent: &encryptedContent,
				Summary:          []string{},
			},
		},
	}

	prompt := fantasy.Prompt{
		{
			Role: fantasy.MessageRoleUser,
			Content: []fantasy.MessagePart{
				fantasy.TextPart{Text: "What is 2+2?"},
			},
		},
		{
			Role: fantasy.MessageRoleAssistant,
			Content: []fantasy.MessagePart{
				reasoningPart,
				fantasy.TextPart{Text: "4"},
			},
		},
		{
			Role: fantasy.MessageRoleUser,
			Content: []fantasy.MessagePart{
				fantasy.TextPart{Text: "And 3+3?"},
			},
		},
	}

	t.Run("store true skips reasoning", func(t *testing.T) {
		t.Parallel()

		input, warnings := toResponsesPrompt(prompt, "system", true)
		require.Empty(t, warnings)

		// With store=true: user, assistant text (reasoning
		// skipped), follow-up user.
		require.Len(t, input, 3)

		// Verify no reasoning item leaked through.
		for _, item := range input {
			require.Nil(t, item.OfReasoning,
				"reasoning items must not appear when store=true")
		}
	})

	t.Run("store false skips reasoning", func(t *testing.T) {
		t.Parallel()

		input, warnings := toResponsesPrompt(prompt, "system", false)
		require.Empty(t, warnings)

		// With store=false: user, assistant text, follow-up user.
		require.Len(t, input, 3)

		for _, item := range input {
			require.Nil(t, item.OfReasoning,
				"reasoning items must not appear when store=false")
		}
	})
}

func TestResponsesStream_WebSearchResponse(t *testing.T) {
	t.Parallel()

	chunks := []string{
		"event: response.output_item.added\n" +
			`data: {"type":"response.output_item.added","output_index":0,"item":{"type":"web_search_call","id":"ws_01","status":"in_progress"}}` + "\n\n",
		"event: response.output_item.done\n" +
			`data: {"type":"response.output_item.done","output_index":0,"item":{"type":"web_search_call","id":"ws_01","status":"completed","action":{"type":"search","query":"latest AI news"}}}` + "\n\n",
		"event: response.output_item.added\n" +
			`data: {"type":"response.output_item.added","output_index":1,"item":{"type":"message","id":"msg_01","role":"assistant","status":"in_progress","content":[]}}` + "\n\n",
		"event: response.output_text.delta\n" +
			`data: {"type":"response.output_text.delta","output_index":1,"content_index":0,"delta":"Here are the results."}` + "\n\n",
		"event: response.output_text.annotation.added\n" +
			`data: {"type":"response.output_text.annotation.added","annotation":{"type":"url_citation","url":"https://example.com/ai-news","title":"Latest AI News","start_index":0,"end_index":21},"annotation_index":0,"content_index":0,"item_id":"msg_01","output_index":1,"sequence_number":10}` + "\n\n",
		"event: response.output_text.annotation.added\n" +
			`data: {"type":"response.output_text.annotation.added","annotation":{"type":"url_citation","url":"https://example.com/more-news","title":"More AI News","start_index":22,"end_index":40},"annotation_index":1,"content_index":0,"item_id":"msg_01","output_index":1,"sequence_number":11}` + "\n\n",
		"event: response.output_item.done\n" +
			`data: {"type":"response.output_item.done","output_index":1,"item":{"type":"message","id":"msg_01","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Here are the results.","annotations":[{"type":"url_citation","url":"https://example.com/ai-news","title":"Latest AI News","start_index":0,"end_index":21},{"type":"url_citation","url":"https://example.com/more-news","title":"More AI News","start_index":22,"end_index":40}]}]}}` + "\n\n",
		"event: response.completed\n" +
			`data: {"type":"response.completed","response":{"id":"resp_01","status":"completed","output":[],"usage":{"input_tokens":100,"output_tokens":50,"total_tokens":150}}}` + "\n\n",
	}

	sms := newStreamingMockServer()
	defer sms.close()
	sms.chunks = chunks

	model := newResponsesProvider(t, sms.server.URL)

	stream, err := model.Stream(context.Background(), fantasy.Call{
		Prompt: testPrompt,
		Tools:  []fantasy.Tool{WebSearchTool(nil)},
	})
	require.NoError(t, err)

	var parts []fantasy.StreamPart
	stream(func(part fantasy.StreamPart) bool {
		parts = append(parts, part)
		return true
	})

	var (
		toolInputStarts []fantasy.StreamPart
		toolCalls       []fantasy.StreamPart
		toolResults     []fantasy.StreamPart
		textDeltas      []fantasy.StreamPart
		sources         []fantasy.StreamPart
		finishes        []fantasy.StreamPart
	)
	for _, p := range parts {
		switch p.Type {
		case fantasy.StreamPartTypeToolInputStart:
			toolInputStarts = append(toolInputStarts, p)
		case fantasy.StreamPartTypeToolCall:
			toolCalls = append(toolCalls, p)
		case fantasy.StreamPartTypeToolResult:
			toolResults = append(toolResults, p)
		case fantasy.StreamPartTypeTextDelta:
			textDeltas = append(textDeltas, p)
		case fantasy.StreamPartTypeSource:
			sources = append(sources, p)
		case fantasy.StreamPartTypeFinish:
			finishes = append(finishes, p)
		}
	}

	require.NotEmpty(t, toolInputStarts, "should have a tool input start")
	require.True(t, toolInputStarts[0].ProviderExecuted)
	require.Equal(t, "web_search", toolInputStarts[0].ToolCallName)

	require.NotEmpty(t, toolCalls, "should have a tool call")
	require.True(t, toolCalls[0].ProviderExecuted)
	require.Equal(t, "web_search", toolCalls[0].ToolCallName)

	require.NotEmpty(t, toolResults, "should have a tool result")
	require.True(t, toolResults[0].ProviderExecuted)
	require.Equal(t, "web_search", toolResults[0].ToolCallName)
	require.Equal(t, "ws_01", toolResults[0].ID)

	require.NotEmpty(t, textDeltas, "should have text deltas")
	require.Equal(t, "Here are the results.", textDeltas[0].Delta)

	require.Len(t, sources, 2, "should have two source citations from annotation events")
	require.Equal(t, fantasy.SourceTypeURL, sources[0].SourceType)
	require.Equal(t, "https://example.com/ai-news", sources[0].URL)
	require.Equal(t, "Latest AI News", sources[0].Title)
	require.NotEmpty(t, sources[0].ID, "source should have an ID")
	require.Equal(t, fantasy.SourceTypeURL, sources[1].SourceType)
	require.Equal(t, "https://example.com/more-news", sources[1].URL)
	require.Equal(t, "More AI News", sources[1].Title)
	require.NotEmpty(t, sources[1].ID, "source should have an ID")

	require.Len(t, finishes, 1)
	responsesMeta, ok := finishes[0].ProviderMetadata[Name].(*ResponsesProviderMetadata)
	require.True(t, ok)
	require.Equal(t, "resp_01", responsesMeta.ResponseID)
}

func TestResponsesStream_StoreOption(t *testing.T) {
	t.Parallel()

	chunks := []string{
		"event: response.completed\n" +
			`data: {"type":"response.completed","response":{"id":"resp_01","status":"completed","output":[],"usage":{"input_tokens":100,"output_tokens":50,"total_tokens":150}}}` + "\n\n",
	}

	sms := newStreamingMockServer()
	defer sms.close()
	sms.chunks = chunks

	model := newResponsesProvider(t, sms.server.URL)

	stream, err := model.Stream(context.Background(), fantasy.Call{
		Prompt: testPrompt,
		ProviderOptions: fantasy.ProviderOptions{
			Name: &ResponsesProviderOptions{
				Store: fantasy.Opt(true),
			},
		},
	})
	require.NoError(t, err)

	stream(func(part fantasy.StreamPart) bool {
		return part.Type != fantasy.StreamPartTypeFinish
	})

	require.Equal(t, "POST", sms.calls[0].method)
	require.Equal(t, "/responses", sms.calls[0].path)
	require.Equal(t, true, sms.calls[0].body["store"])
}

func TestResponsesStream_PreviousResponseIDOption(t *testing.T) {
	t.Parallel()

	chunks := []string{
		"event: response.completed\n" +
			`data: {"type":"response.completed","response":{"id":"resp_01","status":"completed","output":[],"usage":{"input_tokens":100,"output_tokens":50,"total_tokens":150}}}` + "\n\n",
	}

	sms := newStreamingMockServer()
	defer sms.close()
	sms.chunks = chunks

	model := newResponsesProvider(t, sms.server.URL)

	stream, err := model.Stream(context.Background(), fantasy.Call{
		Prompt: testPrompt,
		ProviderOptions: fantasy.ProviderOptions{
			Name: &ResponsesProviderOptions{
				PreviousResponseID: fantasy.Opt("resp_prev_456"),
				Store:              fantasy.Opt(true),
			},
		},
	})
	require.NoError(t, err)

	stream(func(part fantasy.StreamPart) bool {
		return part.Type != fantasy.StreamPartTypeFinish
	})

	require.Equal(t, "POST", sms.calls[0].method)
	require.Equal(t, "/responses", sms.calls[0].path)
	require.Equal(t, "resp_prev_456", sms.calls[0].body["previous_response_id"])
}
