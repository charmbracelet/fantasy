package ollamacloud

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"charm.land/fantasy"
)

type languageModel struct {
	provider *provider
	modelID  string
}

func (lm *languageModel) Generate(ctx context.Context, call fantasy.Call) (*fantasy.Response, error) {
	reqBody, err := lm.prepareRequest(call, false)
	if err != nil {
		return nil, err
	}

	resp, err := lm.provider.doRequest(ctx, reqBody)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var ollamaResp struct {
		Model     string `json:"model"`
		CreatedAt string `json:"created_at"`
		Message   struct {
			Role      string `json:"role"`
			Content   string `json:"content"`
			Thinking  string `json:"thinking,omitempty"`
			ToolCalls []struct {
				Function struct {
					Index     int            `json:"index"`
					Name      string         `json:"name"`
					Arguments map[string]any `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls,omitempty"`
		} `json:"message"`
		Done            bool   `json:"done"`
		DoneReason      string `json:"done_reason"`
		TotalDuration   int64  `json:"total_duration,omitempty"`
		PromptEvalCount int    `json:"prompt_eval_count,omitempty"`
		EvalCount       int    `json:"eval_count,omitempty"`
	}

	if err := json.Unmarshal(body, &ollamaResp); err != nil {
		return nil, fantasy.NewAIError("ParseError", "failed to parse response", err)
	}

	content := make([]fantasy.Content, 0)

	if ollamaResp.Message.Thinking != "" {
		content = append(content, fantasy.ReasoningContent{
			Text: ollamaResp.Message.Thinking,
		})
	}

	if ollamaResp.Message.Content != "" {
		content = append(content, fantasy.TextContent{
			Text: ollamaResp.Message.Content,
		})
	}

	for _, tc := range ollamaResp.Message.ToolCalls {
		argsJSON, _ := json.Marshal(tc.Function.Arguments)
		content = append(content, fantasy.ToolCallContent{
			ProviderExecuted: false,
			ToolCallID:       fmt.Sprintf("%d", tc.Function.Index),
			ToolName:         tc.Function.Name,
			Input:            string(argsJSON),
		})
	}

	finishReason := lm.mapFinishReason(ollamaResp.DoneReason)
	if len(ollamaResp.Message.ToolCalls) > 0 {
		finishReason = fantasy.FinishReasonToolCalls
	}

	return &fantasy.Response{
		Content: content,
		Usage: fantasy.Usage{
			InputTokens:  int64(ollamaResp.PromptEvalCount),
			OutputTokens: int64(ollamaResp.EvalCount),
		},
		FinishReason: finishReason,
	}, nil
}

func (lm *languageModel) Stream(ctx context.Context, call fantasy.Call) (fantasy.StreamResponse, error) {
	reqBody, err := lm.prepareRequest(call, true)
	if err != nil {
		return nil, err
	}

	resp, err := lm.provider.doRequest(ctx, reqBody)
	if err != nil {
		return nil, err
	}

	return func(yield func(fantasy.StreamPart) bool) {
		defer resp.Body.Close()

		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 0, 64*1024), 1<<20)
		var thinkingStarted bool
		var hadAnyToolCalls bool

		for scanner.Scan() {
			line := scanner.Bytes()
			if len(line) == 0 {
				continue
			}

			var chunk struct {
				Model   string `json:"model"`
				Message struct {
					Role      string `json:"role"`
					Content   string `json:"content"`
					Thinking  string `json:"thinking,omitempty"`
					ToolCalls []struct {
						Function struct {
							Index     int            `json:"index"`
							Name      string         `json:"name"`
							Arguments map[string]any `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls,omitempty"`
				} `json:"message"`
				Done            bool   `json:"done"`
				DoneReason      string `json:"done_reason,omitempty"`
				PromptEvalCount int    `json:"prompt_eval_count,omitempty"`
				EvalCount       int    `json:"eval_count,omitempty"`
			}

			if err := json.Unmarshal(line, &chunk); err != nil {
				yield(fantasy.StreamPart{
					Type:  fantasy.StreamPartTypeError,
					Error: fantasy.NewAIError("ParseError", "failed to parse chunk", err),
				})
				return
			}

			if chunk.Message.Thinking != "" {
				if !thinkingStarted {
					if !yield(fantasy.StreamPart{
						Type: fantasy.StreamPartTypeReasoningStart,
						ID:   "0",
					}) {
						return
					}
					thinkingStarted = true
				}
				if !yield(fantasy.StreamPart{
					Type:  fantasy.StreamPartTypeReasoningDelta,
					ID:    "0",
					Delta: chunk.Message.Thinking,
				}) {
					return
				}
			}

			if thinkingStarted && (chunk.Message.Content != "" || len(chunk.Message.ToolCalls) > 0) {
				if !yield(fantasy.StreamPart{
					Type: fantasy.StreamPartTypeReasoningEnd,
					ID:   "0",
				}) {
					return
				}
				thinkingStarted = false
			}

			if chunk.Message.Content != "" {
				if !yield(fantasy.StreamPart{
					Type:  fantasy.StreamPartTypeTextDelta,
					Delta: chunk.Message.Content,
				}) {
					return
				}
			}

			for _, tc := range chunk.Message.ToolCalls {
				if thinkingStarted {
					if !yield(fantasy.StreamPart{
						Type: fantasy.StreamPartTypeReasoningEnd,
						ID:   "0",
					}) {
						return
					}
					thinkingStarted = false
				}

				hadAnyToolCalls = true
				argsJSON, _ := json.Marshal(tc.Function.Arguments)
				if !yield(fantasy.StreamPart{
					Type:          fantasy.StreamPartTypeToolCall,
					ID:            fmt.Sprintf("%d", tc.Function.Index),
					ToolCallName:  tc.Function.Name,
					ToolCallInput: string(argsJSON),
				}) {
					return
				}
			}

			if chunk.Done {
				finishReason := lm.mapFinishReason(chunk.DoneReason)
				if hadAnyToolCalls {
					finishReason = fantasy.FinishReasonToolCalls
				}

				yield(fantasy.StreamPart{
					Type: fantasy.StreamPartTypeFinish,
					Usage: fantasy.Usage{
						InputTokens:  int64(chunk.PromptEvalCount),
						OutputTokens: int64(chunk.EvalCount),
					},
					FinishReason: finishReason,
				})
				return
			}
		}

		if err := scanner.Err(); err != nil {
			yield(fantasy.StreamPart{
				Type:  fantasy.StreamPartTypeError,
				Error: fantasy.NewAIError("StreamError", "failed to read stream", err),
			})
		}
	}, nil
}

func (lm *languageModel) Provider() string {
	return lm.provider.Name()
}

func (lm *languageModel) Model() string {
	return lm.modelID
}

func (lm *languageModel) mapFinishReason(reason string) fantasy.FinishReason {
	switch reason {
	case "stop":
		return fantasy.FinishReasonStop
	case "length":
		return fantasy.FinishReasonLength
	default:
		return fantasy.FinishReasonOther
	}
}

func (lm *languageModel) prepareRequest(call fantasy.Call, stream bool) (map[string]any, error) {
	messages := make([]map[string]any, 0, len(call.Prompt))
	toolCallNames := make(map[string]string)

	for _, msg := range call.Prompt {
		if msg.Role == fantasy.MessageRoleTool {
			for _, part := range msg.Content {
				toolResultPart, ok := fantasy.AsMessagePart[fantasy.ToolResultPart](part)
				if !ok {
					continue
				}

				var content string
				switch toolResultPart.Output.GetType() {
				case fantasy.ToolResultContentTypeText:
					if output, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentText](toolResultPart.Output); ok {
						content = output.Text
					}
				case fantasy.ToolResultContentTypeError:
					if output, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentError](toolResultPart.Output); ok && output.Error != nil {
						content = output.Error.Error()
					}
				case fantasy.ToolResultContentTypeMedia:
					if output, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentMedia](toolResultPart.Output); ok {
						media := map[string]any{
							"data":       output.Data,
							"media_type": output.MediaType,
						}
						if encoded, err := json.Marshal(media); err == nil {
							content = string(encoded)
						}
					}
				}

				toolMessage := map[string]any{
					"role":    msg.Role,
					"content": content,
				}
				if toolResultPart.ToolCallID != "" {
					toolMessage["tool_call_id"] = toolResultPart.ToolCallID
					if name := toolCallNames[toolResultPart.ToolCallID]; name != "" {
						toolMessage["tool_name"] = name
					}
				}
				messages = append(messages, toolMessage)
			}
			continue
		}

		ollamaMsg := map[string]any{
			"role": msg.Role,
		}

		var (
			hasToolCalls bool
			toolCalls    []map[string]any
			textBuilder  strings.Builder
			hasText      bool
		)

		for _, part := range msg.Content {
			if textPart, ok := fantasy.AsMessagePart[fantasy.TextPart](part); ok {
				hasText = true
				textBuilder.WriteString(textPart.Text)
				continue
			}

			if toolCallPart, ok := fantasy.AsMessagePart[fantasy.ToolCallPart](part); ok {
				if toolCallPart.ProviderExecuted {
					continue
				}
				var args map[string]any
				if toolCallPart.Input != "" {
					if err := json.Unmarshal([]byte(toolCallPart.Input), &args); err != nil {
						return nil, fantasy.NewInvalidArgumentError("tool_call.input", "invalid JSON in tool call input", err)
					}
				}
				if args == nil {
					args = map[string]any{}
				}
				hasToolCalls = true
				toolCalls = append(toolCalls, map[string]any{
					"type": "function",
					"function": map[string]any{
						"name":      toolCallPart.ToolName,
						"arguments": args,
					},
				})
				if toolCallPart.ToolCallID != "" {
					toolCalls[len(toolCalls)-1]["id"] = toolCallPart.ToolCallID
					toolCallNames[toolCallPart.ToolCallID] = toolCallPart.ToolName
				}
				continue
			}

			if reasoningPart, ok := fantasy.AsMessagePart[fantasy.ReasoningPart](part); ok {
				hasText = true
				textBuilder.WriteString(reasoningPart.Text)
				continue
			}

			if toolResultPart, ok := fantasy.AsMessagePart[fantasy.ToolResultPart](part); ok {
				var content string
				switch toolResultPart.Output.GetType() {
				case fantasy.ToolResultContentTypeText:
					if output, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentText](toolResultPart.Output); ok {
						content = output.Text
					}
				case fantasy.ToolResultContentTypeError:
					if output, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentError](toolResultPart.Output); ok && output.Error != nil {
						content = output.Error.Error()
					}
				case fantasy.ToolResultContentTypeMedia:
					if output, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentMedia](toolResultPart.Output); ok {
						media := map[string]any{
							"data":       output.Data,
							"media_type": output.MediaType,
						}
						if encoded, err := json.Marshal(media); err == nil {
							content = string(encoded)
						}
					}
				}
				toolMessage := map[string]any{
					"role":    fantasy.MessageRoleTool,
					"content": content,
				}
				if toolResultPart.ToolCallID != "" {
					toolMessage["tool_call_id"] = toolResultPart.ToolCallID
					if name := toolCallNames[toolResultPart.ToolCallID]; name != "" {
						toolMessage["tool_name"] = name
					}
				}
				messages = append(messages, toolMessage)
			}
		}

		if hasToolCalls {
			ollamaMsg["tool_calls"] = toolCalls
			if hasText {
				ollamaMsg["content"] = textBuilder.String()
			} else {
				ollamaMsg["content"] = ""
			}
		} else {
			if hasText {
				ollamaMsg["content"] = textBuilder.String()
			} else if _, hasContent := ollamaMsg["content"]; !hasContent {
				ollamaMsg["content"] = ""
			}
		}

		messages = append(messages, ollamaMsg)
	}

	reqBody := map[string]any{
		"model":    lm.modelID,
		"messages": messages,
		"stream":   stream,
	}

	if len(call.Tools) > 0 {
		tools := make([]map[string]any, 0, len(call.Tools))
		for _, tool := range call.Tools {
			funcTool, ok := tool.(fantasy.FunctionTool)
			if !ok {
				continue
			}
			tools = append(tools, map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        funcTool.Name,
					"description": funcTool.Description,
					"parameters":  funcTool.InputSchema,
				},
			})
		}
		reqBody["tools"] = tools
	}

	if opts, ok := call.ProviderOptions[Name]; ok {
		if providerOpts, ok := opts.(*ProviderOptions); ok {
			if providerOpts.Think != nil && *providerOpts.Think {
				reqBody["think"] = true
			}
		}
	}

	if call.Temperature != nil {
		reqBody["temperature"] = *call.Temperature
	}
	if call.TopP != nil {
		reqBody["top_p"] = *call.TopP
	}
	if call.TopK != nil {
		reqBody["top_k"] = *call.TopK
	}
	if call.MaxOutputTokens != nil {
		reqBody["max_tokens"] = *call.MaxOutputTokens
	}

	return reqBody, nil
}
