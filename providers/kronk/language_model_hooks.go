package kronk

import (
	"encoding/base64"
	"fmt"
	"strings"

	"charm.land/fantasy"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
)

// LanguageModelPrepareCallFunc is a function that prepares the call for the language model.
type LanguageModelPrepareCallFunc func(lm fantasy.LanguageModel, d model.D, call fantasy.Call) ([]fantasy.CallWarning, error)

// LanguageModelMapFinishReasonFunc is a function that maps the finish reason for the language model.
type LanguageModelMapFinishReasonFunc func(finishReason string) fantasy.FinishReason

// LanguageModelToPromptFunc is a function that handles converting fantasy prompts to Kronk SDK messages.
type LanguageModelToPromptFunc func(prompt fantasy.Prompt, provider, modelID string) ([]model.D, []fantasy.CallWarning)

// DefaultPrepareCallFunc is the default implementation for preparing a call to the language model.
func DefaultPrepareCallFunc(lm fantasy.LanguageModel, d model.D, call fantasy.Call) ([]fantasy.CallWarning, error) {
	if call.ProviderOptions == nil {
		return nil, nil
	}

	var warnings []fantasy.CallWarning
	providerOptions := &ProviderOptions{}
	if v, ok := call.ProviderOptions[Name]; ok {
		providerOptions, ok = v.(*ProviderOptions)
		if !ok {
			return nil, &fantasy.Error{Title: "invalid argument", Message: "kronk provider options should be *kronk.ProviderOptions"}
		}
	}

	if providerOptions.TopK != nil {
		d["top_k"] = *providerOptions.TopK
	}

	if providerOptions.RepeatPenalty != nil {
		d["repeat_penalty"] = *providerOptions.RepeatPenalty
	}

	if providerOptions.Seed != nil {
		d["seed"] = *providerOptions.Seed
	}

	if providerOptions.MinP != nil {
		d["min_p"] = *providerOptions.MinP
	}

	if providerOptions.NumPredict != nil {
		d["num_predict"] = *providerOptions.NumPredict
	}

	if providerOptions.Stop != nil {
		d["stop"] = providerOptions.Stop
	}

	return warnings, nil
}

// DefaultMapFinishReasonFunc is the default implementation for mapping finish reasons.
func DefaultMapFinishReasonFunc(finishReason string) fantasy.FinishReason {
	switch finishReason {
	case string(model.FinishReasonStop):
		return fantasy.FinishReasonStop

	case string(model.FinishReasonTool):
		return fantasy.FinishReasonToolCalls

	case string(model.FinishReasonError):
		return fantasy.FinishReasonError

	default:
		return fantasy.FinishReasonUnknown
	}
}

// DefaultToPrompt is the default implementation for converting fantasy prompts to Kronk SDK messages.
func DefaultToPrompt(prompt fantasy.Prompt, provider, modelID string) ([]model.D, []fantasy.CallWarning) {
	var messages []model.D
	var warnings []fantasy.CallWarning

	for _, msg := range prompt {
		switch msg.Role {
		case fantasy.MessageRoleSystem:
			for _, c := range msg.Content {
				if c.GetType() == fantasy.ContentTypeText {
					textPart, ok := fantasy.AsMessagePart[fantasy.TextPart](c)
					if !ok {
						warnings = append(warnings, fantasy.CallWarning{
							Type:    fantasy.CallWarningTypeOther,
							Message: "system message text part does not have the right type",
						})

						continue
					}

					messages = append(messages, model.TextMessage(model.RoleSystem, textPart.Text))
				}
			}

		case fantasy.MessageRoleUser:
			var content []model.D
			for _, c := range msg.Content {
				switch c.GetType() {
				case fantasy.ContentTypeText:
					textPart, ok := fantasy.AsMessagePart[fantasy.TextPart](c)
					if !ok {
						warnings = append(warnings, fantasy.CallWarning{
							Type:    fantasy.CallWarningTypeOther,
							Message: "user message text part does not have the right type",
						})

						continue
					}

					content = append(content, model.D{
						"type": "text",
						"text": textPart.Text,
					})

				case fantasy.ContentTypeFile:
					filePart, ok := fantasy.AsMessagePart[fantasy.FilePart](c)
					if !ok {
						warnings = append(warnings, fantasy.CallWarning{
							Type:    fantasy.CallWarningTypeOther,
							Message: "user message file part does not have the right type",
						})

						continue
					}

					switch {
					case strings.HasPrefix(filePart.MediaType, "image/"):
						base64Encoded := base64.StdEncoding.EncodeToString(filePart.Data)
						data := "data:" + filePart.MediaType + ";base64," + base64Encoded
						content = append(content, model.D{
							"type": "image_url",
							"image_url": model.D{
								"url": data,
							},
						})

					default:
						warnings = append(warnings, fantasy.CallWarning{
							Type:    fantasy.CallWarningTypeOther,
							Message: fmt.Sprintf("file part media type %s not supported", filePart.MediaType),
						})
					}
				}
			}

			switch {
			case len(content) == 1 && content[0]["type"] == "text":
				messages = append(messages, model.TextMessage(model.RoleUser, content[0]["text"].(string)))

			case len(content) > 0:
				messages = append(messages, model.D{
					"role":    model.RoleUser,
					"content": content,
				})
			}

		case fantasy.MessageRoleAssistant:
			var textContent string
			var toolCalls []model.D

			for _, c := range msg.Content {
				switch c.GetType() {
				case fantasy.ContentTypeText:
					textPart, ok := fantasy.AsMessagePart[fantasy.TextPart](c)
					if !ok {
						warnings = append(warnings, fantasy.CallWarning{
							Type:    fantasy.CallWarningTypeOther,
							Message: "assistant message text part does not have the right type",
						})

						continue
					}

					textContent += textPart.Text

				case fantasy.ContentTypeToolCall:
					toolCallPart, ok := fantasy.AsMessagePart[fantasy.ToolCallPart](c)
					if !ok {
						warnings = append(warnings, fantasy.CallWarning{
							Type:    fantasy.CallWarningTypeOther,
							Message: "assistant message tool part does not have the right type",
						})

						continue
					}

					toolCalls = append(toolCalls, model.D{
						"id":   toolCallPart.ToolCallID,
						"type": "function",
						"function": model.D{
							"name":      toolCallPart.ToolName,
							"arguments": toolCallPart.Input,
						},
					})
				}
			}

			assistantMsg := model.D{
				"role": model.RoleAssistant,
			}

			if textContent != "" {
				assistantMsg["content"] = textContent
			}

			if len(toolCalls) > 0 {
				assistantMsg["tool_calls"] = toolCalls
			}

			if textContent != "" || len(toolCalls) > 0 {
				messages = append(messages, assistantMsg)
			}

		case fantasy.MessageRoleTool:
			for _, c := range msg.Content {
				if c.GetType() != fantasy.ContentTypeToolResult {
					warnings = append(warnings, fantasy.CallWarning{
						Type:    fantasy.CallWarningTypeOther,
						Message: "tool message can only have tool result content",
					})

					continue
				}

				toolResultPart, ok := fantasy.AsMessagePart[fantasy.ToolResultPart](c)
				if !ok {
					warnings = append(warnings, fantasy.CallWarning{
						Type:    fantasy.CallWarningTypeOther,
						Message: "tool message result part does not have the right type",
					})

					continue
				}

				var resultContent string
				switch toolResultPart.Output.GetType() {
				case fantasy.ToolResultContentTypeText:
					output, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentText](toolResultPart.Output)
					if ok {
						resultContent = output.Text
					}

				case fantasy.ToolResultContentTypeError:
					output, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentError](toolResultPart.Output)
					if ok {
						resultContent = output.Error.Error()
					}
				}

				messages = append(messages, model.D{
					"role":         "tool",
					"content":      resultContent,
					"tool_call_id": toolResultPart.ToolCallID,
				})
			}
		}
	}

	return messages, warnings
}
