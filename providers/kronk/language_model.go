package kronk

import (
	"context"
	"encoding/json"
	"errors"
	"io"

	"charm.land/fantasy"
	"charm.land/fantasy/object"
	"github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
	xjson "github.com/charmbracelet/x/json"
	"github.com/google/uuid"
)

type languageModel struct {
	provider            string
	modelID             string
	kronk               *kronk.Kronk
	objectMode          fantasy.ObjectMode
	prepareCallFunc     LanguageModelPrepareCallFunc
	mapFinishReasonFunc LanguageModelMapFinishReasonFunc
	toPromptFunc        LanguageModelToPromptFunc
}

// LanguageModelOption is a function that configures a languageModel.
type LanguageModelOption func(*languageModel)

// WithLanguageModelPrepareCallFunc sets the prepare call function for the language model.
func WithLanguageModelPrepareCallFunc(fn LanguageModelPrepareCallFunc) LanguageModelOption {
	return func(l *languageModel) {
		l.prepareCallFunc = fn
	}
}

// WithLanguageModelMapFinishReasonFunc sets the map finish reason function for the language model.
func WithLanguageModelMapFinishReasonFunc(fn LanguageModelMapFinishReasonFunc) LanguageModelOption {
	return func(l *languageModel) {
		l.mapFinishReasonFunc = fn
	}
}

// WithLanguageModelToPromptFunc sets the to prompt function for the language model.
func WithLanguageModelToPromptFunc(fn LanguageModelToPromptFunc) LanguageModelOption {
	return func(l *languageModel) {
		l.toPromptFunc = fn
	}
}

// WithLanguageModelObjectMode sets the object generation mode.
func WithLanguageModelObjectMode(om fantasy.ObjectMode) LanguageModelOption {
	return func(l *languageModel) {
		l.objectMode = om
	}
}

func newLanguageModel(modelID string, provider string, krn *kronk.Kronk, opts ...LanguageModelOption) *languageModel {
	lm := languageModel{
		modelID:             modelID,
		provider:            provider,
		kronk:               krn,
		objectMode:          fantasy.ObjectModeAuto,
		prepareCallFunc:     DefaultPrepareCallFunc,
		mapFinishReasonFunc: DefaultMapFinishReasonFunc,
		toPromptFunc:        DefaultToPrompt,
	}

	for _, o := range opts {
		o(&lm)
	}

	return &lm
}

type streamToolCall struct {
	id          string
	name        string
	arguments   string
	hasFinished bool
}

// Model implements fantasy.LanguageModel.
func (l *languageModel) Model() string {
	return l.modelID
}

// Provider implements fantasy.LanguageModel.
func (l *languageModel) Provider() string {
	return l.provider
}

func (l *languageModel) prepareDocument(call fantasy.Call) (model.D, []fantasy.CallWarning, error) {
	messages, warnings := l.toPromptFunc(call.Prompt, l.provider, l.modelID)

	if call.TopK != nil {
		warnings = append(warnings, fantasy.CallWarning{
			Type:    fantasy.CallWarningTypeUnsupportedSetting,
			Setting: "top_k",
		})
	}

	d := model.D{
		"messages": messages,
	}

	if call.MaxOutputTokens != nil {
		d["max_tokens"] = *call.MaxOutputTokens
	}

	if call.Temperature != nil {
		d["temperature"] = *call.Temperature
	}

	if call.TopP != nil {
		d["top_p"] = *call.TopP
	}

	if call.FrequencyPenalty != nil {
		warnings = append(warnings, fantasy.CallWarning{
			Type:    fantasy.CallWarningTypeUnsupportedSetting,
			Setting: "frequency_penalty",
			Details: "frequency_penalty is not supported by Kronk",
		})
	}

	if call.PresencePenalty != nil {
		warnings = append(warnings, fantasy.CallWarning{
			Type:    fantasy.CallWarningTypeUnsupportedSetting,
			Setting: "presence_penalty",
			Details: "presence_penalty is not supported by Kronk",
		})
	}

	optionsWarnings, err := l.prepareCallFunc(l, d, call)
	if err != nil {
		return nil, nil, err
	}

	if len(optionsWarnings) > 0 {
		warnings = append(warnings, optionsWarnings...)
	}

	if len(call.Tools) > 0 {
		tools, toolWarnings := toKronkTools(call.Tools)
		d["tools"] = tools
		warnings = append(warnings, toolWarnings...)
	}

	return d, warnings, nil
}

// Generate implements fantasy.LanguageModel.
func (l *languageModel) Generate(ctx context.Context, call fantasy.Call) (*fantasy.Response, error) {
	d, warnings, err := l.prepareDocument(call)
	if err != nil {
		return nil, err
	}

	ch, err := l.kronk.ChatStreaming(ctx, d)
	if err != nil {
		return nil, toProviderErr(err)
	}

	var lastResponse model.ChatResponse
	var fullContent string

	for resp := range ch {
		lastResponse = resp

		if len(resp.Choice) > 0 && resp.Choice[0].Delta != nil {
			switch resp.Choice[0].FinishReason() {
			case model.FinishReasonError:
				return nil, &fantasy.Error{Title: "model error", Message: resp.Choice[0].Delta.Content}

			case model.FinishReasonStop, model.FinishReasonTool:
				// Final response already contains full accumulated content in Delta.Content,
				// so we use it directly instead of continuing to accumulate.
				fullContent = resp.Choice[0].Delta.Content

			default:
				fullContent += resp.Choice[0].Delta.Content
			}
		}
	}

	if len(lastResponse.Choice) == 0 {
		return nil, &fantasy.Error{Title: "no response", Message: "no response generated"}
	}

	choice := lastResponse.Choice[0]
	var content []fantasy.Content
	if choice.Delta != nil {
		content = make([]fantasy.Content, 0, 1+len(choice.Delta.ToolCalls))
	}

	if fullContent != "" {
		content = append(content, fantasy.TextContent{
			Text: fullContent,
		})
	}

	if choice.Delta != nil {
		for _, tc := range choice.Delta.ToolCalls {
			// Marshal the underlying map directly, not the ToolCallArguments type
			// which has a custom MarshalJSON that double-encodes to a JSON string.
			argsJSON, _ := json.Marshal(map[string]any(tc.Function.Arguments))

			content = append(content, fantasy.ToolCallContent{
				ProviderExecuted: false,
				ToolCallID:       tc.ID,
				ToolName:         tc.Function.Name,
				Input:            string(argsJSON),
			})
		}
	}

	usage := fantasy.Usage{}
	if lastResponse.Usage != nil {
		usage = fantasy.Usage{
			InputTokens:     int64(lastResponse.Usage.PromptTokens),
			OutputTokens:    int64(lastResponse.Usage.CompletionTokens),
			TotalTokens:     int64(lastResponse.Usage.PromptTokens + lastResponse.Usage.CompletionTokens),
			ReasoningTokens: int64(lastResponse.Usage.ReasoningTokens),
		}
	}

	mappedFinishReason := l.mapFinishReasonFunc(choice.FinishReason())
	if choice.Delta != nil && len(choice.Delta.ToolCalls) > 0 {
		mappedFinishReason = fantasy.FinishReasonToolCalls
	}

	providerMetadata := fantasy.ProviderMetadata{}
	if lastResponse.Usage != nil {
		providerMetadata = fantasy.ProviderMetadata{
			Name: &ProviderMetadata{
				TokensPerSecond: lastResponse.Usage.TokensPerSecond,
				OutputTokens:    int64(lastResponse.Usage.OutputTokens),
			},
		}
	}

	resp := fantasy.Response{
		Content:          content,
		Usage:            usage,
		FinishReason:     mappedFinishReason,
		ProviderMetadata: providerMetadata,
		Warnings:         warnings,
	}

	return &resp, nil
}

// Stream implements fantasy.LanguageModel.
func (l *languageModel) Stream(ctx context.Context, call fantasy.Call) (fantasy.StreamResponse, error) {
	d, warnings, err := l.prepareDocument(call)
	if err != nil {
		return nil, err
	}

	ch, err := l.kronk.ChatStreaming(ctx, d)
	if err != nil {
		return nil, toProviderErr(err)
	}

	isActiveText := false
	isActiveReasoning := false
	toolCalls := make(map[int]streamToolCall)

	providerMetadata := fantasy.ProviderMetadata{
		Name: &ProviderMetadata{},
	}

	var usage fantasy.Usage
	var finishReason string

	return func(yield func(fantasy.StreamPart) bool) {
		if len(warnings) > 0 {
			if !yield(fantasy.StreamPart{
				Type:     fantasy.StreamPartTypeWarnings,
				Warnings: warnings,
			}) {
				return
			}
		}

		toolIndex := 0
		for resp := range ch {
			if len(resp.Choice) == 0 {
				continue
			}

			choice := resp.Choice[0]
			if choice.Delta == nil {
				continue
			}

			if resp.Usage != nil {
				usage = fantasy.Usage{
					InputTokens:     int64(resp.Usage.PromptTokens),
					OutputTokens:    int64(resp.Usage.CompletionTokens),
					TotalTokens:     int64(resp.Usage.PromptTokens + resp.Usage.CompletionTokens),
					ReasoningTokens: int64(resp.Usage.ReasoningTokens),
				}

				if pm, ok := providerMetadata[Name]; ok {
					if metadata, ok := pm.(*ProviderMetadata); ok {
						metadata.TokensPerSecond = resp.Usage.TokensPerSecond
						metadata.OutputTokens = int64(resp.Usage.OutputTokens)
					}
				}
			}

			if choice.FinishReason() != "" {
				finishReason = choice.FinishReason()
			}

			switch choice.FinishReason() {
			case model.FinishReasonError:
				yield(fantasy.StreamPart{
					Type:  fantasy.StreamPartTypeError,
					Error: &fantasy.Error{Title: "model error", Message: choice.Delta.Content},
				})
				return

			case model.FinishReasonTool:
				if isActiveReasoning {
					isActiveReasoning = false
					if !yield(fantasy.StreamPart{
						Type: fantasy.StreamPartTypeReasoningEnd,
						ID:   "reasoning-0",
					}) {
						return
					}
				}

				if isActiveText {
					isActiveText = false
					if !yield(fantasy.StreamPart{
						Type: fantasy.StreamPartTypeTextEnd,
						ID:   "0",
					}) {
						return
					}
				}

				for _, tc := range choice.Delta.ToolCalls {
					argsJSON, _ := json.Marshal(map[string]any(tc.Function.Arguments))
					argsStr := string(argsJSON)

					toolID := tc.ID
					if toolID == "" {
						toolID = uuid.NewString()
					}

					if !yield(fantasy.StreamPart{
						Type:         fantasy.StreamPartTypeToolInputStart,
						ID:           toolID,
						ToolCallName: tc.Function.Name,
					}) {
						return
					}

					if !yield(fantasy.StreamPart{
						Type:  fantasy.StreamPartTypeToolInputDelta,
						ID:    toolID,
						Delta: argsStr,
					}) {
						return
					}

					if !yield(fantasy.StreamPart{
						Type: fantasy.StreamPartTypeToolInputEnd,
						ID:   toolID,
					}) {
						return
					}

					if !yield(fantasy.StreamPart{
						Type:          fantasy.StreamPartTypeToolCall,
						ID:            toolID,
						ToolCallName:  tc.Function.Name,
						ToolCallInput: argsStr,
					}) {
						return
					}

					toolCalls[toolIndex] = streamToolCall{
						id:          toolID,
						name:        tc.Function.Name,
						arguments:   argsStr,
						hasFinished: true,
					}
					toolIndex++
				}

			default:
				if choice.Delta.Reasoning != "" {
					if !isActiveReasoning {
						isActiveReasoning = true
						if !yield(fantasy.StreamPart{
							Type: fantasy.StreamPartTypeReasoningStart,
							ID:   "reasoning-0",
						}) {
							return
						}
					}

					if !yield(fantasy.StreamPart{
						Type:  fantasy.StreamPartTypeReasoningDelta,
						ID:    "reasoning-0",
						Delta: choice.Delta.Reasoning,
					}) {
						return
					}
				}

				hasToolCalls := len(choice.Delta.ToolCalls) > 0
				hasContent := choice.Delta.Content != ""

				if isActiveReasoning && (hasContent || hasToolCalls) {
					isActiveReasoning = false
					if !yield(fantasy.StreamPart{
						Type: fantasy.StreamPartTypeReasoningEnd,
						ID:   "reasoning-0",
					}) {
						return
					}
				}

				if hasContent {
					if !isActiveText {
						isActiveText = true
						if !yield(fantasy.StreamPart{
							Type: fantasy.StreamPartTypeTextStart,
							ID:   "0",
						}) {
							return
						}
					}

					if !yield(fantasy.StreamPart{
						Type:  fantasy.StreamPartTypeTextDelta,
						ID:    "0",
						Delta: choice.Delta.Content,
					}) {
						return
					}
				}

				if hasToolCalls && isActiveText {
					isActiveText = false
					if !yield(fantasy.StreamPart{
						Type: fantasy.StreamPartTypeTextEnd,
						ID:   "0",
					}) {
						return
					}
				}

				for _, tc := range choice.Delta.ToolCalls {
					argsJSON, _ := json.Marshal(map[string]any(tc.Function.Arguments))
					argsStr := string(argsJSON)

					switch existingTC, ok := toolCalls[toolIndex]; ok {
					case true:
						if existingTC.hasFinished {
							continue
						}

						existingTC.arguments += argsStr

						if !yield(fantasy.StreamPart{
							Type:  fantasy.StreamPartTypeToolInputDelta,
							ID:    existingTC.id,
							Delta: argsStr,
						}) {
							return
						}

						toolCalls[toolIndex] = existingTC

						if xjson.IsValid(existingTC.arguments) {
							if !yield(fantasy.StreamPart{
								Type: fantasy.StreamPartTypeToolInputEnd,
								ID:   existingTC.id,
							}) {
								return
							}

							if !yield(fantasy.StreamPart{
								Type:          fantasy.StreamPartTypeToolCall,
								ID:            existingTC.id,
								ToolCallName:  existingTC.name,
								ToolCallInput: existingTC.arguments,
							}) {
								return
							}

							existingTC.hasFinished = true
							toolCalls[toolIndex] = existingTC
						}

					case false:
						toolID := tc.ID
						if toolID == "" {
							toolID = uuid.NewString()
						}

						if !yield(fantasy.StreamPart{
							Type:         fantasy.StreamPartTypeToolInputStart,
							ID:           toolID,
							ToolCallName: tc.Function.Name,
						}) {
							return
						}

						toolCalls[toolIndex] = streamToolCall{
							id:        toolID,
							name:      tc.Function.Name,
							arguments: argsStr,
						}

						if argsStr != "" && argsStr != "null" {
							if !yield(fantasy.StreamPart{
								Type:  fantasy.StreamPartTypeToolInputDelta,
								ID:    toolID,
								Delta: argsStr,
							}) {
								return
							}

							if xjson.IsValid(argsStr) {
								if !yield(fantasy.StreamPart{
									Type: fantasy.StreamPartTypeToolInputEnd,
									ID:   toolID,
								}) {
									return
								}

								if !yield(fantasy.StreamPart{
									Type:          fantasy.StreamPartTypeToolCall,
									ID:            toolID,
									ToolCallName:  tc.Function.Name,
									ToolCallInput: argsStr,
								}) {
									return
								}

								stc := toolCalls[toolIndex]
								stc.hasFinished = true
								toolCalls[toolIndex] = stc
							}
						}

						toolIndex++
					}
				}
			}
		}

		if isActiveReasoning {
			if !yield(fantasy.StreamPart{
				Type: fantasy.StreamPartTypeReasoningEnd,
				ID:   "reasoning-0",
			}) {
				return
			}
		}

		if isActiveText {
			if !yield(fantasy.StreamPart{
				Type: fantasy.StreamPartTypeTextEnd,
				ID:   "0",
			}) {
				return
			}
		}

		mappedFinishReason := l.mapFinishReasonFunc(finishReason)
		if len(toolCalls) > 0 {
			mappedFinishReason = fantasy.FinishReasonToolCalls
		}

		yield(fantasy.StreamPart{
			Type:             fantasy.StreamPartTypeFinish,
			Usage:            usage,
			FinishReason:     mappedFinishReason,
			ProviderMetadata: providerMetadata,
		})
	}, nil
}

// GenerateObject implements fantasy.LanguageModel.
func (l *languageModel) GenerateObject(ctx context.Context, call fantasy.ObjectCall) (*fantasy.ObjectResponse, error) {
	switch l.objectMode {
	case fantasy.ObjectModeText:
		return object.GenerateWithText(ctx, l, call)

	case fantasy.ObjectModeTool:
		return object.GenerateWithTool(ctx, l, call)

	default:
		return object.GenerateWithTool(ctx, l, call)
	}
}

// StreamObject implements fantasy.LanguageModel.
func (l *languageModel) StreamObject(ctx context.Context, call fantasy.ObjectCall) (fantasy.ObjectStreamResponse, error) {
	switch l.objectMode {
	case fantasy.ObjectModeTool:
		return object.StreamWithTool(ctx, l, call)

	case fantasy.ObjectModeText:
		return object.StreamWithText(ctx, l, call)

	default:
		return object.StreamWithTool(ctx, l, call)
	}
}

func toKronkTools(tools []fantasy.Tool) ([]model.D, []fantasy.CallWarning) {
	var kronkTools []model.D
	var warnings []fantasy.CallWarning

	for _, tool := range tools {
		if tool.GetType() == fantasy.ToolTypeFunction {
			ft, ok := tool.(fantasy.FunctionTool)
			if !ok {
				continue
			}

			kronkTools = append(kronkTools, model.D{
				"type": "function",
				"function": model.D{
					"name":        ft.Name,
					"description": ft.Description,
					"parameters":  ft.InputSchema,
				},
			})

			continue
		}

		warnings = append(warnings, fantasy.CallWarning{
			Type:    fantasy.CallWarningTypeUnsupportedTool,
			Tool:    tool,
			Message: "tool is not supported",
		})
	}

	return kronkTools, warnings
}

func toProviderErr(err error) error {
	if err == nil {
		return nil
	}

	if errors.Is(err, io.EOF) {
		return nil
	}

	return &fantasy.ProviderError{
		Title:   "kronk error",
		Message: err.Error(),
		Cause:   err,
	}
}
