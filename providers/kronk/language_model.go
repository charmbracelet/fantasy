package kronk

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"

	"charm.land/fantasy"
	"charm.land/fantasy/object"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
	xjson "github.com/charmbracelet/x/json"
	"github.com/google/uuid"
)

type languageModel struct {
	provider            string
	modelID             string
	kronk               chatModel
	objectMode          fantasy.ObjectMode
	prepareCallFunc     LanguageModelPrepareCallFunc
	mapFinishReasonFunc LanguageModelMapFinishReasonFunc
	toPromptFunc        LanguageModelToPromptFunc
}

type chatModel interface {
	Chat(context.Context, model.D) (model.ChatResponse, error)
	ChatStreaming(context.Context, model.D) (<-chan model.ChatResponse, error)
	ModelInfo() model.ModelInfo
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

func newLanguageModel(modelID string, provider string, krn chatModel, opts ...LanguageModelOption) *languageModel {
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

	if call.TopK != nil {
		d["top_k"] = *call.TopK
	}

	if call.FrequencyPenalty != nil {
		d["frequency_penalty"] = *call.FrequencyPenalty
	}

	if call.PresencePenalty != nil {
		d["presence_penalty"] = *call.PresencePenalty
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

	if call.ToolChoice != nil {
		switch *call.ToolChoice {
		case fantasy.ToolChoiceNone, fantasy.ToolChoiceAuto, fantasy.ToolChoiceRequired:
			d["tool_choice"] = string(*call.ToolChoice)

		default:
			d["tool_choice"] = model.D{
				"type": "function",
				"function": model.D{
					"name": string(*call.ToolChoice),
				},
			}
		}
	}

	return d, warnings, nil
}

// Generate implements fantasy.LanguageModel.
func (l *languageModel) Generate(ctx context.Context, call fantasy.Call) (*fantasy.Response, error) {
	d, warnings, err := l.prepareDocument(call)
	if err != nil {
		return nil, err
	}

	response, err := l.kronk.Chat(ctx, d)
	if err != nil {
		return nil, toProviderErr(err)
	}

	if len(response.Choices) == 0 {
		return nil, &fantasy.Error{Title: "no response", Message: "no response generated"}
	}

	choice := response.Choices[0]
	var content []fantasy.Content
	if choice.Message != nil {
		content = make([]fantasy.Content, 0, 1+len(choice.Message.ToolCalls))
	}

	if choice.Message != nil && choice.Message.Content != "" {
		content = append(content, fantasy.TextContent{
			Text: choice.Message.Content,
		})
	}

	if choice.Message != nil {
		for _, tc := range choice.Message.ToolCalls {
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
	if response.Usage != nil {
		usage = fantasy.Usage{
			InputTokens:     int64(response.Usage.PromptTokens),
			OutputTokens:    int64(response.Usage.CompletionTokens),
			TotalTokens:     int64(response.Usage.PromptTokens + response.Usage.CompletionTokens),
			ReasoningTokens: int64(response.Usage.CompletionTokensDetails.ReasoningTokens),
			CacheReadTokens: int64(response.Usage.PromptTokensDetails.CachedTokens),
		}
	}

	mappedFinishReason := l.mapFinishReasonFunc(choice.FinishReason())
	if choice.Message != nil && len(choice.Message.ToolCalls) > 0 {
		mappedFinishReason = fantasy.FinishReasonToolCalls
	}

	metadata := newProviderMetadata(l.kronk.ModelInfo())
	metadata.update(response)

	resp := fantasy.Response{
		Content:          content,
		Usage:            usage,
		FinishReason:     mappedFinishReason,
		ProviderMetadata: fantasy.ProviderMetadata{Name: metadata},
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
	d["stream_options"] = model.D{"include_usage": true}

	ch, err := l.kronk.ChatStreaming(ctx, d)
	if err != nil {
		return nil, toProviderErr(err)
	}

	isActiveText := false
	isActiveReasoning := false
	toolCalls := make(map[int]streamToolCall)

	metadata := newProviderMetadata(l.kronk.ModelInfo())
	providerMetadata := fantasy.ProviderMetadata{Name: metadata}

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

		for resp := range ch {
			metadata.update(resp)

			if resp.Usage != nil {
				usage = fantasy.Usage{
					InputTokens:     int64(resp.Usage.PromptTokens),
					OutputTokens:    int64(resp.Usage.CompletionTokens),
					TotalTokens:     int64(resp.Usage.PromptTokens + resp.Usage.CompletionTokens),
					ReasoningTokens: int64(resp.Usage.CompletionTokensDetails.ReasoningTokens),
					CacheReadTokens: int64(resp.Usage.PromptTokensDetails.CachedTokens),
				}
			}

			if len(resp.Choices) == 0 {
				continue
			}

			choice := resp.Choices[0]
			if choice.FinishReason() != "" {
				finishReason = choice.FinishReason()
			}

			if choice.FinishReason() == model.FinishReasonError {
				err := ctx.Err()
				if err == nil {
					message := ""
					if choice.Delta != nil {
						message = choice.Delta.Content
					}
					err = toProviderErr(errors.New(cmp.Or(message, "model stopped with an error")))
				}
				yield(fantasy.StreamPart{
					Type:  fantasy.StreamPartTypeError,
					Error: err,
				})
				return
			}

			if choice.Delta == nil {
				continue
			}

			switch choice.FinishReason() {
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

				hasToolCalls := len(choice.Delta.ToolCallDeltas) > 0
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

				for _, tc := range choice.Delta.ToolCallDeltas {
					toolCall, ok := toolCalls[tc.Index]
					if !ok {
						toolID := tc.ID
						if toolID == "" {
							toolID = uuid.NewString()
						}

						toolCall = streamToolCall{
							id:   toolID,
							name: tc.Function.Name,
						}
						if !yield(fantasy.StreamPart{
							Type:         fantasy.StreamPartTypeToolInputStart,
							ID:           toolCall.id,
							ToolCallName: toolCall.name,
						}) {
							return
						}
					}

					if toolCall.hasFinished {
						continue
					}

					if tc.Function.Name != "" {
						toolCall.name = tc.Function.Name
					}
					if tc.Function.Arguments != "" {
						toolCall.arguments += tc.Function.Arguments
						if !yield(fantasy.StreamPart{
							Type:  fantasy.StreamPartTypeToolInputDelta,
							ID:    toolCall.id,
							Delta: tc.Function.Arguments,
						}) {
							return
						}
					}

					if xjson.IsValid(toolCall.arguments) {
						if !yield(fantasy.StreamPart{
							Type: fantasy.StreamPartTypeToolInputEnd,
							ID:   toolCall.id,
						}) {
							return
						}

						if !yield(fantasy.StreamPart{
							Type:          fantasy.StreamPartTypeToolCall,
							ID:            toolCall.id,
							ToolCallName:  toolCall.name,
							ToolCallInput: toolCall.arguments,
						}) {
							return
						}
						toolCall.hasFinished = true
					}

					toolCalls[tc.Index] = toolCall
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

		if finishReason == "" {
			err := ctx.Err()
			if err == nil {
				err = fantasy.NewIncompleteStreamError()
			}
			yield(fantasy.StreamPart{
				Type:  fantasy.StreamPartTypeError,
				Error: err,
			})
			return
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
