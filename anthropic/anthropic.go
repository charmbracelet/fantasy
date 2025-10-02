package anthropic

import (
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/anthropics/anthropic-sdk-go/vertex"
	"github.com/charmbracelet/fantasy/ai"
	"golang.org/x/oauth2/google"
)

const (
	Name       = "anthropic"
	DefaultURL = "https://api.anthropic.com"
)

type options struct {
	baseURL string
	apiKey  string
	name    string
	headers map[string]string
	client  option.HTTPClient

	vertexProject  string
	vertexLocation string
	skipGoogleAuth bool
}

type provider struct {
	options options
}

type Option = func(*options)

func New(opts ...Option) ai.Provider {
	providerOptions := options{
		headers: map[string]string{},
	}
	for _, o := range opts {
		o(&providerOptions)
	}

	providerOptions.baseURL = cmp.Or(providerOptions.baseURL, DefaultURL)
	providerOptions.name = cmp.Or(providerOptions.name, Name)
	return &provider{options: providerOptions}
}

func WithBaseURL(baseURL string) Option {
	return func(o *options) {
		o.baseURL = baseURL
	}
}

func WithAPIKey(apiKey string) Option {
	return func(o *options) {
		o.apiKey = apiKey
	}
}

func WithVertex(project, location string) Option {
	return func(o *options) {
		o.vertexProject = project
		o.vertexLocation = location
	}
}

func WithSkipGoogleAuth(skip bool) Option {
	return func(o *options) {
		o.skipGoogleAuth = skip
	}
}

func WithName(name string) Option {
	return func(o *options) {
		o.name = name
	}
}

func WithHeaders(headers map[string]string) Option {
	return func(o *options) {
		maps.Copy(o.headers, headers)
	}
}

func WithHTTPClient(client option.HTTPClient) Option {
	return func(o *options) {
		o.client = client
	}
}

func (a *provider) LanguageModel(modelID string) (ai.LanguageModel, error) {
	clientOptions := make([]option.RequestOption, 0, 5+len(a.options.headers))
	if a.options.apiKey != "" {
		clientOptions = append(clientOptions, option.WithAPIKey(a.options.apiKey))
	}
	if a.options.baseURL != "" {
		clientOptions = append(clientOptions, option.WithBaseURL(a.options.baseURL))
	}
	for key, value := range a.options.headers {
		clientOptions = append(clientOptions, option.WithHeader(key, value))
	}
	if a.options.client != nil {
		clientOptions = append(clientOptions, option.WithHTTPClient(a.options.client))
	}
	if a.options.vertexProject != "" && a.options.vertexLocation != "" {
		var credentials *google.Credentials
		if a.options.skipGoogleAuth {
			credentials = &google.Credentials{TokenSource: &googleDummyTokenSource{}}
		} else {
			var err error
			credentials, err = google.FindDefaultCredentials(context.Background())
			if err != nil {
				return nil, err
			}
		}

		clientOptions = append(
			clientOptions,
			vertex.WithCredentials(
				context.Background(),
				a.options.vertexLocation,
				a.options.vertexProject,
				credentials,
			),
		)
	}
	return languageModel{
		modelID:  modelID,
		provider: a.options.name,
		options:  a.options,
		client:   anthropic.NewClient(clientOptions...),
	}, nil
}

type languageModel struct {
	provider string
	modelID  string
	client   anthropic.Client
	options  options
}

// Model implements ai.LanguageModel.
func (a languageModel) Model() string {
	return a.modelID
}

// Provider implements ai.LanguageModel.
func (a languageModel) Provider() string {
	return a.provider
}

func (a languageModel) prepareParams(call ai.Call) (*anthropic.MessageNewParams, []ai.CallWarning, error) {
	params := &anthropic.MessageNewParams{}
	providerOptions := &ProviderOptions{}
	if v, ok := call.ProviderOptions[Name]; ok {
		providerOptions, ok = v.(*ProviderOptions)
		if !ok {
			return nil, nil, ai.NewInvalidArgumentError("providerOptions", "anthropic provider options should be *anthropic.ProviderOptions", nil)
		}
	}
	sendReasoning := true
	if providerOptions.SendReasoning != nil {
		sendReasoning = *providerOptions.SendReasoning
	}
	systemBlocks, messages, warnings := toPrompt(call.Prompt, sendReasoning)

	if call.FrequencyPenalty != nil {
		warnings = append(warnings, ai.CallWarning{
			Type:    ai.CallWarningTypeUnsupportedSetting,
			Setting: "FrequencyPenalty",
		})
	}
	if call.PresencePenalty != nil {
		warnings = append(warnings, ai.CallWarning{
			Type:    ai.CallWarningTypeUnsupportedSetting,
			Setting: "PresencePenalty",
		})
	}

	params.System = systemBlocks
	params.Messages = messages
	params.Model = anthropic.Model(a.modelID)
	params.MaxTokens = 4096

	if call.MaxOutputTokens != nil {
		params.MaxTokens = *call.MaxOutputTokens
	}

	if call.Temperature != nil {
		params.Temperature = param.NewOpt(*call.Temperature)
	}
	if call.TopK != nil {
		params.TopK = param.NewOpt(*call.TopK)
	}
	if call.TopP != nil {
		params.TopP = param.NewOpt(*call.TopP)
	}

	isThinking := false
	var thinkingBudget int64
	if providerOptions.Thinking != nil {
		isThinking = true
		thinkingBudget = providerOptions.Thinking.BudgetTokens
	}
	if isThinking {
		if thinkingBudget == 0 {
			return nil, nil, ai.NewUnsupportedFunctionalityError("thinking requires budget", "")
		}
		params.Thinking = anthropic.ThinkingConfigParamOfEnabled(thinkingBudget)
		if call.Temperature != nil {
			params.Temperature = param.Opt[float64]{}
			warnings = append(warnings, ai.CallWarning{
				Type:    ai.CallWarningTypeUnsupportedSetting,
				Setting: "temperature",
				Details: "temperature is not supported when thinking is enabled",
			})
		}
		if call.TopP != nil {
			params.TopP = param.Opt[float64]{}
			warnings = append(warnings, ai.CallWarning{
				Type:    ai.CallWarningTypeUnsupportedSetting,
				Setting: "TopP",
				Details: "TopP is not supported when thinking is enabled",
			})
		}
		if call.TopK != nil {
			params.TopK = param.Opt[int64]{}
			warnings = append(warnings, ai.CallWarning{
				Type:    ai.CallWarningTypeUnsupportedSetting,
				Setting: "TopK",
				Details: "TopK is not supported when thinking is enabled",
			})
		}
		params.MaxTokens = params.MaxTokens + thinkingBudget
	}

	if len(call.Tools) > 0 {
		disableParallelToolUse := false
		if providerOptions.DisableParallelToolUse != nil {
			disableParallelToolUse = *providerOptions.DisableParallelToolUse
		}
		tools, toolChoice, toolWarnings := toTools(call.Tools, call.ToolChoice, disableParallelToolUse)
		params.Tools = tools
		if toolChoice != nil {
			params.ToolChoice = *toolChoice
		}
		warnings = append(warnings, toolWarnings...)
	}

	return params, warnings, nil
}

func (a *provider) Name() string {
	return Name
}

func getCacheControl(providerOptions ai.ProviderOptions) *CacheControl {
	if anthropicOptions, ok := providerOptions[Name]; ok {
		if options, ok := anthropicOptions.(*ProviderCacheControlOptions); ok {
			return &options.CacheControl
		}
	}
	return nil
}

func getReasoningMetadata(providerOptions ai.ProviderOptions) *ReasoningOptionMetadata {
	if anthropicOptions, ok := providerOptions[Name]; ok {
		if reasoning, ok := anthropicOptions.(*ReasoningOptionMetadata); ok {
			return reasoning
		}
	}
	return nil
}

type messageBlock struct {
	Role     ai.MessageRole
	Messages []ai.Message
}

func groupIntoBlocks(prompt ai.Prompt) []*messageBlock {
	var blocks []*messageBlock

	var currentBlock *messageBlock

	for _, msg := range prompt {
		switch msg.Role {
		case ai.MessageRoleSystem:
			if currentBlock == nil || currentBlock.Role != ai.MessageRoleSystem {
				currentBlock = &messageBlock{
					Role:     ai.MessageRoleSystem,
					Messages: []ai.Message{},
				}
				blocks = append(blocks, currentBlock)
			}
			currentBlock.Messages = append(currentBlock.Messages, msg)
		case ai.MessageRoleUser:
			if currentBlock == nil || currentBlock.Role != ai.MessageRoleUser {
				currentBlock = &messageBlock{
					Role:     ai.MessageRoleUser,
					Messages: []ai.Message{},
				}
				blocks = append(blocks, currentBlock)
			}
			currentBlock.Messages = append(currentBlock.Messages, msg)
		case ai.MessageRoleAssistant:
			if currentBlock == nil || currentBlock.Role != ai.MessageRoleAssistant {
				currentBlock = &messageBlock{
					Role:     ai.MessageRoleAssistant,
					Messages: []ai.Message{},
				}
				blocks = append(blocks, currentBlock)
			}
			currentBlock.Messages = append(currentBlock.Messages, msg)
		case ai.MessageRoleTool:
			if currentBlock == nil || currentBlock.Role != ai.MessageRoleUser {
				currentBlock = &messageBlock{
					Role:     ai.MessageRoleUser,
					Messages: []ai.Message{},
				}
				blocks = append(blocks, currentBlock)
			}
			currentBlock.Messages = append(currentBlock.Messages, msg)
		}
	}
	return blocks
}

func toTools(tools []ai.Tool, toolChoice *ai.ToolChoice, disableParallelToolCalls bool) (anthropicTools []anthropic.ToolUnionParam, anthropicToolChoice *anthropic.ToolChoiceUnionParam, warnings []ai.CallWarning) {
	for _, tool := range tools {
		if tool.GetType() == ai.ToolTypeFunction {
			ft, ok := tool.(ai.FunctionTool)
			if !ok {
				continue
			}
			required := []string{}
			var properties any
			if props, ok := ft.InputSchema["properties"]; ok {
				properties = props
			}
			if req, ok := ft.InputSchema["required"]; ok {
				if reqArr, ok := req.([]string); ok {
					required = reqArr
				}
			}
			cacheControl := getCacheControl(ft.ProviderOptions)

			anthropicTool := anthropic.ToolParam{
				Name:        ft.Name,
				Description: anthropic.String(ft.Description),
				InputSchema: anthropic.ToolInputSchemaParam{
					Properties: properties,
					Required:   required,
				},
			}
			if cacheControl != nil {
				anthropicTool.CacheControl = anthropic.NewCacheControlEphemeralParam()
			}
			anthropicTools = append(anthropicTools, anthropic.ToolUnionParam{OfTool: &anthropicTool})
			continue
		}
		// TODO: handle provider tool calls
		warnings = append(warnings, ai.CallWarning{
			Type:    ai.CallWarningTypeUnsupportedTool,
			Tool:    tool,
			Message: "tool is not supported",
		})
	}
	if toolChoice == nil {
		if disableParallelToolCalls {
			anthropicToolChoice = &anthropic.ToolChoiceUnionParam{
				OfAuto: &anthropic.ToolChoiceAutoParam{
					Type:                   "auto",
					DisableParallelToolUse: param.NewOpt(disableParallelToolCalls),
				},
			}
		}
		return anthropicTools, anthropicToolChoice, warnings
	}

	switch *toolChoice {
	case ai.ToolChoiceAuto:
		anthropicToolChoice = &anthropic.ToolChoiceUnionParam{
			OfAuto: &anthropic.ToolChoiceAutoParam{
				Type:                   "auto",
				DisableParallelToolUse: param.NewOpt(disableParallelToolCalls),
			},
		}
	case ai.ToolChoiceRequired:
		anthropicToolChoice = &anthropic.ToolChoiceUnionParam{
			OfAny: &anthropic.ToolChoiceAnyParam{
				Type:                   "any",
				DisableParallelToolUse: param.NewOpt(disableParallelToolCalls),
			},
		}
	case ai.ToolChoiceNone:
		return anthropicTools, anthropicToolChoice, warnings
	default:
		anthropicToolChoice = &anthropic.ToolChoiceUnionParam{
			OfTool: &anthropic.ToolChoiceToolParam{
				Type:                   "tool",
				Name:                   string(*toolChoice),
				DisableParallelToolUse: param.NewOpt(disableParallelToolCalls),
			},
		}
	}
	return anthropicTools, anthropicToolChoice, warnings
}

func toPrompt(prompt ai.Prompt, sendReasoningData bool) ([]anthropic.TextBlockParam, []anthropic.MessageParam, []ai.CallWarning) {
	var systemBlocks []anthropic.TextBlockParam
	var messages []anthropic.MessageParam
	var warnings []ai.CallWarning

	blocks := groupIntoBlocks(prompt)
	finishedSystemBlock := false
	for _, block := range blocks {
		switch block.Role {
		case ai.MessageRoleSystem:
			if finishedSystemBlock {
				// skip multiple system messages that are separated by user/assistant messages
				// TODO: see if we need to send error here?
				continue
			}
			finishedSystemBlock = true
			for _, msg := range block.Messages {
				for i, part := range msg.Content {
					isLastPart := i == len(msg.Content)-1
					cacheControl := getCacheControl(part.Options())
					if cacheControl == nil && isLastPart {
						cacheControl = getCacheControl(msg.ProviderOptions)
					}
					text, ok := ai.AsMessagePart[ai.TextPart](part)
					if !ok {
						continue
					}
					textBlock := anthropic.TextBlockParam{
						Text: text.Text,
					}
					if cacheControl != nil {
						textBlock.CacheControl = anthropic.NewCacheControlEphemeralParam()
					}
					systemBlocks = append(systemBlocks, textBlock)
				}
			}

		case ai.MessageRoleUser:
			var anthropicContent []anthropic.ContentBlockParamUnion
			for _, msg := range block.Messages {
				if msg.Role == ai.MessageRoleUser {
					for i, part := range msg.Content {
						isLastPart := i == len(msg.Content)-1
						cacheControl := getCacheControl(part.Options())
						if cacheControl == nil && isLastPart {
							cacheControl = getCacheControl(msg.ProviderOptions)
						}
						switch part.GetType() {
						case ai.ContentTypeText:
							text, ok := ai.AsMessagePart[ai.TextPart](part)
							if !ok {
								continue
							}
							textBlock := &anthropic.TextBlockParam{
								Text: text.Text,
							}
							if cacheControl != nil {
								textBlock.CacheControl = anthropic.NewCacheControlEphemeralParam()
							}
							anthropicContent = append(anthropicContent, anthropic.ContentBlockParamUnion{
								OfText: textBlock,
							})
						case ai.ContentTypeFile:
							file, ok := ai.AsMessagePart[ai.FilePart](part)
							if !ok {
								continue
							}
							// TODO: handle other file types
							if !strings.HasPrefix(file.MediaType, "image/") {
								continue
							}

							base64Encoded := base64.StdEncoding.EncodeToString(file.Data)
							imageBlock := anthropic.NewImageBlockBase64(file.MediaType, base64Encoded)
							if cacheControl != nil {
								imageBlock.OfImage.CacheControl = anthropic.NewCacheControlEphemeralParam()
							}
							anthropicContent = append(anthropicContent, imageBlock)
						}
					}
				} else if msg.Role == ai.MessageRoleTool {
					for i, part := range msg.Content {
						isLastPart := i == len(msg.Content)-1
						cacheControl := getCacheControl(part.Options())
						if cacheControl == nil && isLastPart {
							cacheControl = getCacheControl(msg.ProviderOptions)
						}
						result, ok := ai.AsMessagePart[ai.ToolResultPart](part)
						if !ok {
							continue
						}
						toolResultBlock := anthropic.ToolResultBlockParam{
							ToolUseID: result.ToolCallID,
						}
						switch result.Output.GetType() {
						case ai.ToolResultContentTypeText:
							content, ok := ai.AsToolResultOutputType[ai.ToolResultOutputContentText](result.Output)
							if !ok {
								continue
							}
							toolResultBlock.Content = []anthropic.ToolResultBlockParamContentUnion{
								{
									OfText: &anthropic.TextBlockParam{
										Text: content.Text,
									},
								},
							}
						case ai.ToolResultContentTypeMedia:
							content, ok := ai.AsToolResultOutputType[ai.ToolResultOutputContentMedia](result.Output)
							if !ok {
								continue
							}
							toolResultBlock.Content = []anthropic.ToolResultBlockParamContentUnion{
								{
									OfImage: anthropic.NewImageBlockBase64(content.MediaType, content.Data).OfImage,
								},
							}
						case ai.ToolResultContentTypeError:
							content, ok := ai.AsToolResultOutputType[ai.ToolResultOutputContentError](result.Output)
							if !ok {
								continue
							}
							toolResultBlock.Content = []anthropic.ToolResultBlockParamContentUnion{
								{
									OfText: &anthropic.TextBlockParam{
										Text: content.Error.Error(),
									},
								},
							}
							toolResultBlock.IsError = param.NewOpt(true)
						}
						if cacheControl != nil {
							toolResultBlock.CacheControl = anthropic.NewCacheControlEphemeralParam()
						}
						anthropicContent = append(anthropicContent, anthropic.ContentBlockParamUnion{
							OfToolResult: &toolResultBlock,
						})
					}
				}
			}
			messages = append(messages, anthropic.NewUserMessage(anthropicContent...))
		case ai.MessageRoleAssistant:
			var anthropicContent []anthropic.ContentBlockParamUnion
			for _, msg := range block.Messages {
				for i, part := range msg.Content {
					isLastPart := i == len(msg.Content)-1
					cacheControl := getCacheControl(part.Options())
					if cacheControl == nil && isLastPart {
						cacheControl = getCacheControl(msg.ProviderOptions)
					}
					switch part.GetType() {
					case ai.ContentTypeText:
						text, ok := ai.AsMessagePart[ai.TextPart](part)
						if !ok {
							continue
						}
						textBlock := &anthropic.TextBlockParam{
							Text: text.Text,
						}
						if cacheControl != nil {
							textBlock.CacheControl = anthropic.NewCacheControlEphemeralParam()
						}
						anthropicContent = append(anthropicContent, anthropic.ContentBlockParamUnion{
							OfText: textBlock,
						})
					case ai.ContentTypeReasoning:
						reasoning, ok := ai.AsMessagePart[ai.ReasoningPart](part)
						if !ok {
							continue
						}
						if !sendReasoningData {
							warnings = append(warnings, ai.CallWarning{
								Type:    "other",
								Message: "sending reasoning content is disabled for this model",
							})
							continue
						}
						reasoningMetadata := getReasoningMetadata(part.Options())
						if reasoningMetadata == nil {
							warnings = append(warnings, ai.CallWarning{
								Type:    "other",
								Message: "unsupported reasoning metadata",
							})
							continue
						}

						if reasoningMetadata.Signature != "" {
							anthropicContent = append(anthropicContent, anthropic.NewThinkingBlock(reasoningMetadata.Signature, reasoning.Text))
						} else if reasoningMetadata.RedactedData != "" {
							anthropicContent = append(anthropicContent, anthropic.NewRedactedThinkingBlock(reasoningMetadata.RedactedData))
						} else {
							warnings = append(warnings, ai.CallWarning{
								Type:    "other",
								Message: "unsupported reasoning metadata",
							})
							continue
						}
					case ai.ContentTypeToolCall:
						toolCall, ok := ai.AsMessagePart[ai.ToolCallPart](part)
						if !ok {
							continue
						}
						if toolCall.ProviderExecuted {
							// TODO: implement provider executed call
							continue
						}

						var inputMap map[string]any
						err := json.Unmarshal([]byte(toolCall.Input), &inputMap)
						if err != nil {
							continue
						}
						toolUseBlock := anthropic.NewToolUseBlock(toolCall.ToolCallID, inputMap, toolCall.ToolName)
						if cacheControl != nil {
							toolUseBlock.OfToolUse.CacheControl = anthropic.NewCacheControlEphemeralParam()
						}
						anthropicContent = append(anthropicContent, toolUseBlock)
					case ai.ContentTypeToolResult:
						// TODO: implement provider executed tool result
					}
				}
			}
			messages = append(messages, anthropic.NewAssistantMessage(anthropicContent...))
		}
	}
	return systemBlocks, messages, warnings
}

func (o languageModel) handleError(err error) error {
	var apiErr *anthropic.Error
	if errors.As(err, &apiErr) {
		requestDump := apiErr.DumpRequest(true)
		responseDump := apiErr.DumpResponse(true)
		headers := map[string]string{}
		for k, h := range apiErr.Response.Header {
			v := h[len(h)-1]
			headers[strings.ToLower(k)] = v
		}
		return ai.NewAPICallError(
			apiErr.Error(),
			apiErr.Request.URL.String(),
			string(requestDump),
			apiErr.StatusCode,
			headers,
			string(responseDump),
			apiErr,
			false,
		)
	}
	return err
}

func mapFinishReason(finishReason string) ai.FinishReason {
	switch finishReason {
	case "end_turn", "pause_turn", "stop_sequence":
		return ai.FinishReasonStop
	case "max_tokens":
		return ai.FinishReasonLength
	case "tool_use":
		return ai.FinishReasonToolCalls
	default:
		return ai.FinishReasonUnknown
	}
}

// Generate implements ai.LanguageModel.
func (a languageModel) Generate(ctx context.Context, call ai.Call) (*ai.Response, error) {
	params, warnings, err := a.prepareParams(call)
	if err != nil {
		return nil, err
	}
	response, err := a.client.Messages.New(ctx, *params)
	if err != nil {
		return nil, a.handleError(err)
	}

	var content []ai.Content
	for _, block := range response.Content {
		switch block.Type {
		case "text":
			text, ok := block.AsAny().(anthropic.TextBlock)
			if !ok {
				continue
			}
			content = append(content, ai.TextContent{
				Text: text.Text,
			})
		case "thinking":
			reasoning, ok := block.AsAny().(anthropic.ThinkingBlock)
			if !ok {
				continue
			}
			content = append(content, ai.ReasoningContent{
				Text: reasoning.Thinking,
				ProviderMetadata: ai.ProviderMetadata{
					Name: &ReasoningOptionMetadata{
						Signature: reasoning.Signature,
					},
				},
			})
		case "redacted_thinking":
			reasoning, ok := block.AsAny().(anthropic.RedactedThinkingBlock)
			if !ok {
				continue
			}
			content = append(content, ai.ReasoningContent{
				Text: "",
				ProviderMetadata: ai.ProviderMetadata{
					Name: &ReasoningOptionMetadata{
						RedactedData: reasoning.Data,
					},
				},
			})
		case "tool_use":
			toolUse, ok := block.AsAny().(anthropic.ToolUseBlock)
			if !ok {
				continue
			}
			content = append(content, ai.ToolCallContent{
				ToolCallID:       toolUse.ID,
				ToolName:         toolUse.Name,
				Input:            string(toolUse.Input),
				ProviderExecuted: false,
			})
		}
	}

	return &ai.Response{
		Content: content,
		Usage: ai.Usage{
			InputTokens:         response.Usage.InputTokens,
			OutputTokens:        response.Usage.OutputTokens,
			TotalTokens:         response.Usage.InputTokens + response.Usage.OutputTokens,
			CacheCreationTokens: response.Usage.CacheCreationInputTokens,
			CacheReadTokens:     response.Usage.CacheReadInputTokens,
		},
		FinishReason:     mapFinishReason(string(response.StopReason)),
		ProviderMetadata: ai.ProviderMetadata{},
		Warnings:         warnings,
	}, nil
}

// Stream implements ai.LanguageModel.
func (a languageModel) Stream(ctx context.Context, call ai.Call) (ai.StreamResponse, error) {
	params, warnings, err := a.prepareParams(call)
	if err != nil {
		return nil, err
	}

	stream := a.client.Messages.NewStreaming(ctx, *params)
	acc := anthropic.Message{}
	return func(yield func(ai.StreamPart) bool) {
		if len(warnings) > 0 {
			if !yield(ai.StreamPart{
				Type:     ai.StreamPartTypeWarnings,
				Warnings: warnings,
			}) {
				return
			}
		}

		for stream.Next() {
			chunk := stream.Current()
			_ = acc.Accumulate(chunk)
			switch chunk.Type {
			case "content_block_start":
				contentBlockType := chunk.ContentBlock.Type
				switch contentBlockType {
				case "text":
					if !yield(ai.StreamPart{
						Type: ai.StreamPartTypeTextStart,
						ID:   fmt.Sprintf("%d", chunk.Index),
					}) {
						return
					}
				case "thinking":
					if !yield(ai.StreamPart{
						Type: ai.StreamPartTypeReasoningStart,
						ID:   fmt.Sprintf("%d", chunk.Index),
					}) {
						return
					}
				case "redacted_thinking":
					if !yield(ai.StreamPart{
						Type: ai.StreamPartTypeReasoningStart,
						ID:   fmt.Sprintf("%d", chunk.Index),
						ProviderMetadata: ai.ProviderMetadata{
							Name: &ReasoningOptionMetadata{
								RedactedData: chunk.ContentBlock.Data,
							},
						},
					}) {
						return
					}
				case "tool_use":
					if !yield(ai.StreamPart{
						Type:          ai.StreamPartTypeToolInputStart,
						ID:            chunk.ContentBlock.ID,
						ToolCallName:  chunk.ContentBlock.Name,
						ToolCallInput: "",
					}) {
						return
					}
				}
			case "content_block_stop":
				if len(acc.Content)-1 < int(chunk.Index) {
					continue
				}
				contentBlock := acc.Content[int(chunk.Index)]
				switch contentBlock.Type {
				case "text":
					if !yield(ai.StreamPart{
						Type: ai.StreamPartTypeTextEnd,
						ID:   fmt.Sprintf("%d", chunk.Index),
					}) {
						return
					}
				case "thinking":
					if !yield(ai.StreamPart{
						Type: ai.StreamPartTypeReasoningEnd,
						ID:   fmt.Sprintf("%d", chunk.Index),
					}) {
						return
					}
				case "tool_use":
					if !yield(ai.StreamPart{
						Type: ai.StreamPartTypeToolInputEnd,
						ID:   contentBlock.ID,
					}) {
						return
					}
					if !yield(ai.StreamPart{
						Type:          ai.StreamPartTypeToolCall,
						ID:            contentBlock.ID,
						ToolCallName:  contentBlock.Name,
						ToolCallInput: string(contentBlock.Input),
					}) {
						return
					}
				}
			case "content_block_delta":
				switch chunk.Delta.Type {
				case "text_delta":
					if !yield(ai.StreamPart{
						Type:  ai.StreamPartTypeTextDelta,
						ID:    fmt.Sprintf("%d", chunk.Index),
						Delta: chunk.Delta.Text,
					}) {
						return
					}
				case "thinking_delta":
					if !yield(ai.StreamPart{
						Type:  ai.StreamPartTypeReasoningDelta,
						ID:    fmt.Sprintf("%d", chunk.Index),
						Delta: chunk.Delta.Thinking,
					}) {
						return
					}
				case "signature_delta":
					if !yield(ai.StreamPart{
						Type: ai.StreamPartTypeReasoningDelta,
						ID:   fmt.Sprintf("%d", chunk.Index),
						ProviderMetadata: ai.ProviderMetadata{
							Name: &ReasoningOptionMetadata{
								Signature: chunk.Delta.Signature,
							},
						},
					}) {
						return
					}
				case "input_json_delta":
					if len(acc.Content)-1 < int(chunk.Index) {
						continue
					}
					contentBlock := acc.Content[int(chunk.Index)]
					if !yield(ai.StreamPart{
						Type:          ai.StreamPartTypeToolInputDelta,
						ID:            contentBlock.ID,
						ToolCallInput: chunk.Delta.PartialJSON,
					}) {
						return
					}
				}
			case "message_stop":
			}
		}

		err := stream.Err()
		if err == nil || errors.Is(err, io.EOF) {
			yield(ai.StreamPart{
				Type:         ai.StreamPartTypeFinish,
				ID:           acc.ID,
				FinishReason: mapFinishReason(string(acc.StopReason)),
				Usage: ai.Usage{
					InputTokens:         acc.Usage.InputTokens,
					OutputTokens:        acc.Usage.OutputTokens,
					TotalTokens:         acc.Usage.InputTokens + acc.Usage.OutputTokens,
					CacheCreationTokens: acc.Usage.CacheCreationInputTokens,
					CacheReadTokens:     acc.Usage.CacheReadInputTokens,
				},
				ProviderMetadata: ai.ProviderMetadata{},
			})
			return
		} else {
			yield(ai.StreamPart{
				Type:  ai.StreamPartTypeError,
				Error: a.handleError(err),
			})
			return
		}
	}, nil
}
