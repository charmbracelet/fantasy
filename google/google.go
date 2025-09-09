package google

import (
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"strings"

	"github.com/charmbracelet/fantasy/ai"
	"github.com/charmbracelet/x/exp/slice"
	"github.com/google/uuid"
	"google.golang.org/genai"
)

type provider struct {
	options options
}

type options struct {
	apiKey  string
	name    string
	headers map[string]string
	client  *http.Client
}

type Option = func(*options)

func New(opts ...Option) ai.Provider {
	options := options{
		headers: map[string]string{},
	}
	for _, o := range opts {
		o(&options)
	}

	if options.name == "" {
		options.name = "google"
	}

	return &provider{
		options: options,
	}
}

func WithAPIKey(apiKey string) Option {
	return func(o *options) {
		o.apiKey = apiKey
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

func WithHTTPClient(client *http.Client) Option {
	return func(o *options) {
		o.client = client
	}
}

type languageModel struct {
	provider        string
	modelID         string
	client          *genai.Client
	providerOptions options
}

// LanguageModel implements ai.Provider.
func (g *provider) LanguageModel(modelID string) (ai.LanguageModel, error) {
	cc := &genai.ClientConfig{
		APIKey:     g.options.apiKey,
		Backend:    genai.BackendGeminiAPI,
		HTTPClient: g.options.client,
	}
	client, err := genai.NewClient(context.Background(), cc)
	if err != nil {
		return nil, err
	}
	return &languageModel{
		modelID:         modelID,
		provider:        fmt.Sprintf("%s.generative-ai", g.options.name),
		providerOptions: g.options,
		client:          client,
	}, nil
}

func (a languageModel) prepareParams(call ai.Call) (*genai.GenerateContentConfig, []*genai.Content, []ai.CallWarning, error) {
	config := &genai.GenerateContentConfig{}
	providerOptions := &providerOptions{}
	if v, ok := call.ProviderOptions["google"]; ok {
		err := ai.ParseOptions(v, providerOptions)
		if err != nil {
			return nil, nil, nil, err
		}
	}

	systemInstructions, content, warnings := toGooglePrompt(call.Prompt)

	if providerOptions.ThinkingConfig != nil &&
		providerOptions.ThinkingConfig.IncludeThoughts != nil &&
		*providerOptions.ThinkingConfig.IncludeThoughts &&
		strings.HasPrefix(a.provider, "google.vertex.") {
		warnings = append(warnings, ai.CallWarning{
			Type: ai.CallWarningTypeOther,
			Message: "The 'includeThoughts' option is only supported with the Google Vertex provider " +
				"and might not be supported or could behave unexpectedly with the current Google provider " +
				fmt.Sprintf("(%s)", a.provider),
		})
	}

	isGemmaModel := strings.HasPrefix(strings.ToLower(a.modelID), "gemma-")

	if isGemmaModel && systemInstructions != nil && len(systemInstructions.Parts) > 0 {
		if len(content) > 0 && content[0].Role == genai.RoleUser {
			systemParts := []string{}
			for _, sp := range systemInstructions.Parts {
				systemParts = append(systemParts, sp.Text)
			}
			systemMsg := strings.Join(systemParts, "\n")
			content[0].Parts = append([]*genai.Part{
				{
					Text: systemMsg + "\n\n",
				},
			}, content[0].Parts...)
			systemInstructions = nil
		}
	}

	config.SystemInstruction = systemInstructions

	if call.MaxOutputTokens != nil {
		config.MaxOutputTokens = int32(*call.MaxOutputTokens)
	}

	if call.Temperature != nil {
		tmp := float32(*call.Temperature)
		config.Temperature = &tmp
	}
	if call.TopK != nil {
		tmp := float32(*call.TopK)
		config.TopK = &tmp
	}
	if call.TopP != nil {
		tmp := float32(*call.TopP)
		config.TopP = &tmp
	}
	if call.FrequencyPenalty != nil {
		tmp := float32(*call.FrequencyPenalty)
		config.FrequencyPenalty = &tmp
	}
	if call.PresencePenalty != nil {
		tmp := float32(*call.PresencePenalty)
		config.PresencePenalty = &tmp
	}

	if providerOptions.ThinkingConfig != nil {
		config.ThinkingConfig = &genai.ThinkingConfig{}
		if providerOptions.ThinkingConfig.IncludeThoughts != nil {
			config.ThinkingConfig.IncludeThoughts = *providerOptions.ThinkingConfig.IncludeThoughts
		}
		if providerOptions.ThinkingConfig.ThinkingBudget != nil {
			tmp := int32(*providerOptions.ThinkingConfig.ThinkingBudget)
			config.ThinkingConfig.ThinkingBudget = &tmp
		}
	}
	for _, safetySetting := range providerOptions.SafetySettings {
		config.SafetySettings = append(config.SafetySettings, &genai.SafetySetting{
			Category:  genai.HarmCategory(safetySetting.Category),
			Threshold: genai.HarmBlockThreshold(safetySetting.Threshold),
		})
	}
	if providerOptions.CachedContent != "" {
		config.CachedContent = providerOptions.CachedContent
	}

	if len(call.Tools) > 0 {
		tools, toolChoice, toolWarnings := toGoogleTools(call.Tools, call.ToolChoice)
		config.ToolConfig = toolChoice
		config.Tools = append(config.Tools, &genai.Tool{
			FunctionDeclarations: tools,
		})
		warnings = append(warnings, toolWarnings...)
	}

	return config, content, warnings, nil
}

func toGooglePrompt(prompt ai.Prompt) (*genai.Content, []*genai.Content, []ai.CallWarning) {
	var systemInstructions *genai.Content
	var content []*genai.Content
	var warnings []ai.CallWarning

	finishedSystemBlock := false
	for _, msg := range prompt {
		switch msg.Role {
		case ai.MessageRoleSystem:
			if finishedSystemBlock {
				// skip multiple system messages that are separated by user/assistant messages
				// TODO: see if we need to send error here?
				continue
			}
			finishedSystemBlock = true

			var systemMessages []string
			for _, part := range msg.Content {
				text, ok := ai.AsMessagePart[ai.TextPart](part)
				if !ok || text.Text == "" {
					continue
				}
				systemMessages = append(systemMessages, text.Text)
			}
			if len(systemMessages) > 0 {
				systemInstructions = &genai.Content{
					Parts: []*genai.Part{
						{
							Text: strings.Join(systemMessages, "\n"),
						},
					},
				}
			}
		case ai.MessageRoleUser:
			var parts []*genai.Part
			for _, part := range msg.Content {
				switch part.GetType() {
				case ai.ContentTypeText:
					text, ok := ai.AsMessagePart[ai.TextPart](part)
					if !ok || text.Text == "" {
						continue
					}
					parts = append(parts, &genai.Part{
						Text: text.Text,
					})
				case ai.ContentTypeFile:
					file, ok := ai.AsMessagePart[ai.FilePart](part)
					if !ok {
						continue
					}
					var encoded []byte
					base64.StdEncoding.Encode(encoded, file.Data)
					parts = append(parts, &genai.Part{
						InlineData: &genai.Blob{
							Data:     encoded,
							MIMEType: file.MediaType,
						},
					})
				}
			}
			if len(parts) > 0 {
				content = append(content, &genai.Content{
					Role:  genai.RoleUser,
					Parts: parts,
				})
			}
		case ai.MessageRoleAssistant:
			var parts []*genai.Part
			for _, part := range msg.Content {
				switch part.GetType() {
				case ai.ContentTypeText:
					text, ok := ai.AsMessagePart[ai.TextPart](part)
					if !ok || text.Text == "" {
						continue
					}
					parts = append(parts, &genai.Part{
						Text: text.Text,
					})
				case ai.ContentTypeToolCall:
					toolCall, ok := ai.AsMessagePart[ai.ToolCallPart](part)
					if !ok {
						continue
					}

					var result map[string]any
					err := json.Unmarshal([]byte(toolCall.Input), &result)
					if err != nil {
						continue
					}
					parts = append(parts, &genai.Part{
						FunctionCall: &genai.FunctionCall{
							ID:   toolCall.ToolCallID,
							Name: toolCall.ToolName,
							Args: result,
						},
					})
				}
			}
			if len(parts) > 0 {
				content = append(content, &genai.Content{
					Role:  genai.RoleModel,
					Parts: parts,
				})
			}
		case ai.MessageRoleTool:
			var parts []*genai.Part
			for _, part := range msg.Content {
				switch part.GetType() {
				case ai.ContentTypeToolResult:
					result, ok := ai.AsMessagePart[ai.ToolResultPart](part)
					if !ok {
						continue
					}
					var toolCall ai.ToolCallPart
					for _, m := range prompt {
						if m.Role == ai.MessageRoleAssistant {
							for _, content := range m.Content {
								tc, ok := ai.AsMessagePart[ai.ToolCallPart](content)
								if !ok {
									continue
								}
								if tc.ToolCallID == result.ToolCallID {
									toolCall = tc
									break
								}
							}
						}
					}
					switch result.Output.GetType() {
					case ai.ToolResultContentTypeText:
						content, ok := ai.AsToolResultOutputType[ai.ToolResultOutputContentText](result.Output)
						if !ok {
							continue
						}
						response := map[string]any{"result": content.Text}
						parts = append(parts, &genai.Part{
							FunctionResponse: &genai.FunctionResponse{
								ID:       result.ToolCallID,
								Response: response,
								Name:     toolCall.ToolName,
							},
						})

					case ai.ToolResultContentTypeError:
						content, ok := ai.AsToolResultOutputType[ai.ToolResultOutputContentError](result.Output)
						if !ok {
							continue
						}
						response := map[string]any{"result": content.Error.Error()}
						parts = append(parts, &genai.Part{
							FunctionResponse: &genai.FunctionResponse{
								ID:       result.ToolCallID,
								Response: response,
								Name:     toolCall.ToolName,
							},
						})

					}
				}
			}
			if len(parts) > 0 {
				content = append(content, &genai.Content{
					Role:  genai.RoleUser,
					Parts: parts,
				})
			}
		}
	}
	return systemInstructions, content, warnings
}

// Generate implements ai.LanguageModel.
func (g *languageModel) Generate(ctx context.Context, call ai.Call) (*ai.Response, error) {
	config, contents, warnings, err := g.prepareParams(call)
	if err != nil {
		return nil, err
	}

	lastMessage, history, ok := slice.Pop(contents)
	if !ok {
		return nil, errors.New("no messages to send")
	}

	chat, err := g.client.Chats.Create(ctx, g.modelID, config, history)
	if err != nil {
		return nil, err
	}

	response, err := chat.SendMessage(ctx, depointerSlice(lastMessage.Parts)...)
	if err != nil {
		return nil, err
	}

	return mapResponse(response, warnings)
}

// Model implements ai.LanguageModel.
func (g *languageModel) Model() string {
	return g.modelID
}

// Provider implements ai.LanguageModel.
func (g *languageModel) Provider() string {
	return g.provider
}

// Stream implements ai.LanguageModel.
func (g *languageModel) Stream(ctx context.Context, call ai.Call) (ai.StreamResponse, error) {
	config, contents, warnings, err := g.prepareParams(call)
	if err != nil {
		return nil, err
	}

	lastMessage, history, ok := slice.Pop(contents)
	if !ok {
		return nil, errors.New("no messages to send")
	}

	chat, err := g.client.Chats.Create(ctx, g.modelID, config, history)
	if err != nil {
		return nil, err
	}

	return func(yield func(ai.StreamPart) bool) {
		if len(warnings) > 0 {
			if !yield(ai.StreamPart{
				Type:     ai.StreamPartTypeWarnings,
				Warnings: warnings,
			}) {
				return
			}
		}

		var currentContent string
		var toolCalls []ai.ToolCallContent
		var isActiveText bool
		var usage ai.Usage

		// Stream the response
		for resp, err := range chat.SendMessageStream(ctx, depointerSlice(lastMessage.Parts)...) {
			if err != nil {
				yield(ai.StreamPart{
					Type:  ai.StreamPartTypeError,
					Error: err,
				})
				return
			}

			if len(resp.Candidates) > 0 && resp.Candidates[0].Content != nil {
				for _, part := range resp.Candidates[0].Content.Parts {
					switch {
					case part.Text != "":
						delta := part.Text
						if delta != "" {
							if !isActiveText {
								isActiveText = true
								if !yield(ai.StreamPart{
									Type: ai.StreamPartTypeTextStart,
									ID:   "0",
								}) {
									return
								}
							}
							if !yield(ai.StreamPart{
								Type:  ai.StreamPartTypeTextDelta,
								ID:    "0",
								Delta: delta,
							}) {
								return
							}
							currentContent += delta
						}
					case part.FunctionCall != nil:
						if isActiveText {
							isActiveText = false
							if !yield(ai.StreamPart{
								Type: ai.StreamPartTypeTextEnd,
								ID:   "0",
							}) {
								return
							}
						}

						toolCallID := cmp.Or(part.FunctionCall.ID, part.FunctionCall.Name, uuid.NewString())

						args, err := json.Marshal(part.FunctionCall.Args)
						if err != nil {
							yield(ai.StreamPart{
								Type:  ai.StreamPartTypeError,
								Error: err,
							})
							return
						}

						if !yield(ai.StreamPart{
							Type:         ai.StreamPartTypeToolInputStart,
							ID:           toolCallID,
							ToolCallName: part.FunctionCall.Name,
						}) {
							return
						}

						if !yield(ai.StreamPart{
							Type:  ai.StreamPartTypeToolInputDelta,
							ID:    toolCallID,
							Delta: string(args),
						}) {
							return
						}

						if !yield(ai.StreamPart{
							Type: ai.StreamPartTypeToolInputEnd,
							ID:   toolCallID,
						}) {
							return
						}

						if !yield(ai.StreamPart{
							Type:             ai.StreamPartTypeToolCall,
							ID:               toolCallID,
							ToolCallName:     part.FunctionCall.Name,
							ToolCallInput:    string(args),
							ProviderExecuted: false,
						}) {
							return
						}

						toolCalls = append(toolCalls, ai.ToolCallContent{
							ToolCallID:       toolCallID,
							ToolName:         part.FunctionCall.Name,
							Input:            string(args),
							ProviderExecuted: false,
						})
					}
				}
			}

			if resp.UsageMetadata != nil {
				usage = mapUsage(resp.UsageMetadata)
			}
		}

		if isActiveText {
			if !yield(ai.StreamPart{
				Type: ai.StreamPartTypeTextEnd,
				ID:   "0",
			}) {
				return
			}
		}

		finishReason := ai.FinishReasonStop
		if len(toolCalls) > 0 {
			finishReason = ai.FinishReasonToolCalls
		}

		yield(ai.StreamPart{
			Type:         ai.StreamPartTypeFinish,
			Usage:        usage,
			FinishReason: finishReason,
		})
	}, nil
}

func toGoogleTools(tools []ai.Tool, toolChoice *ai.ToolChoice) (googleTools []*genai.FunctionDeclaration, googleToolChoice *genai.ToolConfig, warnings []ai.CallWarning) {
	for _, tool := range tools {
		if tool.GetType() == ai.ToolTypeFunction {
			ft, ok := tool.(ai.FunctionTool)
			if !ok {
				continue
			}

			required := []string{}
			var properties map[string]any
			if props, ok := ft.InputSchema["properties"]; ok {
				properties, _ = props.(map[string]any)
			}
			if req, ok := ft.InputSchema["required"]; ok {
				if reqArr, ok := req.([]string); ok {
					required = reqArr
				}
			}
			declaration := &genai.FunctionDeclaration{
				Name:        ft.Name,
				Description: ft.Description,
				Parameters: &genai.Schema{
					Type:       genai.TypeObject,
					Properties: convertSchemaProperties(properties),
					Required:   required,
				},
			}
			googleTools = append(googleTools, declaration)
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
		return
	}
	switch *toolChoice {
	case ai.ToolChoiceAuto:
		googleToolChoice = &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode: genai.FunctionCallingConfigModeAuto,
			},
		}
	case ai.ToolChoiceRequired:
		googleToolChoice = &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode: genai.FunctionCallingConfigModeAny,
			},
		}
	case ai.ToolChoiceNone:
		googleToolChoice = &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode: genai.FunctionCallingConfigModeNone,
			},
		}
	default:
		googleToolChoice = &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode: genai.FunctionCallingConfigModeAny,
				AllowedFunctionNames: []string{
					string(*toolChoice),
				},
			},
		}
	}
	return
}

func convertSchemaProperties(parameters map[string]any) map[string]*genai.Schema {
	properties := make(map[string]*genai.Schema)

	for name, param := range parameters {
		properties[name] = convertToSchema(param)
	}

	return properties
}

func convertToSchema(param any) *genai.Schema {
	schema := &genai.Schema{Type: genai.TypeString}

	paramMap, ok := param.(map[string]any)
	if !ok {
		return schema
	}

	if desc, ok := paramMap["description"].(string); ok {
		schema.Description = desc
	}

	typeVal, hasType := paramMap["type"]
	if !hasType {
		return schema
	}

	typeStr, ok := typeVal.(string)
	if !ok {
		return schema
	}

	schema.Type = mapJSONTypeToGoogle(typeStr)

	switch typeStr {
	case "array":
		schema.Items = processArrayItems(paramMap)
	case "object":
		if props, ok := paramMap["properties"].(map[string]any); ok {
			schema.Properties = convertSchemaProperties(props)
		}
	}

	return schema
}

func processArrayItems(paramMap map[string]any) *genai.Schema {
	items, ok := paramMap["items"].(map[string]any)
	if !ok {
		return nil
	}

	return convertToSchema(items)
}

func mapJSONTypeToGoogle(jsonType string) genai.Type {
	switch jsonType {
	case "string":
		return genai.TypeString
	case "number":
		return genai.TypeNumber
	case "integer":
		return genai.TypeInteger
	case "boolean":
		return genai.TypeBoolean
	case "array":
		return genai.TypeArray
	case "object":
		return genai.TypeObject
	default:
		return genai.TypeString // Default to string for unknown types
	}
}

func mapResponse(response *genai.GenerateContentResponse, warnings []ai.CallWarning) (*ai.Response, error) {
	if len(response.Candidates) == 0 || response.Candidates[0].Content == nil {
		return nil, errors.New("no response from model")
	}

	var (
		content      []ai.Content
		finishReason ai.FinishReason
		hasToolCalls bool
		candidate    = response.Candidates[0]
	)

	for _, part := range candidate.Content.Parts {
		switch {
		case part.Text != "":
			content = append(content, ai.TextContent{Text: part.Text})
		case part.FunctionCall != nil:
			input, err := json.Marshal(part.FunctionCall.Args)
			if err != nil {
				return nil, err
			}
			content = append(content, ai.ToolCallContent{
				ToolCallID:       part.FunctionCall.ID,
				ToolName:         part.FunctionCall.Name,
				Input:            string(input),
				ProviderExecuted: false,
			})
			hasToolCalls = true
		default:
			return nil, fmt.Errorf("not implemented part type")
		}
	}

	if hasToolCalls {
		finishReason = ai.FinishReasonToolCalls
	} else {
		finishReason = mapFinishReason(candidate.FinishReason)
	}

	return &ai.Response{
		Content:      content,
		Usage:        mapUsage(response.UsageMetadata),
		FinishReason: finishReason,
		Warnings:     warnings,
	}, nil
}

func mapFinishReason(reason genai.FinishReason) ai.FinishReason {
	switch reason {
	case genai.FinishReasonStop:
		return ai.FinishReasonStop
	case genai.FinishReasonMaxTokens:
		return ai.FinishReasonLength
	case genai.FinishReasonSafety,
		genai.FinishReasonBlocklist,
		genai.FinishReasonProhibitedContent,
		genai.FinishReasonSPII,
		genai.FinishReasonImageSafety:
		return ai.FinishReasonContentFilter
	case genai.FinishReasonRecitation,
		genai.FinishReasonLanguage,
		genai.FinishReasonMalformedFunctionCall:
		return ai.FinishReasonError
	case genai.FinishReasonOther:
		return ai.FinishReasonOther
	default:
		return ai.FinishReasonUnknown
	}
}

func mapUsage(usage *genai.GenerateContentResponseUsageMetadata) ai.Usage {
	return ai.Usage{
		InputTokens:         int64(usage.ToolUsePromptTokenCount),
		OutputTokens:        int64(usage.CandidatesTokenCount),
		TotalTokens:         int64(usage.TotalTokenCount),
		ReasoningTokens:     int64(usage.ThoughtsTokenCount),
		CacheCreationTokens: int64(usage.CachedContentTokenCount),
		CacheReadTokens:     0,
	}
}
