package google

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"strings"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/anthropic"
	"cloud.google.com/go/auth"
	"github.com/charmbracelet/go-genai"
	"github.com/charmbracelet/x/exp/slice"
	"github.com/google/uuid"
)

// Name is the name of the Google provider.
const Name = "google"

type provider struct {
	options options
}

type options struct {
	apiKey   string
	name     string
	baseURL  string
	headers  map[string]string
	client   *http.Client
	backend  genai.Backend
	project  string
	location string
	skipAuth bool
}

// Option defines a function that configures Google provider options.
type Option = func(*options)

// New creates a new Google provider with the given options.
func New(opts ...Option) fantasy.Provider {
	options := options{
		headers: map[string]string{},
	}
	for _, o := range opts {
		o(&options)
	}

	options.name = cmp.Or(options.name, Name)

	return &provider{
		options: options,
	}
}

// WithBaseURL sets the base URL for the Google provider.
func WithBaseURL(baseURL string) Option {
	return func(o *options) {
		o.baseURL = baseURL
	}
}

// WithGeminiAPIKey sets the Gemini API key for the Google provider.
func WithGeminiAPIKey(apiKey string) Option {
	return func(o *options) {
		o.backend = genai.BackendGeminiAPI
		o.apiKey = apiKey
		o.project = ""
		o.location = ""
	}
}

// WithVertex configures the Google provider to use Vertex AI.
func WithVertex(project, location string) Option {
	if project == "" || location == "" {
		panic("project and location must be provided")
	}
	return func(o *options) {
		o.backend = genai.BackendVertexAI
		o.apiKey = ""
		o.project = project
		o.location = location
	}
}

// WithSkipAuth configures whether to skip authentication for the Google provider.
func WithSkipAuth(skipAuth bool) Option {
	return func(o *options) {
		o.skipAuth = skipAuth
	}
}

// WithName sets the name for the Google provider.
func WithName(name string) Option {
	return func(o *options) {
		o.name = name
	}
}

// WithHeaders sets the headers for the Google provider.
func WithHeaders(headers map[string]string) Option {
	return func(o *options) {
		maps.Copy(o.headers, headers)
	}
}

// WithHTTPClient sets the HTTP client for the Google provider.
func WithHTTPClient(client *http.Client) Option {
	return func(o *options) {
		o.client = client
	}
}

func (*provider) Name() string {
	return Name
}

type languageModel struct {
	provider        string
	modelID         string
	client          *genai.Client
	providerOptions options
}

// LanguageModel implements fantasy.Provider.
func (a *provider) LanguageModel(modelID string) (fantasy.LanguageModel, error) {
	if strings.Contains(modelID, "anthropic") || strings.Contains(modelID, "claude") {
		return anthropic.New(
			anthropic.WithVertex(a.options.project, a.options.location),
			anthropic.WithHTTPClient(a.options.client),
			anthropic.WithSkipAuth(a.options.skipAuth),
		).LanguageModel(modelID)
	}

	cc := &genai.ClientConfig{
		HTTPClient: a.options.client,
		Backend:    a.options.backend,
		APIKey:     a.options.apiKey,
		Project:    a.options.project,
		Location:   a.options.location,
	}
	if a.options.skipAuth {
		cc.Credentials = &auth.Credentials{TokenProvider: dummyTokenProvider{}}
	}

	if a.options.baseURL != "" || len(a.options.headers) > 0 {
		headers := http.Header{}
		for k, v := range a.options.headers {
			headers.Add(k, v)
		}
		cc.HTTPOptions = genai.HTTPOptions{
			BaseURL: a.options.baseURL,
			Headers: headers,
		}
	}
	client, err := genai.NewClient(context.Background(), cc)
	if err != nil {
		return nil, err
	}
	return &languageModel{
		modelID:         modelID,
		provider:        a.options.name,
		providerOptions: a.options,
		client:          client,
	}, nil
}

func (g languageModel) prepareParams(call fantasy.Call) (*genai.GenerateContentConfig, []*genai.Content, []fantasy.CallWarning, error) {
	config := &genai.GenerateContentConfig{}

	providerOptions := &ProviderOptions{}
	if v, ok := call.ProviderOptions[Name]; ok {
		providerOptions, ok = v.(*ProviderOptions)
		if !ok {
			return nil, nil, nil, fantasy.NewInvalidArgumentError("providerOptions", "google provider options should be *google.ProviderOptions", nil)
		}
	}

	systemInstructions, content, warnings := toGooglePrompt(call.Prompt)

	if providerOptions.ThinkingConfig != nil {
		if providerOptions.ThinkingConfig.IncludeThoughts != nil &&
			*providerOptions.ThinkingConfig.IncludeThoughts &&
			strings.HasPrefix(g.provider, "google.vertex.") {
			warnings = append(warnings, fantasy.CallWarning{
				Type: fantasy.CallWarningTypeOther,
				Message: "The 'includeThoughts' option is only supported with the Google Vertex provider " +
					"and might not be supported or could behave unexpectedly with the current Google provider " +
					fmt.Sprintf("(%s)", g.provider),
			})
		}

		if providerOptions.ThinkingConfig.ThinkingBudget != nil &&
			*providerOptions.ThinkingConfig.ThinkingBudget < 128 {
			warnings = append(warnings, fantasy.CallWarning{
				Type:    fantasy.CallWarningTypeOther,
				Message: "The 'thinking_budget' option can not be under 128 and will be set to 128 by default",
			})
			providerOptions.ThinkingConfig.ThinkingBudget = fantasy.Opt(int64(128))
		}
	}

	isGemmaModel := strings.HasPrefix(strings.ToLower(g.modelID), "gemma-")

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
		config.MaxOutputTokens = int32(*call.MaxOutputTokens) //nolint: gosec
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
			tmp := int32(*providerOptions.ThinkingConfig.ThinkingBudget) //nolint: gosec
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

func toGooglePrompt(prompt fantasy.Prompt) (*genai.Content, []*genai.Content, []fantasy.CallWarning) { //nolint: unparam
	var systemInstructions *genai.Content
	var content []*genai.Content
	var warnings []fantasy.CallWarning

	finishedSystemBlock := false
	for _, msg := range prompt {
		switch msg.Role {
		case fantasy.MessageRoleSystem:
			if finishedSystemBlock {
				// skip multiple system messages that are separated by user/assistant messages
				// TODO: see if we need to send error here?
				continue
			}
			finishedSystemBlock = true

			var systemMessages []string
			for _, part := range msg.Content {
				text, ok := fantasy.AsMessagePart[fantasy.TextPart](part)
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
		case fantasy.MessageRoleUser:
			var parts []*genai.Part
			for _, part := range msg.Content {
				switch part.GetType() {
				case fantasy.ContentTypeText:
					text, ok := fantasy.AsMessagePart[fantasy.TextPart](part)
					if !ok || text.Text == "" {
						continue
					}
					parts = append(parts, &genai.Part{
						Text: text.Text,
					})
				case fantasy.ContentTypeFile:
					file, ok := fantasy.AsMessagePart[fantasy.FilePart](part)
					if !ok {
						continue
					}
					parts = append(parts, &genai.Part{
						InlineData: &genai.Blob{
							Data:     file.Data,
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
		case fantasy.MessageRoleAssistant:
			var parts []*genai.Part
			for _, part := range msg.Content {
				switch part.GetType() {
				case fantasy.ContentTypeText:
					text, ok := fantasy.AsMessagePart[fantasy.TextPart](part)
					if !ok || text.Text == "" {
						continue
					}
					parts = append(parts, &genai.Part{
						Text: text.Text,
					})
				case fantasy.ContentTypeToolCall:
					toolCall, ok := fantasy.AsMessagePart[fantasy.ToolCallPart](part)
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
		case fantasy.MessageRoleTool:
			var parts []*genai.Part
			for _, part := range msg.Content {
				switch part.GetType() {
				case fantasy.ContentTypeToolResult:
					result, ok := fantasy.AsMessagePart[fantasy.ToolResultPart](part)
					if !ok {
						continue
					}
					var toolCall fantasy.ToolCallPart
					for _, m := range prompt {
						if m.Role == fantasy.MessageRoleAssistant {
							for _, content := range m.Content {
								tc, ok := fantasy.AsMessagePart[fantasy.ToolCallPart](content)
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
					case fantasy.ToolResultContentTypeText:
						content, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentText](result.Output)
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

					case fantasy.ToolResultContentTypeError:
						content, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentError](result.Output)
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
		default:
			panic("unsupported message role: " + msg.Role)
		}
	}
	return systemInstructions, content, warnings
}

// Generate implements fantasy.LanguageModel.
func (g *languageModel) Generate(ctx context.Context, call fantasy.Call) (*fantasy.Response, error) {
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

// Model implements fantasy.LanguageModel.
func (g *languageModel) Model() string {
	return g.modelID
}

// Provider implements fantasy.LanguageModel.
func (g *languageModel) Provider() string {
	return g.provider
}

// Stream implements fantasy.LanguageModel.
func (g *languageModel) Stream(ctx context.Context, call fantasy.Call) (fantasy.StreamResponse, error) {
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

	return func(yield func(fantasy.StreamPart) bool) {
		if len(warnings) > 0 {
			if !yield(fantasy.StreamPart{
				Type:     fantasy.StreamPartTypeWarnings,
				Warnings: warnings,
			}) {
				return
			}
		}

		var currentContent string
		var toolCalls []fantasy.ToolCallContent
		var isActiveText bool
		var isActiveReasoning bool
		var blockCounter int
		var currentTextBlockID string
		var currentReasoningBlockID string
		var usage fantasy.Usage
		var lastFinishReason fantasy.FinishReason

		for resp, err := range chat.SendMessageStream(ctx, depointerSlice(lastMessage.Parts)...) {
			if err != nil {
				yield(fantasy.StreamPart{
					Type:  fantasy.StreamPartTypeError,
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
							// Check if this is a reasoning/thought part
							if part.Thought {
								// End any active text block before starting reasoning
								if isActiveText {
									isActiveText = false
									if !yield(fantasy.StreamPart{
										Type: fantasy.StreamPartTypeTextEnd,
										ID:   currentTextBlockID,
									}) {
										return
									}
								}

								// Start new reasoning block if not already active
								if !isActiveReasoning {
									isActiveReasoning = true
									currentReasoningBlockID = fmt.Sprintf("%d", blockCounter)
									blockCounter++
									if !yield(fantasy.StreamPart{
										Type: fantasy.StreamPartTypeReasoningStart,
										ID:   currentReasoningBlockID,
									}) {
										return
									}
								}

								if !yield(fantasy.StreamPart{
									Type:  fantasy.StreamPartTypeReasoningDelta,
									ID:    currentReasoningBlockID,
									Delta: delta,
								}) {
									return
								}
							} else {
								// Regular text part
								// End any active reasoning block before starting text
								if isActiveReasoning {
									isActiveReasoning = false
									if !yield(fantasy.StreamPart{
										Type: fantasy.StreamPartTypeReasoningEnd,
										ID:   currentReasoningBlockID,
									}) {
										return
									}
								}

								// Start new text block if not already active
								if !isActiveText {
									isActiveText = true
									currentTextBlockID = fmt.Sprintf("%d", blockCounter)
									blockCounter++
									if !yield(fantasy.StreamPart{
										Type: fantasy.StreamPartTypeTextStart,
										ID:   currentTextBlockID,
									}) {
										return
									}
								}

								if !yield(fantasy.StreamPart{
									Type:  fantasy.StreamPartTypeTextDelta,
									ID:    currentTextBlockID,
									Delta: delta,
								}) {
									return
								}
								currentContent += delta
							}
						}
					case part.FunctionCall != nil:
						// End any active text or reasoning blocks
						if isActiveText {
							isActiveText = false
							if !yield(fantasy.StreamPart{
								Type: fantasy.StreamPartTypeTextEnd,
								ID:   currentTextBlockID,
							}) {
								return
							}
						}
						if isActiveReasoning {
							isActiveReasoning = false
							if !yield(fantasy.StreamPart{
								Type: fantasy.StreamPartTypeReasoningEnd,
								ID:   currentReasoningBlockID,
							}) {
								return
							}
						}

						toolCallID := cmp.Or(part.FunctionCall.ID, part.FunctionCall.Name, uuid.NewString())

						args, err := json.Marshal(part.FunctionCall.Args)
						if err != nil {
							yield(fantasy.StreamPart{
								Type:  fantasy.StreamPartTypeError,
								Error: err,
							})
							return
						}

						if !yield(fantasy.StreamPart{
							Type:         fantasy.StreamPartTypeToolInputStart,
							ID:           toolCallID,
							ToolCallName: part.FunctionCall.Name,
						}) {
							return
						}

						if !yield(fantasy.StreamPart{
							Type:  fantasy.StreamPartTypeToolInputDelta,
							ID:    toolCallID,
							Delta: string(args),
						}) {
							return
						}

						if !yield(fantasy.StreamPart{
							Type: fantasy.StreamPartTypeToolInputEnd,
							ID:   toolCallID,
						}) {
							return
						}

						if !yield(fantasy.StreamPart{
							Type:             fantasy.StreamPartTypeToolCall,
							ID:               toolCallID,
							ToolCallName:     part.FunctionCall.Name,
							ToolCallInput:    string(args),
							ProviderExecuted: false,
						}) {
							return
						}

						toolCalls = append(toolCalls, fantasy.ToolCallContent{
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

			if len(resp.Candidates) > 0 && resp.Candidates[0].FinishReason != "" {
				lastFinishReason = mapFinishReason(resp.Candidates[0].FinishReason)
			}
		}

		// Close any open blocks before finishing
		if isActiveText {
			if !yield(fantasy.StreamPart{
				Type: fantasy.StreamPartTypeTextEnd,
				ID:   currentTextBlockID,
			}) {
				return
			}
		}
		if isActiveReasoning {
			if !yield(fantasy.StreamPart{
				Type: fantasy.StreamPartTypeReasoningEnd,
				ID:   currentReasoningBlockID,
			}) {
				return
			}
		}

		finishReason := lastFinishReason
		if len(toolCalls) > 0 {
			finishReason = fantasy.FinishReasonToolCalls
		} else if finishReason == "" {
			finishReason = fantasy.FinishReasonStop
		}

		yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeFinish,
			Usage:        usage,
			FinishReason: finishReason,
		})
	}, nil
}

func toGoogleTools(tools []fantasy.Tool, toolChoice *fantasy.ToolChoice) (googleTools []*genai.FunctionDeclaration, googleToolChoice *genai.ToolConfig, warnings []fantasy.CallWarning) {
	for _, tool := range tools {
		if tool.GetType() == fantasy.ToolTypeFunction {
			ft, ok := tool.(fantasy.FunctionTool)
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
		warnings = append(warnings, fantasy.CallWarning{
			Type:    fantasy.CallWarningTypeUnsupportedTool,
			Tool:    tool,
			Message: "tool is not supported",
		})
	}
	if toolChoice == nil {
		return googleTools, googleToolChoice, warnings
	}
	switch *toolChoice {
	case fantasy.ToolChoiceAuto:
		googleToolChoice = &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode: genai.FunctionCallingConfigModeAuto,
			},
		}
	case fantasy.ToolChoiceRequired:
		googleToolChoice = &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode: genai.FunctionCallingConfigModeAny,
			},
		}
	case fantasy.ToolChoiceNone:
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
	return googleTools, googleToolChoice, warnings
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

func mapResponse(response *genai.GenerateContentResponse, warnings []fantasy.CallWarning) (*fantasy.Response, error) {
	if len(response.Candidates) == 0 || response.Candidates[0].Content == nil {
		return nil, errors.New("no response from model")
	}

	var (
		content      []fantasy.Content
		finishReason fantasy.FinishReason
		hasToolCalls bool
		candidate    = response.Candidates[0]
	)

	for _, part := range candidate.Content.Parts {
		switch {
		case part.Text != "":
			if part.Thought {
				content = append(content, fantasy.ReasoningContent{Text: part.Text})
			} else {
				content = append(content, fantasy.TextContent{Text: part.Text})
			}
		case part.FunctionCall != nil:
			input, err := json.Marshal(part.FunctionCall.Args)
			if err != nil {
				return nil, err
			}
			toolCallID := cmp.Or(part.FunctionCall.ID, part.FunctionCall.Name, uuid.NewString())
			content = append(content, fantasy.ToolCallContent{
				ToolCallID:       toolCallID,
				ToolName:         part.FunctionCall.Name,
				Input:            string(input),
				ProviderExecuted: false,
			})
			hasToolCalls = true
		default:
			// Silently skip unknown part types instead of erroring
			// This allows for forward compatibility with new part types
		}
	}

	if hasToolCalls {
		finishReason = fantasy.FinishReasonToolCalls
	} else {
		finishReason = mapFinishReason(candidate.FinishReason)
	}

	return &fantasy.Response{
		Content:      content,
		Usage:        mapUsage(response.UsageMetadata),
		FinishReason: finishReason,
		Warnings:     warnings,
	}, nil
}

func mapFinishReason(reason genai.FinishReason) fantasy.FinishReason {
	switch reason {
	case genai.FinishReasonStop:
		return fantasy.FinishReasonStop
	case genai.FinishReasonMaxTokens:
		return fantasy.FinishReasonLength
	case genai.FinishReasonSafety,
		genai.FinishReasonBlocklist,
		genai.FinishReasonProhibitedContent,
		genai.FinishReasonSPII,
		genai.FinishReasonImageSafety:
		return fantasy.FinishReasonContentFilter
	case genai.FinishReasonRecitation,
		genai.FinishReasonLanguage,
		genai.FinishReasonMalformedFunctionCall:
		return fantasy.FinishReasonError
	case genai.FinishReasonOther:
		return fantasy.FinishReasonOther
	default:
		return fantasy.FinishReasonUnknown
	}
}

func mapUsage(usage *genai.GenerateContentResponseUsageMetadata) fantasy.Usage {
	return fantasy.Usage{
		InputTokens:         int64(usage.ToolUsePromptTokenCount),
		OutputTokens:        int64(usage.CandidatesTokenCount),
		TotalTokens:         int64(usage.TotalTokenCount),
		ReasoningTokens:     int64(usage.ThoughtsTokenCount),
		CacheCreationTokens: int64(usage.CachedContentTokenCount),
		CacheReadTokens:     0,
	}
}
