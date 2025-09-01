package google

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"maps"
	"net/http"
	"strings"

	"github.com/charmbracelet/ai/ai"
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
		options.name = "anthropic"
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
	// params, err := g.prepareParams(call)
	// if err != nil {
	// 	return nil, err
	// }
	panic("unimplemented")
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
func (g *languageModel) Stream(context.Context, ai.Call) (ai.StreamResponse, error) {
	panic("unimplemented")
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
