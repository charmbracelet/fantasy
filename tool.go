package fantasy

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strings"
)

// Schema represents a JSON schema for tool input validation.
type Schema struct {
	Type        string             `json:"type,omitempty"`
	Properties  map[string]*Schema `json:"properties,omitempty"`
	Required    []string           `json:"required,omitempty"`
	Items       *Schema            `json:"items,omitempty"`
	Description string             `json:"description,omitempty"`
	Enum        []any              `json:"enum,omitempty"`
	Format      string             `json:"format,omitempty"`
	Minimum     *float64           `json:"minimum,omitempty"`
	Maximum     *float64           `json:"maximum,omitempty"`
	MinLength   *int               `json:"minLength,omitempty"`
	MaxLength   *int               `json:"maxLength,omitempty"`
}

// ToolInfo represents tool metadata, matching the existing pattern.
type ToolInfo struct {
	Name        string
	Description string
	Parameters  map[string]any
	Required    []string
}

// ToolCall represents a tool invocation, matching the existing pattern.
type ToolCall struct {
	ID    string `json:"id"`
	Name  string `json:"name"`
	Input string `json:"input"`
}

// ToolResponse represents the response from a tool execution, matching the existing pattern.
type ToolResponse struct {
	Type     string `json:"type"`
	Content  string `json:"content"`
	Metadata string `json:"metadata,omitempty"`
	IsError  bool   `json:"is_error"`
}

// NewTextResponse creates a text response.
func NewTextResponse(content string) ToolResponse {
	return ToolResponse{
		Type:    "text",
		Content: content,
	}
}

// NewTextErrorResponse creates an error response.
func NewTextErrorResponse(content string) ToolResponse {
	return ToolResponse{
		Type:    "text",
		Content: content,
		IsError: true,
	}
}

// WithResponseMetadata adds metadata to a response.
func WithResponseMetadata(response ToolResponse, metadata any) ToolResponse {
	if metadata != nil {
		metadataBytes, err := json.Marshal(metadata)
		if err != nil {
			return response
		}
		response.Metadata = string(metadataBytes)
	}
	return response
}

// AgentTool represents a tool that can be called by a language model.
// This matches the existing BaseTool interface pattern.
type AgentTool interface {
	Info() ToolInfo
	Run(ctx context.Context, params ToolCall) (ToolResponse, error)
	ProviderOptions() ProviderOptions
	SetProviderOptions(opts ProviderOptions)
}

// NewAgentTool creates a typed tool from a function with automatic schema generation.
// This is the recommended way to create tools.
func NewAgentTool[TInput any](
	name string,
	description string,
	fn func(ctx context.Context, input TInput, call ToolCall) (ToolResponse, error),
) AgentTool {
	var input TInput
	schema := generateSchema(reflect.TypeOf(input))

	return &funcToolWrapper[TInput]{
		name:        name,
		description: description,
		fn:          fn,
		schema:      schema,
	}
}

// funcToolWrapper wraps a function to implement the AgentTool interface.
type funcToolWrapper[TInput any] struct {
	name            string
	description     string
	fn              func(ctx context.Context, input TInput, call ToolCall) (ToolResponse, error)
	schema          Schema
	providerOptions ProviderOptions
}

func (w *funcToolWrapper[TInput]) SetProviderOptions(opts ProviderOptions) {
	w.providerOptions = opts
}

func (w *funcToolWrapper[TInput]) ProviderOptions() ProviderOptions {
	return w.providerOptions
}

func (w *funcToolWrapper[TInput]) Info() ToolInfo {
	if w.schema.Required == nil {
		w.schema.Required = []string{}
	}
	return ToolInfo{
		Name:        w.name,
		Description: w.description,
		Parameters:  schemaToParameters(w.schema),
		Required:    w.schema.Required,
	}
}

func (w *funcToolWrapper[TInput]) Run(ctx context.Context, params ToolCall) (ToolResponse, error) {
	var input TInput
	if err := json.Unmarshal([]byte(params.Input), &input); err != nil {
		return NewTextErrorResponse(fmt.Sprintf("invalid parameters: %s", err)), nil
	}

	return w.fn(ctx, input, params)
}

// schemaToParameters converts a Schema to the parameters map format expected by ToolInfo.
func schemaToParameters(schema Schema) map[string]any {
	if schema.Properties == nil {
		return make(map[string]any)
	}

	result := make(map[string]any)
	for name, propSchema := range schema.Properties {
		result[name] = SchemaToMap(*propSchema)
	}
	return result
}

// SchemaToMap converts a Schema to a map representation suitable for JSON Schema.
func SchemaToMap(schema Schema) map[string]any {
	result := make(map[string]any)

	if schema.Type != "" {
		result["type"] = schema.Type
	}

	if schema.Description != "" {
		result["description"] = schema.Description
	}

	if len(schema.Enum) > 0 {
		result["enum"] = schema.Enum
	}

	if schema.Format != "" {
		result["format"] = schema.Format
	}

	if schema.Minimum != nil {
		result["minimum"] = *schema.Minimum
	}

	if schema.Maximum != nil {
		result["maximum"] = *schema.Maximum
	}

	if schema.MinLength != nil {
		result["minLength"] = *schema.MinLength
	}

	if schema.MaxLength != nil {
		result["maxLength"] = *schema.MaxLength
	}

	if schema.Properties != nil {
		props := make(map[string]any)
		for name, propSchema := range schema.Properties {
			props[name] = SchemaToMap(*propSchema)
		}
		result["properties"] = props
	}

	if len(schema.Required) > 0 {
		result["required"] = schema.Required
	}

	if schema.Items != nil {
		result["items"] = SchemaToMap(*schema.Items)
	}

	return result
}

func generateSchema(t reflect.Type) Schema {
	return generateSchemaRecursive(t, nil, make(map[reflect.Type]bool))
}

func generateSchemaRecursive(t, parent reflect.Type, visited map[reflect.Type]bool) Schema {
	if t.Kind() == reflect.Pointer {
		t = t.Elem()
	}

	if visited[t] {
		return Schema{Type: "object"}
	}
	visited[t] = true
	defer delete(visited, t)

	switch t.Kind() {
	case reflect.String:
		return Schema{Type: "string"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return Schema{Type: "integer"}
	case reflect.Float32, reflect.Float64:
		return Schema{Type: "number"}
	case reflect.Bool:
		return Schema{Type: "boolean"}
	case reflect.Slice, reflect.Array:
		itemSchema := generateSchemaRecursive(t.Elem(), t, visited)
		return Schema{
			Type:  "array",
			Items: &itemSchema,
		}
	case reflect.Map:
		if t.Key().Kind() == reflect.String {
			valueSchema := generateSchemaRecursive(t.Elem(), t, visited)
			schema := Schema{
				Type: "object",
				Properties: map[string]*Schema{
					"*": &valueSchema,
				},
			}
			if useBlankType(parent) {
				schema.Type = ""
			}
			return schema
		}
		return Schema{Type: "object"}
	case reflect.Struct:
		schema := Schema{
			Type:       "object",
			Properties: make(map[string]*Schema),
		}
		if useBlankType(parent) {
			schema.Type = ""
		}

		for i := range t.NumField() {
			field := t.Field(i)

			if !field.IsExported() {
				continue
			}

			jsonTag := field.Tag.Get("json")
			if jsonTag == "-" {
				continue
			}

			fieldName := field.Name
			required := true

			if jsonTag != "" {
				parts := strings.Split(jsonTag, ",")
				if parts[0] != "" {
					fieldName = parts[0]
				}

				if slices.Contains(parts[1:], "omitempty") {
					required = false
				}
			} else {
				fieldName = toSnakeCase(fieldName)
			}

			fieldSchema := generateSchemaRecursive(field.Type, t, visited)

			if desc := field.Tag.Get("description"); desc != "" {
				fieldSchema.Description = desc
			}

			if enumTag := field.Tag.Get("enum"); enumTag != "" {
				enumValues := strings.Split(enumTag, ",")
				fieldSchema.Enum = make([]any, len(enumValues))
				for i, v := range enumValues {
					fieldSchema.Enum[i] = strings.TrimSpace(v)
				}
			}

			schema.Properties[fieldName] = &fieldSchema

			if required {
				schema.Required = append(schema.Required, fieldName)
			}
		}

		return schema
	case reflect.Interface:
		return Schema{Type: "object"}
	default:
		return Schema{Type: "object"}
	}
}

func toSnakeCase(s string) string {
	var result strings.Builder
	for i, r := range s {
		if i > 0 && r >= 'A' && r <= 'Z' {
			result.WriteByte('_')
		}
		result.WriteRune(r)
	}
	return strings.ToLower(result.String())
}

// NOTE(@andreynering): This is a hacky workaround for llama.cpp.
// Ideally, we should always output `type: object` for objects, but
// llama.cpp complains if we do for arrays of objects.
func useBlankType(parent reflect.Type) bool {
	if parent == nil {
		return false
	}
	switch parent.Kind() {
	case reflect.Slice, reflect.Array:
		return true
	default:
		return false
	}
}
