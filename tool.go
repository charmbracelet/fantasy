// WIP NEED TO REVISIT
package ai

import (
	"context"
	"encoding/json"
	"fmt"
)

// AgentTool represents a function that can be called by a language model.
type AgentTool interface {
	Name() string
	Description() string
	InputSchema() Schema
	Execute(ctx context.Context, input json.RawMessage) (json.RawMessage, error)
}

// Schema represents a JSON schema for tool input validation.
type Schema struct {
	Type        string             `json:"type"`
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

// BasicTool provides a basic implementation of the Tool interface
//
// Example usage:
//
//	calculator := &tools.BasicTool{
//	    ToolName:        "calculate",
//	    ToolDescription: "Evaluates mathematical expressions",
//	    ToolInputSchema: tools.Schema{
//	        Type: "object",
//	        Properties: map[string]*tools.Schema{
//	            "expression": {
//	                Type:        "string",
//	                Description: "Mathematical expression to evaluate",
//	            },
//	        },
//	        Required: []string{"expression"},
//	    },
//	    ExecuteFunc: func(ctx context.Context, input json.RawMessage) (json.RawMessage, error) {
//	        var req struct {
//	            Expression string `json:"expression"`
//	        }
//	        if err := json.Unmarshal(input, &req); err != nil {
//	            return nil, err
//	        }
//	        result := evaluateExpression(req.Expression)
//	        return json.Marshal(map[string]any{"result": result})
//	    },
//	}
type BasicTool struct {
	ToolName        string
	ToolDescription string
	ToolInputSchema Schema
	ExecuteFunc     func(context.Context, json.RawMessage) (json.RawMessage, error)
}

// Name returns the tool name.
func (t *BasicTool) Name() string {
	return t.ToolName
}

// Description returns the tool description.
func (t *BasicTool) Description() string {
	return t.ToolDescription
}

// InputSchema returns the tool input schema.
func (t *BasicTool) InputSchema() Schema {
	return t.ToolInputSchema
}

// Execute executes the tool with the given input.
func (t *BasicTool) Execute(ctx context.Context, input json.RawMessage) (json.RawMessage, error) {
	if t.ExecuteFunc == nil {
		return nil, fmt.Errorf("tool %s has no execute function", t.ToolName)
	}
	return t.ExecuteFunc(ctx, input)
}

// ToolBuilder provides a fluent interface for building tools.
type ToolBuilder struct {
	tool *BasicTool
}

// NewTool creates a new tool builder.
func NewTool(name string) *ToolBuilder {
	return &ToolBuilder{
		tool: &BasicTool{
			ToolName: name,
		},
	}
}

// Description sets the tool description.
func (b *ToolBuilder) Description(desc string) *ToolBuilder {
	b.tool.ToolDescription = desc
	return b
}

// InputSchema sets the tool input schema.
func (b *ToolBuilder) InputSchema(schema Schema) *ToolBuilder {
	b.tool.ToolInputSchema = schema
	return b
}

// Execute sets the tool execution function.
func (b *ToolBuilder) Execute(fn func(context.Context, json.RawMessage) (json.RawMessage, error)) *ToolBuilder {
	b.tool.ExecuteFunc = fn
	return b
}

// Build creates the final tool.
func (b *ToolBuilder) Build() AgentTool {
	return b.tool
}

// SchemaBuilder provides a fluent interface for building JSON schemas.
type SchemaBuilder struct {
	schema Schema
}

// NewSchema creates a new schema builder.
func NewSchema(schemaType string) *SchemaBuilder {
	return &SchemaBuilder{
		schema: Schema{
			Type: schemaType,
		},
	}
}

// Object creates a schema builder for an object type.
func Object() *SchemaBuilder {
	return NewSchema("object")
}

// String creates a schema builder for a string type.
func String() *SchemaBuilder {
	return NewSchema("string")
}

// Number creates a schema builder for a number type.
func Number() *SchemaBuilder {
	return NewSchema("number")
}

// Array creates a schema builder for an array type.
func Array() *SchemaBuilder {
	return NewSchema("array")
}

// Description sets the schema description.
func (b *SchemaBuilder) Description(desc string) *SchemaBuilder {
	b.schema.Description = desc
	return b
}

// Properties sets the schema properties.
func (b *SchemaBuilder) Properties(props map[string]*Schema) *SchemaBuilder {
	b.schema.Properties = props
	return b
}

// Property adds a property to the schema.
func (b *SchemaBuilder) Property(name string, schema *Schema) *SchemaBuilder {
	if b.schema.Properties == nil {
		b.schema.Properties = make(map[string]*Schema)
	}
	b.schema.Properties[name] = schema
	return b
}

// Required marks fields as required.
func (b *SchemaBuilder) Required(fields ...string) *SchemaBuilder {
	b.schema.Required = append(b.schema.Required, fields...)
	return b
}

// Items sets the schema for array items.
func (b *SchemaBuilder) Items(schema *Schema) *SchemaBuilder {
	b.schema.Items = schema
	return b
}

// Enum sets allowed values for the schema.
func (b *SchemaBuilder) Enum(values ...any) *SchemaBuilder {
	b.schema.Enum = values
	return b
}

// Format sets the string format.
func (b *SchemaBuilder) Format(format string) *SchemaBuilder {
	b.schema.Format = format
	return b
}

// Min sets the minimum value.
func (b *SchemaBuilder) Min(minimum float64) *SchemaBuilder {
	b.schema.Minimum = &minimum
	return b
}

// Max sets the maximum value.
func (b *SchemaBuilder) Max(maximum float64) *SchemaBuilder {
	b.schema.Maximum = &maximum
	return b
}

// MinLength sets the minimum string length.
func (b *SchemaBuilder) MinLength(minimum int) *SchemaBuilder {
	b.schema.MinLength = &minimum
	return b
}

// MaxLength sets the maximum string length.
func (b *SchemaBuilder) MaxLength(maximum int) *SchemaBuilder {
	b.schema.MaxLength = &maximum
	return b
}

// Build creates the final schema.
func (b *SchemaBuilder) Build() *Schema {
	return &b.schema
}
