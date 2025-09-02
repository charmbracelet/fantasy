package ai

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

// Example of a simple typed tool using the function approach
type CalculatorInput struct {
	Expression string `json:"expression" description:"Mathematical expression to evaluate"`
}

func TestTypedToolFuncExample(t *testing.T) {
	// Create a typed tool using the function API
	tool := NewAgentTool(
		"calculator",
		"Evaluates simple mathematical expressions",
		func(ctx context.Context, input CalculatorInput, _ ToolCall) (ToolResponse, error) {
			if input.Expression == "2+2" {
				return NewTextResponse("4"), nil
			}
			return NewTextErrorResponse("unsupported expression"), nil
		},
	)

	// Check the tool info
	info := tool.Info()
	if info.Name != "calculator" {
		t.Errorf("Expected tool name 'calculator', got %s", info.Name)
	}
	if len(info.Required) != 1 || info.Required[0] != "expression" {
		t.Errorf("Expected required field 'expression', got %v", info.Required)
	}

	// Test execution
	call := ToolCall{
		ID:    "test-1",
		Name:  "calculator",
		Input: `{"expression": "2+2"}`,
	}

	result, err := tool.Run(context.Background(), call)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if result.Content != "4" {
		t.Errorf("Expected result '4', got %s", result.Content)
	}
	if result.IsError {
		t.Errorf("Expected successful result, got error")
	}
}

func TestEnumToolExample(t *testing.T) {
	type WeatherInput struct {
		Location string `json:"location" description:"City name"`
		Units    string `json:"units" enum:"celsius,fahrenheit" description:"Temperature units"`
	}

	// Create a weather tool with enum support
	tool := NewAgentTool(
		"weather",
		"Gets current weather for a location",
		func(ctx context.Context, input WeatherInput, _ ToolCall) (ToolResponse, error) {
			temp := "22°C"
			if input.Units == "fahrenheit" {
				temp = "72°F"
			}
			return NewTextResponse(fmt.Sprintf("Weather in %s: %s, sunny", input.Location, temp)), nil
		},
	)

	// Check that the schema includes enum values
	info := tool.Info()
	unitsParam, ok := info.Parameters["units"].(map[string]any)
	if !ok {
		t.Fatal("Expected units parameter to exist")
	}
	enumValues, ok := unitsParam["enum"].([]any)
	if !ok || len(enumValues) != 2 {
		t.Errorf("Expected 2 enum values, got %v", enumValues)
	}

	// Test execution with enum value
	call := ToolCall{
		ID:    "test-2",
		Name:  "weather",
		Input: `{"location": "San Francisco", "units": "fahrenheit"}`,
	}

	result, err := tool.Run(context.Background(), call)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !strings.Contains(result.Content, "San Francisco") {
		t.Errorf("Expected result to contain 'San Francisco', got %s", result.Content)
	}
	if !strings.Contains(result.Content, "72°F") {
		t.Errorf("Expected result to contain '72°F', got %s", result.Content)
	}
}

func TestEnumSupport(t *testing.T) {
	// Test enum via struct tags
	type WeatherInput struct {
		Location string `json:"location" description:"City name"`
		Units    string `json:"units" enum:"celsius,fahrenheit,kelvin" description:"Temperature units"`
		Format   string `json:"format,omitempty" enum:"json,xml,text"`
	}

	schema := generateSchema(reflect.TypeOf(WeatherInput{}))

	if schema.Type != "object" {
		t.Errorf("Expected schema type 'object', got %s", schema.Type)
	}

	// Check units field has enum values
	unitsSchema := schema.Properties["units"]
	if unitsSchema == nil {
		t.Fatal("Expected units property to exist")
	}
	if len(unitsSchema.Enum) != 3 {
		t.Errorf("Expected 3 enum values for units, got %d", len(unitsSchema.Enum))
	}
	expectedUnits := []string{"celsius", "fahrenheit", "kelvin"}
	for i, expected := range expectedUnits {
		if unitsSchema.Enum[i] != expected {
			t.Errorf("Expected enum value %s, got %v", expected, unitsSchema.Enum[i])
		}
	}

	// Check required fields (format should not be required due to omitempty)
	expectedRequired := []string{"location", "units"}
	if len(schema.Required) != len(expectedRequired) {
		t.Errorf("Expected %d required fields, got %d", len(expectedRequired), len(schema.Required))
	}
}

func TestSchemaToParameters(t *testing.T) {
	schema := Schema{
		Type: "object",
		Properties: map[string]*Schema{
			"name": {
				Type:        "string",
				Description: "The name field",
			},
			"age": {
				Type:    "integer",
				Minimum: func() *float64 { v := 0.0; return &v }(),
				Maximum: func() *float64 { v := 120.0; return &v }(),
			},
			"tags": {
				Type: "array",
				Items: &Schema{
					Type: "string",
				},
			},
			"priority": {
				Type: "string",
				Enum: []any{"low", "medium", "high"},
			},
		},
		Required: []string{"name"},
	}

	params := schemaToParameters(schema)

	// Check name parameter
	nameParam, ok := params["name"].(map[string]any)
	if !ok {
		t.Fatal("Expected name parameter to exist")
	}
	if nameParam["type"] != "string" {
		t.Errorf("Expected name type 'string', got %v", nameParam["type"])
	}
	if nameParam["description"] != "The name field" {
		t.Errorf("Expected name description 'The name field', got %v", nameParam["description"])
	}

	// Check age parameter with min/max
	ageParam, ok := params["age"].(map[string]any)
	if !ok {
		t.Fatal("Expected age parameter to exist")
	}
	if ageParam["type"] != "integer" {
		t.Errorf("Expected age type 'integer', got %v", ageParam["type"])
	}
	if ageParam["minimum"] != 0.0 {
		t.Errorf("Expected age minimum 0, got %v", ageParam["minimum"])
	}
	if ageParam["maximum"] != 120.0 {
		t.Errorf("Expected age maximum 120, got %v", ageParam["maximum"])
	}

	// Check priority parameter with enum
	priorityParam, ok := params["priority"].(map[string]any)
	if !ok {
		t.Fatal("Expected priority parameter to exist")
	}
	if priorityParam["type"] != "string" {
		t.Errorf("Expected priority type 'string', got %v", priorityParam["type"])
	}
	enumValues, ok := priorityParam["enum"].([]any)
	if !ok || len(enumValues) != 3 {
		t.Errorf("Expected 3 enum values, got %v", enumValues)
	}
}

func TestGenerateSchemaBasicTypes(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		input    any
		expected Schema
	}{
		{
			name:     "string type",
			input:    "",
			expected: Schema{Type: "string"},
		},
		{
			name:     "int type",
			input:    0,
			expected: Schema{Type: "integer"},
		},
		{
			name:     "int64 type",
			input:    int64(0),
			expected: Schema{Type: "integer"},
		},
		{
			name:     "uint type",
			input:    uint(0),
			expected: Schema{Type: "integer"},
		},
		{
			name:     "float64 type",
			input:    0.0,
			expected: Schema{Type: "number"},
		},
		{
			name:     "float32 type",
			input:    float32(0.0),
			expected: Schema{Type: "number"},
		},
		{
			name:     "bool type",
			input:    false,
			expected: Schema{Type: "boolean"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			schema := generateSchema(reflect.TypeOf(tt.input))
			if schema.Type != tt.expected.Type {
				t.Errorf("Expected type %s, got %s", tt.expected.Type, schema.Type)
			}
		})
	}
}

func TestGenerateSchemaArrayTypes(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		input    any
		expected Schema
	}{
		{
			name:  "string slice",
			input: []string{},
			expected: Schema{
				Type:  "array",
				Items: &Schema{Type: "string"},
			},
		},
		{
			name:  "int slice",
			input: []int{},
			expected: Schema{
				Type:  "array",
				Items: &Schema{Type: "integer"},
			},
		},
		{
			name:  "string array",
			input: [3]string{},
			expected: Schema{
				Type:  "array",
				Items: &Schema{Type: "string"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			schema := generateSchema(reflect.TypeOf(tt.input))
			if schema.Type != tt.expected.Type {
				t.Errorf("Expected type %s, got %s", tt.expected.Type, schema.Type)
			}
			if schema.Items == nil {
				t.Fatal("Expected items schema to exist")
			}
			if schema.Items.Type != tt.expected.Items.Type {
				t.Errorf("Expected items type %s, got %s", tt.expected.Items.Type, schema.Items.Type)
			}
		})
	}
}

func TestGenerateSchemaMapTypes(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		input    any
		expected string
	}{
		{
			name:     "string to string map",
			input:    map[string]string{},
			expected: "object",
		},
		{
			name:     "string to int map",
			input:    map[string]int{},
			expected: "object",
		},
		{
			name:     "int to string map",
			input:    map[int]string{},
			expected: "object",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			schema := generateSchema(reflect.TypeOf(tt.input))
			if schema.Type != tt.expected {
				t.Errorf("Expected type %s, got %s", tt.expected, schema.Type)
			}
		})
	}
}

func TestGenerateSchemaStructTypes(t *testing.T) {
	t.Parallel()

	type SimpleStruct struct {
		Name string `json:"name" description:"The name field"`
		Age  int    `json:"age"`
	}

	type StructWithOmitEmpty struct {
		Required string `json:"required"`
		Optional string `json:"optional,omitempty"`
	}

	type StructWithJSONIgnore struct {
		Visible string `json:"visible"`
		Hidden  string `json:"-"`
	}

	type StructWithoutJSONTags struct {
		FirstName string
		LastName  string
	}

	tests := []struct {
		name     string
		input    any
		validate func(t *testing.T, schema Schema)
	}{
		{
			name:  "simple struct",
			input: SimpleStruct{},
			validate: func(t *testing.T, schema Schema) {
				if schema.Type != "object" {
					t.Errorf("Expected type object, got %s", schema.Type)
				}
				if len(schema.Properties) != 2 {
					t.Errorf("Expected 2 properties, got %d", len(schema.Properties))
				}
				if schema.Properties["name"] == nil {
					t.Error("Expected name property to exist")
				}
				if schema.Properties["name"].Description != "The name field" {
					t.Errorf("Expected description 'The name field', got %s", schema.Properties["name"].Description)
				}
				if len(schema.Required) != 2 {
					t.Errorf("Expected 2 required fields, got %d", len(schema.Required))
				}
			},
		},
		{
			name:  "struct with omitempty",
			input: StructWithOmitEmpty{},
			validate: func(t *testing.T, schema Schema) {
				if len(schema.Required) != 1 {
					t.Errorf("Expected 1 required field, got %d", len(schema.Required))
				}
				if schema.Required[0] != "required" {
					t.Errorf("Expected required field 'required', got %s", schema.Required[0])
				}
			},
		},
		{
			name:  "struct with json ignore",
			input: StructWithJSONIgnore{},
			validate: func(t *testing.T, schema Schema) {
				if len(schema.Properties) != 1 {
					t.Errorf("Expected 1 property, got %d", len(schema.Properties))
				}
				if schema.Properties["visible"] == nil {
					t.Error("Expected visible property to exist")
				}
				if schema.Properties["hidden"] != nil {
					t.Error("Expected hidden property to not exist")
				}
			},
		},
		{
			name:  "struct without json tags",
			input: StructWithoutJSONTags{},
			validate: func(t *testing.T, schema Schema) {
				if schema.Properties["first_name"] == nil {
					t.Error("Expected first_name property to exist")
				}
				if schema.Properties["last_name"] == nil {
					t.Error("Expected last_name property to exist")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			schema := generateSchema(reflect.TypeOf(tt.input))
			tt.validate(t, schema)
		})
	}
}

func TestGenerateSchemaPointerTypes(t *testing.T) {
	t.Parallel()

	type StructWithPointers struct {
		Name *string `json:"name"`
		Age  *int    `json:"age"`
	}

	schema := generateSchema(reflect.TypeOf(StructWithPointers{}))

	if schema.Type != "object" {
		t.Errorf("Expected type object, got %s", schema.Type)
	}

	if schema.Properties["name"] == nil {
		t.Fatal("Expected name property to exist")
	}
	if schema.Properties["name"].Type != "string" {
		t.Errorf("Expected name type string, got %s", schema.Properties["name"].Type)
	}

	if schema.Properties["age"] == nil {
		t.Fatal("Expected age property to exist")
	}
	if schema.Properties["age"].Type != "integer" {
		t.Errorf("Expected age type integer, got %s", schema.Properties["age"].Type)
	}
}

func TestGenerateSchemaNestedStructs(t *testing.T) {
	t.Parallel()

	type Address struct {
		Street string `json:"street"`
		City   string `json:"city"`
	}

	type Person struct {
		Name    string  `json:"name"`
		Address Address `json:"address"`
	}

	schema := generateSchema(reflect.TypeOf(Person{}))

	if schema.Type != "object" {
		t.Errorf("Expected type object, got %s", schema.Type)
	}

	if schema.Properties["address"] == nil {
		t.Fatal("Expected address property to exist")
	}

	addressSchema := schema.Properties["address"]
	if addressSchema.Type != "object" {
		t.Errorf("Expected address type object, got %s", addressSchema.Type)
	}

	if addressSchema.Properties["street"] == nil {
		t.Error("Expected street property in address to exist")
	}
	if addressSchema.Properties["city"] == nil {
		t.Error("Expected city property in address to exist")
	}
}

func TestGenerateSchemaRecursiveStructs(t *testing.T) {
	t.Parallel()

	type Node struct {
		Value string `json:"value"`
		Next  *Node  `json:"next,omitempty"`
	}

	schema := generateSchema(reflect.TypeOf(Node{}))

	if schema.Type != "object" {
		t.Errorf("Expected type object, got %s", schema.Type)
	}

	if schema.Properties["value"] == nil {
		t.Error("Expected value property to exist")
	}

	if schema.Properties["next"] == nil {
		t.Error("Expected next property to exist")
	}

	// The recursive reference should be handled gracefully
	nextSchema := schema.Properties["next"]
	if nextSchema.Type != "object" {
		t.Errorf("Expected next type object, got %s", nextSchema.Type)
	}
}

func TestGenerateSchemaWithEnumTags(t *testing.T) {
	t.Parallel()

	type ConfigInput struct {
		Level    string `json:"level" enum:"debug,info,warn,error" description:"Log level"`
		Format   string `json:"format" enum:"json,text"`
		Optional string `json:"optional,omitempty" enum:"a,b,c"`
	}

	schema := generateSchema(reflect.TypeOf(ConfigInput{}))

	// Check level field
	levelSchema := schema.Properties["level"]
	if levelSchema == nil {
		t.Fatal("Expected level property to exist")
	}
	if len(levelSchema.Enum) != 4 {
		t.Errorf("Expected 4 enum values for level, got %d", len(levelSchema.Enum))
	}
	expectedLevels := []string{"debug", "info", "warn", "error"}
	for i, expected := range expectedLevels {
		if levelSchema.Enum[i] != expected {
			t.Errorf("Expected enum value %s, got %v", expected, levelSchema.Enum[i])
		}
	}

	// Check format field
	formatSchema := schema.Properties["format"]
	if formatSchema == nil {
		t.Fatal("Expected format property to exist")
	}
	if len(formatSchema.Enum) != 2 {
		t.Errorf("Expected 2 enum values for format, got %d", len(formatSchema.Enum))
	}

	// Check required fields (optional should not be required due to omitempty)
	expectedRequired := []string{"level", "format"}
	if len(schema.Required) != len(expectedRequired) {
		t.Errorf("Expected %d required fields, got %d", len(expectedRequired), len(schema.Required))
	}
}

func TestGenerateSchemaComplexTypes(t *testing.T) {
	t.Parallel()

	type ComplexInput struct {
		StringSlice []string            `json:"string_slice"`
		IntMap      map[string]int      `json:"int_map"`
		NestedSlice []map[string]string `json:"nested_slice"`
		Interface   any                 `json:"interface"`
	}

	schema := generateSchema(reflect.TypeOf(ComplexInput{}))

	// Check string slice
	stringSliceSchema := schema.Properties["string_slice"]
	if stringSliceSchema == nil {
		t.Fatal("Expected string_slice property to exist")
	}
	if stringSliceSchema.Type != "array" {
		t.Errorf("Expected string_slice type array, got %s", stringSliceSchema.Type)
	}
	if stringSliceSchema.Items.Type != "string" {
		t.Errorf("Expected string_slice items type string, got %s", stringSliceSchema.Items.Type)
	}

	// Check int map
	intMapSchema := schema.Properties["int_map"]
	if intMapSchema == nil {
		t.Fatal("Expected int_map property to exist")
	}
	if intMapSchema.Type != "object" {
		t.Errorf("Expected int_map type object, got %s", intMapSchema.Type)
	}

	// Check nested slice
	nestedSliceSchema := schema.Properties["nested_slice"]
	if nestedSliceSchema == nil {
		t.Fatal("Expected nested_slice property to exist")
	}
	if nestedSliceSchema.Type != "array" {
		t.Errorf("Expected nested_slice type array, got %s", nestedSliceSchema.Type)
	}
	if nestedSliceSchema.Items.Type != "object" {
		t.Errorf("Expected nested_slice items type object, got %s", nestedSliceSchema.Items.Type)
	}

	// Check interface
	interfaceSchema := schema.Properties["interface"]
	if interfaceSchema == nil {
		t.Fatal("Expected interface property to exist")
	}
	if interfaceSchema.Type != "object" {
		t.Errorf("Expected interface type object, got %s", interfaceSchema.Type)
	}
}

func TestToSnakeCase(t *testing.T) {
	t.Parallel()

	tests := []struct {
		input    string
		expected string
	}{
		{"FirstName", "first_name"},
		{"XMLHttpRequest", "x_m_l_http_request"},
		{"ID", "i_d"},
		{"HTTPSProxy", "h_t_t_p_s_proxy"},
		{"simple", "simple"},
		{"", ""},
		{"A", "a"},
		{"AB", "a_b"},
		{"CamelCase", "camel_case"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			t.Parallel()
			result := toSnakeCase(tt.input)
			if result != tt.expected {
				t.Errorf("toSnakeCase(%s) = %s, expected %s", tt.input, result, tt.expected)
			}
		})
	}
}

func TestSchemaToParametersEdgeCases(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		schema   Schema
		expected map[string]any
	}{
		{
			name: "non-object schema",
			schema: Schema{
				Type: "string",
			},
			expected: map[string]any{},
		},
		{
			name: "object with no properties",
			schema: Schema{
				Type:       "object",
				Properties: nil,
			},
			expected: map[string]any{},
		},
		{
			name: "object with empty properties",
			schema: Schema{
				Type:       "object",
				Properties: map[string]*Schema{},
			},
			expected: map[string]any{},
		},
		{
			name: "schema with all constraint types",
			schema: Schema{
				Type: "object",
				Properties: map[string]*Schema{
					"text": {
						Type:      "string",
						Format:    "email",
						MinLength: func() *int { v := 5; return &v }(),
						MaxLength: func() *int { v := 100; return &v }(),
					},
					"number": {
						Type:    "number",
						Minimum: func() *float64 { v := 0.0; return &v }(),
						Maximum: func() *float64 { v := 100.0; return &v }(),
					},
				},
			},
			expected: map[string]any{
				"text": map[string]any{
					"type":      "string",
					"format":    "email",
					"minLength": 5,
					"maxLength": 100,
				},
				"number": map[string]any{
					"type":    "number",
					"minimum": 0.0,
					"maximum": 100.0,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			result := schemaToParameters(tt.schema)
			if len(result) != len(tt.expected) {
				t.Errorf("Expected %d parameters, got %d", len(tt.expected), len(result))
			}
			for key, expectedValue := range tt.expected {
				if result[key] == nil {
					t.Errorf("Expected parameter %s to exist", key)
					continue
				}
				// Deep comparison would be complex, so we'll check key properties
				resultParam := result[key].(map[string]any)
				expectedParam := expectedValue.(map[string]any)
				for propKey, propValue := range expectedParam {
					if resultParam[propKey] != propValue {
						t.Errorf("Expected %s.%s = %v, got %v", key, propKey, propValue, resultParam[propKey])
					}
				}
			}
		})
	}
}
