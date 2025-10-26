package fantasy

import (
	"context"
	"fmt"
	"reflect"
	"testing"

	"github.com/stretchr/testify/require"
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
	require.Equal(t, "calculator", info.Name)
	require.Len(t, info.Required, 1)
	require.Equal(t, "expression", info.Required[0])

	// Test execution
	call := ToolCall{
		ID:    "test-1",
		Name:  "calculator",
		Input: `{"expression": "2+2"}`,
	}

	result, err := tool.Run(context.Background(), call)
	require.NoError(t, err)
	require.Equal(t, "4", result.Content)
	require.False(t, result.IsError)
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
	require.True(t, ok, "Expected units parameter to exist")
	enumValues, ok := unitsParam["enum"].([]any)
	require.True(t, ok)
	require.Len(t, enumValues, 2)

	// Test execution with enum value
	call := ToolCall{
		ID:    "test-2",
		Name:  "weather",
		Input: `{"location": "San Francisco", "units": "fahrenheit"}`,
	}

	result, err := tool.Run(context.Background(), call)
	require.NoError(t, err)
	require.Contains(t, result.Content, "San Francisco")
	require.Contains(t, result.Content, "72°F")
}

func TestEnumSupport(t *testing.T) {
	// Test enum via struct tags
	type WeatherInput struct {
		Location string `json:"location" description:"City name"`
		Units    string `json:"units" enum:"celsius,fahrenheit,kelvin" description:"Temperature units"`
		Format   string `json:"format,omitempty" enum:"json,xml,text"`
	}

	schema := generateSchema(reflect.TypeOf(WeatherInput{}))

	require.Equal(t, "object", schema.Type)

	// Check units field has enum values
	unitsSchema := schema.Properties["units"]
	require.NotNil(t, unitsSchema, "Expected units property to exist")
	require.Len(t, unitsSchema.Enum, 3)
	expectedUnits := []string{"celsius", "fahrenheit", "kelvin"}
	for i, expected := range expectedUnits {
		require.Equal(t, expected, unitsSchema.Enum[i])
	}

	// Check required fields (format should not be required due to omitempty)
	expectedRequired := []string{"location", "units"}
	require.Len(t, schema.Required, len(expectedRequired))
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
	require.True(t, ok, "Expected name parameter to exist")
	require.Equal(t, "string", nameParam["type"])
	require.Equal(t, "The name field", nameParam["description"])

	// Check age parameter with min/max
	ageParam, ok := params["age"].(map[string]any)
	require.True(t, ok, "Expected age parameter to exist")
	require.Equal(t, "integer", ageParam["type"])
	require.Equal(t, 0.0, ageParam["minimum"])
	require.Equal(t, 120.0, ageParam["maximum"])

	// Check priority parameter with enum
	priorityParam, ok := params["priority"].(map[string]any)
	require.True(t, ok, "Expected priority parameter to exist")
	require.Equal(t, "string", priorityParam["type"])
	enumValues, ok := priorityParam["enum"].([]any)
	require.True(t, ok)
	require.Len(t, enumValues, 3)
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
			require.Equal(t, tt.expected.Type, schema.Type)
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
			require.Equal(t, tt.expected.Type, schema.Type)
			require.NotNil(t, schema.Items, "Expected items schema to exist")
			require.Equal(t, tt.expected.Items.Type, schema.Items.Type)
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
			require.Equal(t, tt.expected, schema.Type)
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
				require.Equal(t, "object", schema.Type)
				require.Len(t, schema.Properties, 2)
				require.NotNil(t, schema.Properties["name"], "Expected name property to exist")
				require.Equal(t, "The name field", schema.Properties["name"].Description)
				require.Len(t, schema.Required, 2)
			},
		},
		{
			name:  "struct with omitempty",
			input: StructWithOmitEmpty{},
			validate: func(t *testing.T, schema Schema) {
				require.Len(t, schema.Required, 1)
				require.Equal(t, "required", schema.Required[0])
			},
		},
		{
			name:  "struct with json ignore",
			input: StructWithJSONIgnore{},
			validate: func(t *testing.T, schema Schema) {
				require.Len(t, schema.Properties, 1)
				require.NotNil(t, schema.Properties["visible"], "Expected visible property to exist")
				require.Nil(t, schema.Properties["hidden"], "Expected hidden property to not exist")
			},
		},
		{
			name:  "struct without json tags",
			input: StructWithoutJSONTags{},
			validate: func(t *testing.T, schema Schema) {
				require.NotNil(t, schema.Properties["first_name"], "Expected first_name property to exist")
				require.NotNil(t, schema.Properties["last_name"], "Expected last_name property to exist")
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

	require.Equal(t, "object", schema.Type)

	require.NotNil(t, schema.Properties["name"], "Expected name property to exist")
	require.Equal(t, "string", schema.Properties["name"].Type)

	require.NotNil(t, schema.Properties["age"], "Expected age property to exist")
	require.Equal(t, "integer", schema.Properties["age"].Type)
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

	require.Equal(t, "object", schema.Type)

	require.NotNil(t, schema.Properties["address"], "Expected address property to exist")

	addressSchema := schema.Properties["address"]
	require.Equal(t, "object", addressSchema.Type)

	require.NotNil(t, addressSchema.Properties["street"], "Expected street property in address to exist")
	require.NotNil(t, addressSchema.Properties["city"], "Expected city property in address to exist")
}

func TestGenerateSchemaRecursiveStructs(t *testing.T) {
	t.Parallel()

	type Node struct {
		Value string `json:"value"`
		Next  *Node  `json:"next,omitempty"`
	}

	schema := generateSchema(reflect.TypeOf(Node{}))

	require.Equal(t, "object", schema.Type)

	require.NotNil(t, schema.Properties["value"], "Expected value property to exist")

	require.NotNil(t, schema.Properties["next"], "Expected next property to exist")

	// The recursive reference should be handled gracefully
	nextSchema := schema.Properties["next"]
	require.Equal(t, "object", nextSchema.Type)
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
	require.NotNil(t, levelSchema, "Expected level property to exist")
	require.Len(t, levelSchema.Enum, 4)
	expectedLevels := []string{"debug", "info", "warn", "error"}
	for i, expected := range expectedLevels {
		require.Equal(t, expected, levelSchema.Enum[i])
	}

	// Check format field
	formatSchema := schema.Properties["format"]
	require.NotNil(t, formatSchema, "Expected format property to exist")
	require.Len(t, formatSchema.Enum, 2)

	// Check required fields (optional should not be required due to omitempty)
	expectedRequired := []string{"level", "format"}
	require.Len(t, schema.Required, len(expectedRequired))
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
	require.NotNil(t, stringSliceSchema, "Expected string_slice property to exist")
	require.Equal(t, "array", stringSliceSchema.Type)
	require.Equal(t, "string", stringSliceSchema.Items.Type)

	// Check int map
	intMapSchema := schema.Properties["int_map"]
	require.NotNil(t, intMapSchema, "Expected int_map property to exist")
	require.Equal(t, "object", intMapSchema.Type)

	// Check nested slice
	nestedSliceSchema := schema.Properties["nested_slice"]
	require.NotNil(t, nestedSliceSchema, "Expected nested_slice property to exist")
	require.Equal(t, "array", nestedSliceSchema.Type)
	require.Equal(t, "object", nestedSliceSchema.Items.Type)

	// Check interface
	interfaceSchema := schema.Properties["interface"]
	require.NotNil(t, interfaceSchema, "Expected interface property to exist")
	require.Equal(t, "object", interfaceSchema.Type)
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
			require.Equal(t, tt.expected, result, "toSnakeCase(%s)", tt.input)
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
			require.Len(t, result, len(tt.expected))
			for key, expectedValue := range tt.expected {
				require.NotNil(t, result[key], "Expected parameter %s to exist", key)
				// Deep comparison would be complex, so we'll check key properties
				resultParam := result[key].(map[string]any)
				expectedParam := expectedValue.(map[string]any)
				for propKey, propValue := range expectedParam {
					require.Equal(t, propValue, resultParam[propKey], "Expected %s.%s", key, propKey)
				}
			}
		})
	}
}
