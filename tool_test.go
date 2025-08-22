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
	tool := NewTypedToolFunc(
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
	tool := NewTypedToolFunc(
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
