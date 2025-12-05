package fantasy

import (
	"context"
	"fmt"
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

func TestNewImageResponse(t *testing.T) {
	imageData := []byte{0x89, 0x50, 0x4E, 0x47} // PNG header bytes
	mediaType := "image/png"

	resp := NewImageResponse(imageData, mediaType)

	require.Equal(t, "image", resp.Type)
	require.Equal(t, imageData, resp.Data)
	require.Equal(t, mediaType, resp.MediaType)
	require.False(t, resp.IsError)
	require.Empty(t, resp.Content)
}

func TestNewMediaResponse(t *testing.T) {
	audioData := []byte{0x52, 0x49, 0x46, 0x46} // RIFF header bytes
	mediaType := "audio/wav"

	resp := NewMediaResponse(audioData, mediaType)

	require.Equal(t, "media", resp.Type)
	require.Equal(t, audioData, resp.Data)
	require.Equal(t, mediaType, resp.MediaType)
	require.False(t, resp.IsError)
	require.Empty(t, resp.Content)
}
