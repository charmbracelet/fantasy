package google

import (
	"reflect"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/schema"
	"github.com/stretchr/testify/require"
)

func TestToGoogleToolsKeepsEnumAndNestedRequired(t *testing.T) {
	t.Parallel()

	type filter struct {
		Field string `json:"field"`
		Op    string `json:"op" enum:"eq,ne"`
		Value string `json:"value,omitempty"`
	}
	type input struct {
		Unit    string   `json:"unit" enum:"celsius,fahrenheit"`
		Filters []filter `json:"filters"`
	}
	s := schema.Generate(reflect.TypeFor[input]())

	tools, _, warnings := toGoogleTools([]fantasy.Tool{fantasy.FunctionTool{
		Name: "search",
		InputSchema: map[string]any{
			"type":       "object",
			"properties": schema.ToParameters(s),
			"required":   s.Required,
		},
	}}, nil)

	require.Empty(t, warnings)
	require.Len(t, tools, 1)
	params := tools[0].Parameters
	require.Equal(t, []string{"celsius", "fahrenheit"}, params.Properties["unit"].Enum)
	item := params.Properties["filters"].Items
	require.Equal(t, []string{"eq", "ne"}, item.Properties["op"].Enum)
	require.Equal(t, []string{"field", "op"}, item.Required)
}

func TestToGoogleToolsDecodedJSONSchema(t *testing.T) {
	t.Parallel()

	// Schemas decoded from JSON (e.g. MCP tools) use []any for arrays.
	tools, _, _ := toGoogleTools([]fantasy.Tool{fantasy.FunctionTool{
		Name: "search",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"opts": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"mode":  map[string]any{"type": "string", "enum": []any{"fast", "slow"}},
						"level": map[string]any{"type": "integer", "enum": []any{float64(1), float64(2)}},
						"maybe": map[string]any{"type": "string", "enum": []any{"a", nil}},
					},
					"required": []any{"mode", "unknown"},
				},
			},
			"required": []any{"opts"},
		},
	}}, nil)

	require.Len(t, tools, 1)
	params := tools[0].Parameters
	require.Equal(t, []string{"opts"}, params.Required)
	opts := params.Properties["opts"]
	require.Equal(t, []string{"mode"}, opts.Required)
	require.Equal(t, []string{"fast", "slow"}, opts.Properties["mode"].Enum)
	// Gemini only accepts enum on strings, and values cannot be dropped.
	require.Empty(t, opts.Properties["level"].Enum)
	require.Empty(t, opts.Properties["maybe"].Enum)
}
