package google

import (
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestToGooglePromptKeepsToolCallsWithBadInput(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		input       string
		wantWarning bool
	}{
		{name: "malformed input", input: "{not-json", wantWarning: true},
		{name: "empty input", input: ""},
		{name: "whitespace input", input: "  "},
		{name: "null input", input: "null"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			prompt := fantasy.Prompt{
				{
					Role:    fantasy.MessageRoleUser,
					Content: []fantasy.MessagePart{fantasy.TextPart{Text: "Hi"}},
				},
				{
					Role: fantasy.MessageRoleAssistant,
					Content: []fantasy.MessagePart{
						fantasy.ToolCallPart{
							ToolCallID: "call_123",
							ToolName:   "get_weather",
							Input:      tt.input,
						},
					},
				},
				{
					Role: fantasy.MessageRoleTool,
					Content: []fantasy.MessagePart{
						fantasy.ToolResultPart{
							ToolCallID: "call_123",
							Output:     fantasy.ToolResultOutputContentText{Text: "sunny"},
						},
					},
				},
			}

			_, contents, warnings := toGooglePrompt(prompt, false)

			// The call must stay in the model turn so the function response
			// that follows has a call to pair with.
			require.Len(t, contents, 3)
			require.Len(t, contents[1].Parts, 1)
			call := contents[1].Parts[0].FunctionCall
			require.NotNil(t, call, "function call must be kept")
			require.Equal(t, "call_123", call.ID)
			require.Equal(t, "get_weather", call.Name)
			require.Empty(t, call.Args)
			require.NotNil(t, call.Args)
			require.Len(t, contents[2].Parts, 1)
			require.NotNil(t, contents[2].Parts[0].FunctionResponse)

			if tt.wantWarning {
				require.Len(t, warnings, 1)
				require.Equal(t, fantasy.CallWarningTypeOther, warnings[0].Type)
				require.Contains(t, warnings[0].Message, "malformed input JSON")
				require.Contains(t, warnings[0].Message, "call_123")
			} else {
				require.Empty(t, warnings)
			}
		})
	}
}

func TestToGooglePromptKeepsValidToolCallInput(t *testing.T) {
	t.Parallel()

	prompt := fantasy.Prompt{
		{
			Role: fantasy.MessageRoleAssistant,
			Content: []fantasy.MessagePart{
				fantasy.ToolCallPart{
					ToolCallID: "call_1",
					ToolName:   "get_weather",
					Input:      `{"city":"Paris"}`,
				},
			},
		},
	}

	_, contents, warnings := toGooglePrompt(prompt, false)

	require.Empty(t, warnings)
	require.Len(t, contents, 1)
	require.Equal(t, map[string]any{"city": "Paris"}, contents[0].Parts[0].FunctionCall.Args)
}
