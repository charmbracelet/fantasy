package google

import (
	"encoding/base64"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestToGooglePrompt_ToolResultMedia(t *testing.T) {
	t.Parallel()

	png := []byte("\x89PNG fake")
	media := func(id, mediaType, data, text string) fantasy.ToolResultPart {
		return fantasy.ToolResultPart{
			ToolCallID: id,
			Output: fantasy.ToolResultOutputContentMedia{
				Data:      data,
				MediaType: mediaType,
				Text:      text,
			},
		}
	}
	prompt := func(results ...fantasy.MessagePart) fantasy.Prompt {
		return fantasy.Prompt{
			{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{
				fantasy.ToolCallPart{ToolCallID: "call_1", ToolName: "screenshot", Input: "{}"},
				fantasy.ToolCallPart{ToolCallID: "call_2", ToolName: "screenshot", Input: "{}"},
			}},
			{Role: fantasy.MessageRoleTool, Content: results},
		}
	}
	encoded := base64.StdEncoding.EncodeToString(png)

	t.Run("image is attached after all function responses", func(t *testing.T) {
		t.Parallel()

		_, content, warnings := toGooglePrompt(prompt(
			media("call_1", "image/png", encoded, "the screen"),
			media("call_2", "image/png", encoded, ""),
		), false)

		require.Empty(t, warnings)
		require.Len(t, content, 2)
		parts := content[1].Parts
		require.Len(t, parts, 4)
		require.Equal(t, "call_1", parts[0].FunctionResponse.ID)
		require.Equal(t, "screenshot", parts[0].FunctionResponse.Name)
		require.Equal(t, "the screen", parts[0].FunctionResponse.Response["result"])
		require.Equal(t, "call_2", parts[1].FunctionResponse.ID)
		require.NotEmpty(t, parts[1].FunctionResponse.Response["result"])
		for _, part := range parts[2:] {
			require.Equal(t, "image/png", part.InlineData.MIMEType)
			require.Equal(t, png, part.InlineData.Data)
		}
	})

	for name, result := range map[string]fantasy.ToolResultPart{
		"unsupported media type": media("call_1", "application/zip", encoded, ""),
		"invalid base64":         media("call_1", "image/png", "%%%", ""),
	} {
		t.Run(name+" still answers the call and warns", func(t *testing.T) {
			t.Parallel()

			_, content, warnings := toGooglePrompt(prompt(result), true)

			require.Len(t, warnings, 1)
			require.Len(t, content, 2)
			require.Len(t, content[1].Parts, 1)
			require.Empty(t, content[1].Parts[0].FunctionResponse.ID)
			require.NotEmpty(t, content[1].Parts[0].FunctionResponse.Response["result"])
		})
	}
}
