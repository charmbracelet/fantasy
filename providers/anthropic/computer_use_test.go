package anthropic

import (
	"encoding/base64"
	"errors"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestParseComputerUseInput(t *testing.T) {
	t.Parallel()

	t.Run("screenshot", func(t *testing.T) {
		t.Parallel()
		input, err := ParseComputerUseInput(`{"action":"screenshot"}`)
		require.NoError(t, err)
		require.Equal(t, ActionScreenshot, input.Action)
		require.Equal(t, [2]int64{0, 0}, input.Coordinate)
		require.Equal(t, "", input.Text)
	})

	t.Run("left_click with coordinate", func(t *testing.T) {
		t.Parallel()
		input, err := ParseComputerUseInput(`{"action":"left_click","coordinate":[100,200]}`)
		require.NoError(t, err)
		require.Equal(t, ActionLeftClick, input.Action)
		require.Equal(t, [2]int64{100, 200}, input.Coordinate)
	})

	t.Run("right_click with coordinate", func(t *testing.T) {
		t.Parallel()
		input, err := ParseComputerUseInput(`{"action":"right_click","coordinate":[50,75]}`)
		require.NoError(t, err)
		require.Equal(t, ActionRightClick, input.Action)
		require.Equal(t, [2]int64{50, 75}, input.Coordinate)
	})

	t.Run("double_click with coordinate", func(t *testing.T) {
		t.Parallel()
		input, err := ParseComputerUseInput(`{"action":"double_click","coordinate":[300,400]}`)
		require.NoError(t, err)
		require.Equal(t, ActionDoubleClick, input.Action)
		require.Equal(t, [2]int64{300, 400}, input.Coordinate)
	})

	t.Run("middle_click with coordinate", func(t *testing.T) {
		t.Parallel()
		input, err := ParseComputerUseInput(`{"action":"middle_click","coordinate":[10,20]}`)
		require.NoError(t, err)
		require.Equal(t, ActionMiddleClick, input.Action)
		require.Equal(t, [2]int64{10, 20}, input.Coordinate)
	})

	t.Run("mouse_move with coordinate", func(t *testing.T) {
		t.Parallel()
		input, err := ParseComputerUseInput(`{"action":"mouse_move","coordinate":[500,600]}`)
		require.NoError(t, err)
		require.Equal(t, ActionMouseMove, input.Action)
		require.Equal(t, [2]int64{500, 600}, input.Coordinate)
	})

	t.Run("left_click_drag with start_coordinate and coordinate", func(t *testing.T) {
		t.Parallel()
		input, err := ParseComputerUseInput(`{"action":"left_click_drag","start_coordinate":[10,20],"coordinate":[300,400]}`)
		require.NoError(t, err)
		require.Equal(t, ActionLeftClickDrag, input.Action)
		require.Equal(t, [2]int64{10, 20}, input.StartCoordinate)
		require.Equal(t, [2]int64{300, 400}, input.Coordinate)
	})

	t.Run("type with text", func(t *testing.T) {
		t.Parallel()
		input, err := ParseComputerUseInput(`{"action":"type","text":"hello world"}`)
		require.NoError(t, err)
		require.Equal(t, ActionType, input.Action)
		require.Equal(t, "hello world", input.Text)
	})

	t.Run("key with text", func(t *testing.T) {
		t.Parallel()
		input, err := ParseComputerUseInput(`{"action":"key","text":"ctrl+c"}`)
		require.NoError(t, err)
		require.Equal(t, ActionKey, input.Action)
		require.Equal(t, "ctrl+c", input.Text)
	})

	t.Run("scroll with coordinate direction and amount", func(t *testing.T) {
		t.Parallel()
		input, err := ParseComputerUseInput(`{"action":"scroll","coordinate":[960,540],"scroll_direction":"down","scroll_amount":3}`)
		require.NoError(t, err)
		require.Equal(t, ActionScroll, input.Action)
		require.Equal(t, [2]int64{960, 540}, input.Coordinate)
		require.Equal(t, "down", input.ScrollDirection)
		require.Equal(t, int64(3), input.ScrollAmount)
	})

	t.Run("invalid JSON returns error", func(t *testing.T) {
		t.Parallel()
		_, err := ParseComputerUseInput(`{not valid json}`)
		require.Error(t, err)
	})

	t.Run("triple_click with coordinate", func(t *testing.T) {
		t.Parallel()
		input, err := ParseComputerUseInput(`{"action":"triple_click","coordinate":[120,240]}`)
		require.NoError(t, err)
		require.Equal(t, ActionTripleClick, input.Action)
		require.Equal(t, [2]int64{120, 240}, input.Coordinate)
	})

	t.Run("left_mouse_down with coordinate", func(t *testing.T) {
		t.Parallel()
		input, err := ParseComputerUseInput(`{"action":"left_mouse_down","coordinate":[80,90]}`)
		require.NoError(t, err)
		require.Equal(t, ActionLeftMouseDown, input.Action)
		require.Equal(t, [2]int64{80, 90}, input.Coordinate)
	})

	t.Run("left_mouse_up with coordinate", func(t *testing.T) {
		t.Parallel()
		input, err := ParseComputerUseInput(`{"action":"left_mouse_up","coordinate":[80,90]}`)
		require.NoError(t, err)
		require.Equal(t, ActionLeftMouseUp, input.Action)
		require.Equal(t, [2]int64{80, 90}, input.Coordinate)
	})

	t.Run("wait", func(t *testing.T) {
		t.Parallel()
		input, err := ParseComputerUseInput(`{"action":"wait"}`)
		require.NoError(t, err)
		require.Equal(t, ActionWait, input.Action)
		require.Equal(t, [2]int64{0, 0}, input.Coordinate)
		require.Equal(t, "", input.Text)
	})

	t.Run("zoom with region", func(t *testing.T) {
		t.Parallel()
		input, err := ParseComputerUseInput(`{"action":"zoom","region":[100,200,500,600]}`)
		require.NoError(t, err)
		require.Equal(t, ActionZoom, input.Action)
		require.Equal(t, [4]int64{100, 200, 500, 600}, input.Region)
	})

	t.Run("left_click with modifier key", func(t *testing.T) {
		t.Parallel()
		input, err := ParseComputerUseInput(`{"action":"left_click","coordinate":[100,200],"text":"shift"}`)
		require.NoError(t, err)
		require.Equal(t, ActionLeftClick, input.Action)
		require.Equal(t, [2]int64{100, 200}, input.Coordinate)
		require.Equal(t, "shift", input.Text)
	})

	t.Run("unknown action parses without error", func(t *testing.T) {
		t.Parallel()
		input, err := ParseComputerUseInput(`{"action":"future_action","coordinate":[1,2]}`)
		require.NoError(t, err)
		require.Equal(t, ComputerAction("future_action"), input.Action)
		require.Equal(t, [2]int64{1, 2}, input.Coordinate)
	})

	t.Run("hold_key with duration", func(t *testing.T) {
		t.Parallel()
		input, err := ParseComputerUseInput(`{"action":"hold_key","text":"shift","duration":2}`)
		require.NoError(t, err)
		require.Equal(t, ActionHoldKey, input.Action)
		require.Equal(t, "shift", input.Text)
		require.Equal(t, int64(2), input.Duration)
	})
}

func TestNewComputerUseScreenshotResult(t *testing.T) {
	t.Parallel()

	t.Run("base64 encodes PNG bytes", func(t *testing.T) {
		t.Parallel()
		pngData := []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A}
		result := NewComputerUseScreenshotResult("call-123", pngData)

		require.Equal(t, "call-123", result.ToolCallID)

		media, ok := result.Output.(fantasy.ToolResultOutputContentMedia)
		require.True(t, ok, "output should be ToolResultOutputContentMedia")
		require.Equal(t, "image/png", media.MediaType)
		require.Equal(t, base64.StdEncoding.EncodeToString(pngData), media.Data)
	})

	t.Run("preserves tool call ID", func(t *testing.T) {
		t.Parallel()
		result := NewComputerUseScreenshotResult("tc_abc", []byte{0x01})
		require.Equal(t, "tc_abc", result.ToolCallID)
	})

	t.Run("empty screenshot bytes", func(t *testing.T) {
		t.Parallel()
		result := NewComputerUseScreenshotResult("call-empty", []byte{})

		media, ok := result.Output.(fantasy.ToolResultOutputContentMedia)
		require.True(t, ok)
		require.Equal(t, "image/png", media.MediaType)
		require.Equal(t, "", media.Data)
	})

	t.Run("output content type is media", func(t *testing.T) {
		t.Parallel()
		result := NewComputerUseScreenshotResult("call-type", []byte{0xFF})
		require.Equal(t, fantasy.ToolResultContentTypeMedia, result.Output.GetType())
	})
}

func TestNewComputerUseScreenshotResultWithMediaType(t *testing.T) {
	t.Parallel()

	t.Run("custom media type and base64 data", func(t *testing.T) {
		t.Parallel()
		b64 := base64.StdEncoding.EncodeToString([]byte("jpeg-data"))
		result := NewComputerUseScreenshotResultWithMediaType("call-456", b64, "image/jpeg")

		require.Equal(t, "call-456", result.ToolCallID)

		media, ok := result.Output.(fantasy.ToolResultOutputContentMedia)
		require.True(t, ok, "output should be ToolResultOutputContentMedia")
		require.Equal(t, "image/jpeg", media.MediaType)
		require.Equal(t, b64, media.Data)
	})

	t.Run("preserves tool call ID", func(t *testing.T) {
		t.Parallel()
		result := NewComputerUseScreenshotResultWithMediaType("tc_xyz", "data", "image/webp")
		require.Equal(t, "tc_xyz", result.ToolCallID)
	})

	t.Run("output content type is media", func(t *testing.T) {
		t.Parallel()
		result := NewComputerUseScreenshotResultWithMediaType("call-type", "data", "image/png")
		require.Equal(t, fantasy.ToolResultContentTypeMedia, result.Output.GetType())
	})
}

func TestNewComputerUseErrorResult(t *testing.T) {
	t.Parallel()

	t.Run("error message propagates", func(t *testing.T) {
		t.Parallel()
		err := errors.New("screenshot capture failed")
		result := NewComputerUseErrorResult("call-err", err)

		require.Equal(t, "call-err", result.ToolCallID)

		errOutput, ok := result.Output.(fantasy.ToolResultOutputContentError)
		require.True(t, ok, "output should be ToolResultOutputContentError")
		require.Equal(t, "screenshot capture failed", errOutput.Error.Error())
	})

	t.Run("preserves tool call ID", func(t *testing.T) {
		t.Parallel()
		result := NewComputerUseErrorResult("tc_err", errors.New("fail"))
		require.Equal(t, "tc_err", result.ToolCallID)
	})

	t.Run("output content type is error", func(t *testing.T) {
		t.Parallel()
		result := NewComputerUseErrorResult("call-type", errors.New("oops"))
		require.Equal(t, fantasy.ToolResultContentTypeError, result.Output.GetType())
	})
}

func TestNewComputerUseTextResult(t *testing.T) {
	t.Parallel()

	t.Run("text content is set", func(t *testing.T) {
		t.Parallel()
		result := NewComputerUseTextResult("call-txt", "action completed successfully")

		require.Equal(t, "call-txt", result.ToolCallID)

		textOutput, ok := result.Output.(fantasy.ToolResultOutputContentText)
		require.True(t, ok, "output should be ToolResultOutputContentText")
		require.Equal(t, "action completed successfully", textOutput.Text)
	})

	t.Run("preserves tool call ID", func(t *testing.T) {
		t.Parallel()
		result := NewComputerUseTextResult("tc_text", "hello")
		require.Equal(t, "tc_text", result.ToolCallID)
	})

	t.Run("empty text", func(t *testing.T) {
		t.Parallel()
		result := NewComputerUseTextResult("call-empty", "")

		textOutput, ok := result.Output.(fantasy.ToolResultOutputContentText)
		require.True(t, ok)
		require.Equal(t, "", textOutput.Text)
	})

	t.Run("output content type is text", func(t *testing.T) {
		t.Parallel()
		result := NewComputerUseTextResult("call-type", "test")
		require.Equal(t, fantasy.ToolResultContentTypeText, result.Output.GetType())
	})
}
