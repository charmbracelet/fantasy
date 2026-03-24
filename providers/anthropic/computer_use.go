package anthropic

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"

	"charm.land/fantasy"
	anthropicsdk "github.com/charmbracelet/anthropic-sdk-go"
	"github.com/charmbracelet/anthropic-sdk-go/packages/param"
)

// computerUseToolID is the canonical identifier for
// Anthropic computer use tools. It follows the
// <provider>.<tool> convention used by ProviderDefinedTool.ID.
const computerUseToolID = "anthropic.computer"

// computerUseAPIName is the tool name Anthropic's API expects
// on the wire.
const computerUseAPIName = "computer"

// ComputerUseToolVersion identifies which version of the Anthropic
// computer use tool to use.
type ComputerUseToolVersion string

const (
	// ComputerUse20251124 selects the November 2025 version of the
	// computer use tool.
	ComputerUse20251124 ComputerUseToolVersion = "computer_20251124"
	// ComputerUse20250124 selects the January 2025 version of the
	// computer use tool.
	ComputerUse20250124 ComputerUseToolVersion = "computer_20250124"
)

// ComputerUseToolOptions holds the configuration for creating a
// computer use tool instance.
type ComputerUseToolOptions struct {
	// DisplayWidthPx is the width of the display in pixels.
	DisplayWidthPx int64
	// DisplayHeightPx is the height of the display in pixels.
	DisplayHeightPx int64
	// DisplayNumber is an optional X11 display number.
	DisplayNumber *int64
	// EnableZoom enables zoom support. Only used with the
	// ComputerUse20251124 version.
	EnableZoom *bool
	// ToolVersion selects which computer use tool version to use.
	ToolVersion ComputerUseToolVersion
	// CacheControl sets optional cache control for the tool.
	CacheControl *CacheControl
}

// NewComputerUseTool creates a new provider-defined tool configured
// for Anthropic computer use. The returned tool can be passed
// directly into a fantasy tool set via WithProviderDefinedTools.
func NewComputerUseTool(
	opts ComputerUseToolOptions,
	run func(ctx context.Context, call fantasy.ToolCall) (fantasy.ToolResponse, error),
) fantasy.ExecutableProviderTool {
	args := map[string]any{
		"display_width_px":  opts.DisplayWidthPx,
		"display_height_px": opts.DisplayHeightPx,
		"tool_version":      string(opts.ToolVersion),
	}
	if opts.DisplayNumber != nil {
		args["display_number"] = *opts.DisplayNumber
	}
	if opts.EnableZoom != nil {
		args["enable_zoom"] = *opts.EnableZoom
	}
	if opts.CacheControl != nil {
		args["cache_control"] = *opts.CacheControl
	}
	pdt := fantasy.ProviderDefinedTool{
		ID:   computerUseToolID,
		Name: computerUseAPIName,
		Args: args,
	}
	return fantasy.NewExecutableProviderTool(pdt, run)
}

// IsComputerUseTool reports whether tool is an Anthropic computer
// use tool. It checks for a ProviderDefinedTool whose ID matches
// the computer use tool identifier exactly.
func IsComputerUseTool(tool fantasy.Tool) bool {
	pdt, ok := asProviderDefinedTool(tool)
	if !ok {
		return false
	}
	return pdt.ID == computerUseToolID
}

// getComputerUseVersion extracts the ComputerUseToolVersion from a
// provider-defined tool's Args map. It returns the version and true
// if present, or the zero value and false otherwise.
func getComputerUseVersion(tool fantasy.ProviderDefinedTool) (ComputerUseToolVersion, bool) {
	v, ok := tool.Args["tool_version"]
	if !ok {
		return "", false
	}
	s, ok := v.(string)
	if !ok {
		return "", false
	}
	return ComputerUseToolVersion(s), true
}

// computerUseBetaFlag returns the Anthropic beta header value for
// the given computer use tool version.
func computerUseBetaFlag(version ComputerUseToolVersion) (string, error) {
	switch version {
	case ComputerUse20251124:
		// TODO: Replace with SDK constant when available.
		return "computer-use-2025-11-24", nil
	case ComputerUse20250124:
		return anthropicsdk.AnthropicBetaComputerUse2025_01_24, nil
	default:
		return "", fmt.Errorf(
			"unsupported computer use tool version: %q", version,
		)
	}
}

// computerUseToolJSON builds the JSON representation of a computer
// use tool from a ProviderDefinedTool's Args, using the beta SDK
// types for serialization.
func computerUseToolJSON(pdt fantasy.ProviderDefinedTool) (json.RawMessage, error) {
	version, ok := getComputerUseVersion(pdt)
	if !ok {
		return nil, fmt.Errorf("computerUseToolJSON: tool_version arg is missing")
	}

	h, hOK := anyToInt64(pdt.Args["display_height_px"])
	w, wOK := anyToInt64(pdt.Args["display_width_px"])
	if !hOK || !wOK {
		return nil, fmt.Errorf(
			"display_height_px and display_width_px must be numeric"+
				" (height ok=%t, width ok=%t)", hOK, wOK,
		)
	}

	switch version {
	case ComputerUse20250124:
		tool := anthropicsdk.BetaToolUnionParamOfComputerUseTool20250124(h, w)
		if v, ok := pdt.Args["display_number"]; ok {
			dn, ok := anyToInt64(v)
			if !ok {
				return nil, fmt.Errorf("computer use tool has invalid display_number")
			}
			tool.OfComputerUseTool20250124.DisplayNumber = param.NewOpt(dn)
		}
		if _, ok := pdt.Args["cache_control"]; ok {
			tool.OfComputerUseTool20250124.CacheControl = anthropicsdk.NewBetaCacheControlEphemeralParam()
		}
		return json.Marshal(tool)
	case ComputerUse20251124:
		tool := anthropicsdk.BetaToolUnionParamOfComputerUseTool20251124(h, w)
		if v, ok := pdt.Args["display_number"]; ok {
			dn, ok := anyToInt64(v)
			if !ok {
				return nil, fmt.Errorf("computer use tool has invalid display_number")
			}
			tool.OfComputerUseTool20251124.DisplayNumber = param.NewOpt(dn)
		}
		if v, ok := pdt.Args["enable_zoom"]; ok {
			if b, ok := v.(bool); ok {
				tool.OfComputerUseTool20251124.EnableZoom = param.NewOpt(b)
			}
		}
		if _, ok := pdt.Args["cache_control"]; ok {
			tool.OfComputerUseTool20251124.CacheControl = anthropicsdk.NewBetaCacheControlEphemeralParam()
		}
		return json.Marshal(tool)
	default:
		return nil, fmt.Errorf(
			"unsupported computer use tool version: %q", version,
		)
	}
}

// ComputerAction identifies the action Claude wants to perform.
//
// Unless noted otherwise on a specific action, respond by returning a
// screenshot using NewComputerUseScreenshotResult.
type ComputerAction string

const (
	// ActionScreenshot captures the current screen.
	//
	// No additional fields are populated.
	ActionScreenshot ComputerAction = "screenshot"
	// ActionLeftClick performs a left click.
	//
	//   - Coordinate: [x, y] target.
	//   - Text: optional modifier key (e.g. "shift", "ctrl").
	ActionLeftClick ComputerAction = "left_click"
	// ActionRightClick performs a right click (v20250124+).
	//
	//   - Coordinate: [x, y] target.
	//   - Text: optional modifier key (e.g. "shift", "ctrl").
	ActionRightClick ComputerAction = "right_click"
	// ActionDoubleClick performs a double click (v20250124+).
	//
	//   - Coordinate: [x, y] target.
	//   - Text: optional modifier key (e.g. "shift", "ctrl").
	ActionDoubleClick ComputerAction = "double_click"
	// ActionTripleClick performs a triple click (v20250124+).
	//
	//   - Coordinate: [x, y] target.
	//   - Text: optional modifier key (e.g. "shift", "ctrl").
	ActionTripleClick ComputerAction = "triple_click"
	// ActionMiddleClick performs a middle click (v20250124+).
	//
	//   - Coordinate: [x, y] target.
	//   - Text: optional modifier key (e.g. "shift", "ctrl").
	ActionMiddleClick ComputerAction = "middle_click"
	// ActionMouseMove moves the cursor.
	//
	//   - Coordinate: [x, y] destination.
	ActionMouseMove ComputerAction = "mouse_move"
	// ActionLeftClickDrag drags from one point to another
	// (v20250124+).
	//
	//   - StartCoordinate: [x, y] drag origin.
	//   - Coordinate: [x, y] drag destination.
	ActionLeftClickDrag ComputerAction = "left_click_drag"
	// ActionType types text.
	//
	//   - Text: the string to type.
	ActionType ComputerAction = "type"
	// ActionKey presses a key combination.
	//
	//   - Text: key combo string (e.g. "ctrl+c", "Return").
	ActionKey ComputerAction = "key"
	// ActionScroll scrolls the screen (v20250124+).
	//
	//   - Coordinate: [x, y] scroll origin.
	//   - ScrollDirection: "up", "down", "left", or "right".
	//   - ScrollAmount: scroll distance.
	//   - Text: optional modifier key.
	ActionScroll ComputerAction = "scroll"
	// ActionLeftMouseDown presses and holds the left mouse button
	// (v20250124+).
	//
	//   - Coordinate: [x, y] target.
	ActionLeftMouseDown ComputerAction = "left_mouse_down"
	// ActionLeftMouseUp releases the left mouse button
	// (v20250124+).
	//
	//   - Coordinate: [x, y] target.
	ActionLeftMouseUp ComputerAction = "left_mouse_up"
	// ActionHoldKey holds down a key for a specified duration
	// (v20250124+).
	//
	//   - Text: the key to hold.
	//   - Duration: hold time in seconds.
	ActionHoldKey ComputerAction = "hold_key"
	// ActionWait pauses between actions (v20250124+).
	//
	// No additional fields are populated.
	ActionWait ComputerAction = "wait"
	// ActionZoom views a specific screen region at full
	// resolution (v20251124 only). Requires enable_zoom in the
	// tool definition.
	//
	//   - Region: [x1, y1, x2, y2] top-left and bottom-right.
	//
	// Response: return a screenshot of the zoomed region at
	// full resolution.
	ActionZoom ComputerAction = "zoom"
)

// ComputerUseInput is the parsed, typed representation of a computer
// use tool call's Input JSON. Not all fields are populated for every
// action — check Action first, then read the relevant fields.
type ComputerUseInput struct {
	Action ComputerAction `json:"action"`
	// Coordinate is [x, y] for click, move, scroll, and
	// drag-end actions.
	Coordinate [2]int64 `json:"coordinate,omitempty"`
	// StartCoordinate is [x, y] for left_click_drag start point.
	StartCoordinate [2]int64 `json:"start_coordinate,omitempty"`
	// Text is the string to type (ActionType), key combo
	// (ActionKey), modifier key for click/scroll actions, or key
	// to hold (ActionHoldKey).
	Text string `json:"text,omitempty"`
	// ScrollDirection is the scroll direction: "up", "down",
	// "left", or "right".
	ScrollDirection string `json:"scroll_direction,omitempty"`
	// ScrollAmount is the number of scroll clicks.
	ScrollAmount int64 `json:"scroll_amount,omitempty"`
	// Duration is how long to hold the key in seconds
	// (ActionHoldKey).
	Duration int64 `json:"duration,omitempty"`
	// Region is [x1, y1, x2, y2] defining the zoom area
	// (ActionZoom, v20251124 only).
	Region [4]int64 `json:"region,omitempty"`
}

// ParseComputerUseInput parses a ToolCallContent's Input string into
// a typed ComputerUseInput. Returns an error if the JSON is invalid
// or if coordinate arrays have the wrong number of elements.
func ParseComputerUseInput(input string) (ComputerUseInput, error) {
	var result ComputerUseInput
	if err := json.Unmarshal([]byte(input), &result); err != nil {
		return result, err
	}

	// Validate array field lengths. json.Unmarshal silently pads
	// or truncates arrays that don't match the Go fixed-size type,
	// which would produce wrong coordinates.
	var raw map[string]json.RawMessage
	if err := json.Unmarshal([]byte(input), &raw); err != nil {
		return result, err
	}
	if err := validateArrayLen(raw, "coordinate", 2); err != nil {
		return ComputerUseInput{}, err
	}
	if err := validateArrayLen(raw, "start_coordinate", 2); err != nil {
		return ComputerUseInput{}, err
	}
	if err := validateArrayLen(raw, "region", 4); err != nil {
		return ComputerUseInput{}, err
	}

	return result, nil
}

// validateArrayLen checks that the JSON array at key has exactly
// wantLen elements. If the key is absent from raw it returns nil.
func validateArrayLen(raw map[string]json.RawMessage, key string, wantLen int) error {
	v, ok := raw[key]
	if !ok {
		return nil
	}
	var elems []json.RawMessage
	if err := json.Unmarshal(v, &elems); err != nil {
		return fmt.Errorf("%s: expected array: %w", key, err)
	}
	if len(elems) != wantLen {
		return fmt.Errorf(
			"%s: expected %d elements, got %d",
			key, wantLen, len(elems),
		)
	}
	return nil
}

// NewComputerUseScreenshotResult constructs a ToolResultPart
// containing a screenshot image. This is the standard response for
// almost every computer use action — Claude expects to see what
// happened after executing the action.
//
// Parameters:
//   - toolCallID: the ToolCallID from the ToolCallContent that
//     requested this action.
//   - screenshotPNG: the raw PNG bytes of the screenshot. The
//     caller is responsible for capturing and (optionally) resizing
//     the screenshot before passing it here.
//
// The function base64-encodes the image data and sets the media
// type to "image/png".
func NewComputerUseScreenshotResult(
	toolCallID string,
	screenshotPNG []byte,
) fantasy.ToolResultPart {
	return fantasy.ToolResultPart{
		ToolCallID: toolCallID,
		Output: fantasy.ToolResultOutputContentMedia{
			Data:      base64.StdEncoding.EncodeToString(screenshotPNG),
			MediaType: "image/png",
		},
	}
}

// NewComputerUseScreenshotResultWithMediaType is like
// NewComputerUseScreenshotResult but allows specifying a custom
// media type (e.g. "image/jpeg") and pre-encoded base64 data.
func NewComputerUseScreenshotResultWithMediaType(
	toolCallID string,
	base64Data string,
	mediaType string,
) fantasy.ToolResultPart {
	return fantasy.ToolResultPart{
		ToolCallID: toolCallID,
		Output: fantasy.ToolResultOutputContentMedia{
			Data:      base64Data,
			MediaType: mediaType,
		},
	}
}

// NewComputerUseErrorResult constructs a ToolResultPart indicating
// that the requested action failed. Claude will see this as an
// error and may retry or adjust its approach.
//
// Use this when screenshot capture fails, coordinates are out of
// bounds, the application is unresponsive, or any other execution
// error occurs.
func NewComputerUseErrorResult(
	toolCallID string,
	err error,
) fantasy.ToolResultPart {
	return fantasy.ToolResultPart{
		ToolCallID: toolCallID,
		Output: fantasy.ToolResultOutputContentError{
			Error: err,
		},
	}
}

// NewComputerUseTextResult constructs a ToolResultPart containing a
// plain text response. This is rarely needed for computer use —
// most actions should return a screenshot — but can be useful for
// returning metadata alongside the action or for testing.
func NewComputerUseTextResult(
	toolCallID string,
	text string,
) fantasy.ToolResultPart {
	return fantasy.ToolResultPart{
		ToolCallID: toolCallID,
		Output: fantasy.ToolResultOutputContentText{
			Text: text,
		},
	}
}
