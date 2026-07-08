package main

// This example demonstrates Anthropic computer use with the agent
// helper. It shows how to:
//
//  1. Wire up the provider, model, and computer use tool.
//  2. Register the tool via WithProviderDefinedTools so the agent
//     handles the tool-call loop automatically.
//  3. Parse incoming tool calls with ParseComputerUseInput inside
//     the Run function.
//  4. Return results (screenshots, errors) back to the agent.

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/anthropic"
)

// takeScreenshot is a stub that simulates capturing a screenshot.
// In a real implementation this would capture the virtual display
// and return raw PNG bytes.
func takeScreenshot() ([]byte, error) {
	// Generate a valid 1x1 black PNG as a placeholder.
	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	img.Set(0, 0, color.Black)
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func main() {
	// Set up the Anthropic provider.
	provider, err := anthropic.New(anthropic.WithAPIKey(os.Getenv("ANTHROPIC_API_KEY")))
	if err != nil {
		fmt.Fprintln(os.Stderr, "could not create provider:", err)
		os.Exit(1)
	}

	ctx := context.Background()

	// Pick the model.
	model, err := provider.LanguageModel(ctx, "claude-opus-4-6")
	if err != nil {
		fmt.Fprintln(os.Stderr, "could not get language model:", err)
		os.Exit(1)
	}

	// Create a computer use tool with a Run function that executes
	// actions and returns screenshots.
	computerTool := anthropic.NewComputerUseTool(anthropic.ComputerUseToolOptions{
		DisplayWidthPx:  1920,
		DisplayHeightPx: 1080,
		ToolVersion:     anthropic.ComputerUse20251124,
	}, func(ctx context.Context, call fantasy.ToolCall) (fantasy.ToolResponse, error) {
		action, err := anthropic.ParseComputerUseInput(call.Input)
		if err != nil {
			return fantasy.ToolResponse{}, fmt.Errorf("parse computer use input: %w", err)
		}

		fmt.Printf("Action: %s\n", action.Action)

		// In production you would execute the action (click,
		// type, scroll, etc.) against the virtual display and
		// then capture a screenshot.
		png, err := takeScreenshot()
		if err != nil {
			return fantasy.ToolResponse{}, fmt.Errorf("take screenshot: %w", err)
		}
		return fantasy.NewImageResponse(png, "image/png"), nil
	})

	// Build an agent with the computer use tool. The agent handles
	// the tool-call loop: it sends the prompt, executes any tool
	// calls the model returns, feeds the results back, and repeats
	// until the model stops requesting tools.
	agent := fantasy.NewAgent(model,
		fantasy.WithProviderDefinedTools(computerTool),
		fantasy.WithStopConditions(fantasy.StepCountIs(10)),
	)

	result, err := agent.Generate(ctx, fantasy.AgentCall{
		Prompt: "Take a screenshot of the desktop",
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "agent error:", err)
		os.Exit(1)
	}

	fmt.Println("Agent finished.")
	fmt.Printf("Steps: %d\n", len(result.Steps))
	if text := result.Response.Content.Text(); text != "" {
		fmt.Println("Claude said:", text)
	}
}
