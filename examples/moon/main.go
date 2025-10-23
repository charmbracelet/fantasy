package main

// This example demonstrates how to create an agent with a tool that can
// provide moon phase information for a given date, defaulting to today.

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/anthropic"
	"github.com/charmbracelet/lipgloss/v2"
	"github.com/charmbracelet/lipgloss/v2/table"
	"github.com/charmbracelet/log/v2"
	"github.com/charmbracelet/x/term"
)

const systemPrompt = `
You are a quiet assistant named Francine who knows about celestial bodies. You
use as few words as possible, but always reply in full, proper sentences.
You never reply with markdown. You always respond with weekdays.

You also have a cat named Federico who you think performs the actual queries,
and you give him credit in your responses whenever you use the moon phase tool.
`

func main() {
	// We'll use Anthropic for this example.
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		log.Fatal("missing ANTHROPIC_API_KEY")
	}

	// Specifically, we'll use Claude Haiku 4.5.
	provider, err := anthropic.New(anthropic.WithAPIKey(apiKey))
	if err != nil {
		log.Fatalf("could not create Anthropic provider: %v", err)
	}

	ctx := context.Background()

	// Choose the model.
	model, err := provider.LanguageModel(ctx, "claude-haiku-4-5-20251001")
	if err != nil {
		log.Fatalf("could not get language model: %v", err)
	}

	// Add a moon phase tool.
	moonTool := fantasy.NewAgentTool(
		"moon_phase",
		"Get information about the moon phase",
		moonPhaseTool,
	)

	// Create the agent.
	agent := fantasy.NewAgent(
		model,
		fantasy.WithSystemPrompt(systemPrompt),
		fantasy.WithTools(moonTool),
	)

	// Here's our prompt.
	prompt := fmt.Sprintf(
		"What is the moon phase today? And what will it be on December 31 this year? Today's date is %s.",
		time.Now().Format("January 2, 2006"),
	)
	fmt.Println("\n" + formatText(prompt))

	// Let's go! Ask the agent to generate a response.
	result, err := agent.Generate(ctx, fantasy.AgentCall{Prompt: prompt})
	if err != nil {
		log.Fatalf("agent generation failed: %v", err)
	}

	// Print out the final response.
	fmt.Println(formatText(result.Response.Content.Text()))

	// Print out usage statistics.
	t := table.New().
		StyleFunc(func(row, col int) lipgloss.Style {
			return lipgloss.NewStyle().Padding(0, 1)
		}).
		Row("Tokens in", fmt.Sprint(result.TotalUsage.InputTokens)).
		Row("Tokens out", fmt.Sprint(result.TotalUsage.OutputTokens)).
		Row("Steps", fmt.Sprint(len(result.Steps)))
	fmt.Print(lipgloss.NewStyle().MarginLeft(3).Render(t.String()), "\n\n")
}

// Input for the moon phase tool. The model will provide the date when
// necessary.
type moonPhaseInput struct {
	Date string `json:"date,omitempty" description:"Optional date in YYYY-MM-DD; if omitted, use today"`
}

// This is the moon phase tool definition. It queries wttr.in for the moon
// phase on a given date. If no date is provided, it uses today's date.
//
// The date format should be in YYYY-MM-DD format.
func moonPhaseTool(ctx context.Context, input moonPhaseInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
	url := "https://wttr.in/moon?T&q"

	// Validate date format if provided, and update the URL accordingly.
	if input.Date != "" {
		if _, timeErr := time.Parse("2006-01-02", input.Date); timeErr != nil {
			return fantasy.NewTextErrorResponse("invalid date format; use YYYY-MM-DD"), nil
		}
		url = "https://wttr.in/moon@" + input.Date + "?T&q"
	}

	// Prepare an HTTP request.
	req, reqErr := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if reqErr != nil {
		return fantasy.NewTextErrorResponse("failed to build request: " + reqErr.Error()), nil
	}

	// wttr.in changes rendering based on the user agent, so we
	// need to set a user agent to force plain text.
	req.Header.Set("User-Agent", "curl/8.0")

	// Perform the HTTP request.
	resp, reqErr := http.DefaultClient.Do(req)
	if reqErr != nil {
		return fantasy.NewTextErrorResponse("request failed: " + reqErr.Error()), nil
	}

	// Read the response body.
	defer resp.Body.Close()
	b, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		return fantasy.NewTextErrorResponse("read failed: " + readErr.Error()), nil
	}

	// Did it work?
	if resp.StatusCode >= 400 {
		return fantasy.NewTextErrorResponse("wttr.in error: " + resp.Status + "\n" + string(b)), nil
	}

	// It worked!
	return fantasy.NewTextResponse(string(b)), nil
}

// Just a Lip Gloss text formatter.
var formatText func(...string) string

// Setup the text formatter based on terminal width so we can wrap lines
// nicely.
func init() {
	w, _, err := term.GetSize(os.Stdout.Fd())
	if err != nil {
		log.Fatalf("failed to get terminal size: %v", err)
	}
	formatText = lipgloss.NewStyle().Padding(0, 3, 1, 3).Width(w).Render
}
