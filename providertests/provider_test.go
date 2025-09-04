package providertests

import (
	"context"
	"strings"
	"testing"

	"github.com/charmbracelet/ai/ai"
	_ "github.com/joho/godotenv/autoload"
)

func TestSimple(t *testing.T) {
	for _, pair := range languageModelBuilders {
		t.Run(pair.name, func(t *testing.T) {
			r := newRecorder(t)

			languageModel, err := pair.builder(r)
			if err != nil {
				t.Fatalf("failed to build language model: %v", err)
			}

			agent := ai.NewAgent(
				languageModel,
				ai.WithSystemPrompt("You are a helpful assistant"),
			)
			result, err := agent.Generate(t.Context(), ai.AgentCall{
				Prompt: "Say hi in Portuguese",
			})
			if err != nil {
				t.Fatalf("failed to generate: %v", err)
			}

			want := "Olá"
			got := result.Response.Content.Text()
			if !strings.Contains(got, want) {
				t.Fatalf("unexpected response: got %q, want %q", got, want)
			}
		})
	}
}

func TestTool(t *testing.T) {
	for _, pair := range languageModelBuilders {
		t.Run(pair.name, func(t *testing.T) {
			r := newRecorder(t)

			languageModel, err := pair.builder(r)
			if err != nil {
				t.Fatalf("failed to build language model: %v", err)
			}

			type WeatherInput struct {
				Location string `json:"location" description:"the city"`
			}

			weatherTool := ai.NewAgentTool(
				"weather",
				"Get weather information for a location",
				func(ctx context.Context, input WeatherInput, _ ai.ToolCall) (ai.ToolResponse, error) {
					return ai.NewTextResponse("40 C"), nil
				},
			)

			agent := ai.NewAgent(
				languageModel,
				ai.WithSystemPrompt("You are a helpful assistant"),
				ai.WithTools(weatherTool),
			)
			result, err := agent.Generate(t.Context(), ai.AgentCall{
				Prompt: "What's the weather in Florence?",
			})
			if err != nil {
				t.Fatalf("failed to generate: %v", err)
			}

			want1 := "Florence"
			want2 := "40"
			got := result.Response.Content.Text()
			if !strings.Contains(got, want1) || !strings.Contains(got, want2) {
				t.Fatalf("unexpected response: got %q, want %q %q", got, want1, want2)
			}
		})
	}
}
