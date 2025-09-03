package providertests

import (
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
