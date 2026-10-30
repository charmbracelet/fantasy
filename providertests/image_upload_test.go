package providertests

import (
	"cmp"
	"os"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/google"
	"charm.land/x/vcr"
	"github.com/stretchr/testify/require"
)

func geminiImageBuilder(model string) builderFunc {
	return vcrProviderBuilder(model,
		google.New,
		google.WithGeminiAPIKey,
		google.WithHTTPClient,
		cmp.Or(os.Getenv("FANTASY_GEMINI_API_KEY"), "(missing)"),
	)
}

func imageUploadPairs(t *testing.T) ([]builderPair, fantasy.FilePart) {
	pairs := []builderPair{
		{
			name:    "anthropic-claude-sonnet-4",
			builder: anthropicBuilder("claude-sonnet-4-20250514"),
		},
		{
			name:    "openai-gpt-5",
			builder: openAIBuilder("gpt-5"),
		},
		{
			name:    "gemini-2.5-pro",
			builder: geminiImageBuilder("gemini-2.5-pro"),
		},
	}

	img, err := os.ReadFile("testdata/wish.png")
	require.NoError(t, err)

	return pairs, fantasy.FilePart{Filename: "wish.png", Data: img, MediaType: "image/png"}
}

func TestImageUploadAgent(t *testing.T) {
	pairs, file := imageUploadPairs(t)

	for _, pair := range pairs {
		t.Run(pair.name, func(t *testing.T) {
			r := vcr.NewRecorder(t)

			lm, err := pair.builder(t, r)
			require.NoError(t, err)

			agent := fantasy.NewAgent(
				lm,
				fantasy.WithSystemPrompt("You are a helpful assistant"),
			)

			result, err := agent.Generate(t.Context(), fantasy.AgentCall{
				Prompt:          "Describe the image briefly in English.",
				Files:           []fantasy.FilePart{file},
				ProviderOptions: pair.providerOptions,
				MaxOutputTokens: new(int64(4000)),
			})
			require.NoError(t, err)
			got := result.Response.Content.Text()
			require.NotEmpty(t, got, "expected non-empty description for %s", pair.name)
		})
	}
}

func TestImageUploadAgentStreaming(t *testing.T) {
	pairs, file := imageUploadPairs(t)

	for _, pair := range pairs {
		t.Run(pair.name+"-stream", func(t *testing.T) {
			r := vcr.NewRecorder(t)

			lm, err := pair.builder(t, r)
			require.NoError(t, err)

			agent := fantasy.NewAgent(
				lm,
				fantasy.WithSystemPrompt("You are a helpful assistant"),
			)

			result, err := agent.Stream(t.Context(), fantasy.AgentStreamCall{
				Prompt:          "Describe the image briefly in English.",
				Files:           []fantasy.FilePart{file},
				ProviderOptions: pair.providerOptions,
				MaxOutputTokens: new(int64(4000)),
			})
			require.NoError(t, err)
			got := result.Response.Content.Text()
			require.NotEmpty(t, got, "expected non-empty description for %s", pair.name)
		})
	}
}
