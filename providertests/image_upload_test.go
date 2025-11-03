package providertests

import (
	"cmp"
	"net/http"
	"os"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/anthropic"
	"charm.land/fantasy/providers/google"
	"charm.land/fantasy/providers/openai"
	"github.com/stretchr/testify/require"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func anthropicImageBuilder(model string) builderFunc {
	return func(t *testing.T, r *recorder.Recorder) (fantasy.LanguageModel, error) {
		provider, err := anthropic.New(
			anthropic.WithAPIKey(cmp.Or(os.Getenv("FANTASY_ANTHROPIC_API_KEY"), "(missing)")),
			anthropic.WithHTTPClient(&http.Client{Transport: r}),
		)
		if err != nil {
			return nil, err
		}
		return provider.LanguageModel(t.Context(), model)
	}
}

func openAIImageBuilder(model string) builderFunc {
	return func(t *testing.T, r *recorder.Recorder) (fantasy.LanguageModel, error) {
		provider, err := openai.New(
			openai.WithAPIKey(cmp.Or(os.Getenv("FANTASY_OPENAI_API_KEY"), "(missing)")),
			openai.WithHTTPClient(&http.Client{Transport: r}),
		)
		if err != nil {
			return nil, err
		}
		return provider.LanguageModel(t.Context(), model)
	}
}

func geminiImageBuilder(model string) builderFunc {
	return func(t *testing.T, r *recorder.Recorder) (fantasy.LanguageModel, error) {
		provider, err := google.New(
			google.WithGeminiAPIKey(cmp.Or(os.Getenv("FANTASY_GEMINI_API_KEY"), "(missing)")),
			google.WithHTTPClient(&http.Client{Transport: r}),
		)
		if err != nil {
			return nil, err
		}
		return provider.LanguageModel(t.Context(), model)
	}
}

func TestImageUploadAgent(t *testing.T) {
	pairs := []builderPair{
		{
			name:    "anthropic-claude-sonnet-4",
			builder: anthropicImageBuilder("claude-sonnet-4-20250514"),
		},
		{
			name:    "openai-gpt-5",
			builder: openAIImageBuilder("gpt-5"),
		},
		{
			name:    "gemini-2.5-pro",
			builder: geminiImageBuilder("gemini-2.5-pro"),
		},
	}

	img, err := os.ReadFile("testdata/wish.png")
	require.NoError(t, err)

	file := fantasy.FilePart{Filename: "wish.png", Data: img, MediaType: "image/png"}

	for _, pair := range pairs {
		t.Run(pair.name, func(t *testing.T) {
			r := newRecorder(t)

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
				MaxOutputTokens: fantasy.Opt(int64(4000)),
			})
			require.NoError(t, err)
			got := result.Response.Content.Text()
			require.NotEmpty(t, got, "expected non-empty description for %s", pair.name)
		})
	}
}

func TestImageUploadAgentStreaming(t *testing.T) {
	pairs := []builderPair{
		{
			name:    "anthropic-claude-sonnet-4",
			builder: anthropicImageBuilder("claude-sonnet-4-20250514"),
		},
		{
			name:    "openai-gpt-5",
			builder: openAIImageBuilder("gpt-5"),
		},
		{
			name:    "gemini-2.5-pro",
			builder: geminiImageBuilder("gemini-2.5-pro"),
		},
	}

	img, err := os.ReadFile("testdata/wish.png")
	require.NoError(t, err)

	file := fantasy.FilePart{Filename: "wish.png", Data: img, MediaType: "image/png"}

	for _, pair := range pairs {
		t.Run(pair.name+"-stream", func(t *testing.T) {
			r := newRecorder(t)

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
				MaxOutputTokens: fantasy.Opt(int64(4000)),
			})
			require.NoError(t, err)
			got := result.Response.Content.Text()
			require.NotEmpty(t, got, "expected non-empty description for %s", pair.name)
		})
	}
}
