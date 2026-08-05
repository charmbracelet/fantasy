package providertests

import (
	"fmt"
	"net/http"
	"os"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openai"
	"charm.land/x/vcr"
	"github.com/stretchr/testify/require"
)

type embeddingBuilderFunc func(t *testing.T, r *vcr.Recorder) (fantasy.EmbeddingModel, error)

func openAIEmbeddingBuilder(modelID string) embeddingBuilderFunc {
	return func(t *testing.T, r *vcr.Recorder) (fantasy.EmbeddingModel, error) {
		baseURL := "https://api.synthetic.new/openai/v1"
		if os.Getenv("FANTASY_BASE_URL") != "" {
			baseURL = os.Getenv("FANTASY_BASE_URL")
		}

		apiKey := os.Getenv("FANTASY_OPENAI_API_KEY")
		if os.Getenv("FANTASY_API_KEY") != "" {
			apiKey = os.Getenv("FANTASY_API_KEY")
		}

		provider, err := openai.New(
			openai.WithBaseURL(baseURL),
			openai.WithAPIKey(apiKey),
			openai.WithHTTPClient(&http.Client{Transport: r}),
		)
		if err != nil {
			return nil, err
		}

		embeddingProvider, ok := provider.(fantasy.EmbeddingProvider)
		if !ok {
			return nil, fmt.Errorf("provider %q does not support embeddings", provider.Name())
		}

		return embeddingProvider.EmbeddingModel(t.Context(), modelID)
	}
}

func embeddingModelID() string {
	if os.Getenv("FANTASY_EMBEDDING_MODEL") != "" {
		return os.Getenv("FANTASY_EMBEDDING_MODEL")
	}
	return "hf:nomic-ai/nomic-embed-text-v1.5"
}

func TestOpenAIEmbeddings(t *testing.T) {
	builder := openAIEmbeddingBuilder(embeddingModelID())

	t.Run("single input", func(t *testing.T) {
		r := vcr.NewRecorder(t)

		model, err := builder(t, r)
		require.NoError(t, err)

		response, err := model.Embed(t.Context(), fantasy.EmbeddingCall{
			Inputs: []string{"The quick brown fox"},
		})
		require.NoError(t, err)
		require.NotNil(t, response)
		require.Equal(t, embeddingModelID(), response.Model)
		require.Len(t, response.Embeddings, 1)
		require.Equal(t, 0, response.Embeddings[0].Index)
		require.NotEmpty(t, response.Embeddings[0].Vector)
	})

	t.Run("batch input", func(t *testing.T) {
		r := vcr.NewRecorder(t)

		model, err := builder(t, r)
		require.NoError(t, err)

		response, err := model.Embed(t.Context(), fantasy.EmbeddingCall{
			Inputs: []string{
				"The quick brown fox",
				"Pack my box with five dozen liquor jugs",
				"How vexingly quick daft zebras jump",
			},
		})
		require.NoError(t, err)
		require.NotNil(t, response)
		require.Equal(t, embeddingModelID(), response.Model)
		require.Len(t, response.Embeddings, 3)
		for i, embedding := range response.Embeddings {
			require.Equal(t, i, embedding.Index)
			require.NotEmpty(t, embedding.Vector)
		}
	})
}

func TestOpenAIEmbeddingsWithDimensions(t *testing.T) {
	builder := openAIEmbeddingBuilder(embeddingModelID())

	t.Run("with dimensions", func(t *testing.T) {
		r := vcr.NewRecorder(t)

		model, err := builder(t, r)
		require.NoError(t, err)

		dimensions := int64(256)
		response, err := model.Embed(t.Context(), fantasy.EmbeddingCall{
			Inputs:     []string{"The quick brown fox"},
			Dimensions: &dimensions,
		})
		require.NoError(t, err)
		require.NotNil(t, response)
		require.Equal(t, embeddingModelID(), response.Model)
		require.Len(t, response.Embeddings, 1)
		require.Equal(t, 256, len(response.Embeddings[0].Vector))
	})
}
