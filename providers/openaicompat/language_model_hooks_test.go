package openaicompat

import (
	"context"
	"testing"

	"charm.land/fantasy"
	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/require"
)

func testModel(t *testing.T) fantasy.LanguageModel {
	t.Helper()

	provider, err := New(WithAPIKey("test"), WithBaseURL("http://127.0.0.1:1/v1"))
	require.NoError(t, err)
	model, err := provider.LanguageModel(context.Background(), "test-model")
	require.NoError(t, err)
	return model
}

func TestPrepareCallFunc_TopK(t *testing.T) {
	t.Parallel()

	t.Run("sends top_k as an extra body field", func(t *testing.T) {
		t.Parallel()

		params := &openai.ChatCompletionNewParams{}
		warnings, err := PrepareCallFunc(testModel(t), params, fantasy.Call{TopK: new(int64(20))})

		require.NoError(t, err)
		require.Empty(t, warnings)
		require.Equal(t, int64(20), params.ExtraFields()["top_k"])
	})

	t.Run("keeps extra body entries alongside top_k", func(t *testing.T) {
		t.Parallel()

		params := &openai.ChatCompletionNewParams{}
		call := fantasy.Call{
			TopK: new(int64(20)),
			ProviderOptions: NewProviderOptions(&ProviderOptions{
				ExtraBody: map[string]any{"min_p": 0.05},
			}),
		}
		warnings, err := PrepareCallFunc(testModel(t), params, call)

		require.NoError(t, err)
		require.Empty(t, warnings)
		require.Equal(t, map[string]any{"min_p": 0.05, "top_k": int64(20)}, params.ExtraFields())
	})

	t.Run("an explicit extra body top_k wins", func(t *testing.T) {
		t.Parallel()

		params := &openai.ChatCompletionNewParams{}
		call := fantasy.Call{
			TopK: new(int64(20)),
			ProviderOptions: NewProviderOptions(&ProviderOptions{
				ExtraBody: map[string]any{"top_k": 40},
			}),
		}
		warnings, err := PrepareCallFunc(testModel(t), params, call)

		require.NoError(t, err)
		require.Empty(t, warnings)
		require.Equal(t, 40, params.ExtraFields()["top_k"])
	})

	t.Run("does not mutate the caller extra body", func(t *testing.T) {
		t.Parallel()

		extraBody := map[string]any{"min_p": 0.05}
		call := fantasy.Call{
			TopK:            new(int64(20)),
			ProviderOptions: NewProviderOptions(&ProviderOptions{ExtraBody: extraBody}),
		}
		_, err := PrepareCallFunc(testModel(t), &openai.ChatCompletionNewParams{}, call)

		require.NoError(t, err)
		require.Equal(t, map[string]any{"min_p": 0.05}, extraBody)
	})

	t.Run("omits top_k when unset", func(t *testing.T) {
		t.Parallel()

		params := &openai.ChatCompletionNewParams{}
		warnings, err := PrepareCallFunc(testModel(t), params, fantasy.Call{})

		require.NoError(t, err)
		require.Empty(t, warnings)
		require.NotContains(t, params.ExtraFields(), "top_k")
	})
}
