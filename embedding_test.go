package fantasy

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestValidateEmbeddingCall(t *testing.T) {
	t.Run("requires one input", func(t *testing.T) {
		err := ValidateEmbeddingCall(EmbeddingCall{})
		require.Error(t, err)
	})

	t.Run("rejects both input and inputs", func(t *testing.T) {
		input := "hello"
		err := ValidateEmbeddingCall(EmbeddingCall{
			Input:  &input,
			Inputs: []string{"world"},
		})
		require.Error(t, err)
	})

	t.Run("rejects empty input", func(t *testing.T) {
		input := ""
		err := ValidateEmbeddingCall(EmbeddingCall{Input: &input})
		require.Error(t, err)
	})

	t.Run("rejects empty inputs", func(t *testing.T) {
		err := ValidateEmbeddingCall(EmbeddingCall{Inputs: []string{""}})
		require.Error(t, err)
	})

	t.Run("accepts single input", func(t *testing.T) {
		input := "hello"
		err := ValidateEmbeddingCall(EmbeddingCall{Input: &input})
		require.NoError(t, err)
	})

	t.Run("accepts batch inputs", func(t *testing.T) {
		err := ValidateEmbeddingCall(EmbeddingCall{Inputs: []string{"a", "b"}})
		require.NoError(t, err)
	})
}
