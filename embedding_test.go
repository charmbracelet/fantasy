package fantasy

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestValidateEmbeddingCall(t *testing.T) {
	t.Run("requires inputs", func(t *testing.T) {
		err := ValidateEmbeddingCall(EmbeddingCall{})
		require.Error(t, err)
	})

	t.Run("rejects empty inputs", func(t *testing.T) {
		err := ValidateEmbeddingCall(EmbeddingCall{Inputs: []string{""}})
		require.Error(t, err)
	})

	t.Run("accepts single input in inputs", func(t *testing.T) {
		err := ValidateEmbeddingCall(EmbeddingCall{Inputs: []string{"hello"}})
		require.NoError(t, err)
	})

	t.Run("accepts batch inputs", func(t *testing.T) {
		err := ValidateEmbeddingCall(EmbeddingCall{Inputs: []string{"a", "b"}})
		require.NoError(t, err)
	})
}
