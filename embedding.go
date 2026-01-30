package fantasy

import (
	"context"
	"fmt"
)

// EmbeddingProvider represents a provider that can create embedding models.
// This is separate from Provider to avoid breaking changes.
type EmbeddingProvider interface {
	EmbeddingModel(ctx context.Context, modelID string) (EmbeddingModel, error)
}

// EmbeddingModel represents a model that can generate embeddings.
type EmbeddingModel interface {
	Embed(context.Context, EmbeddingCall) (*EmbeddingResponse, error)

	Provider() string
	Model() string
}

// EmbeddingCall represents a request to generate embeddings.
// Exactly one of Input or Inputs must be provided.
type EmbeddingCall struct {
	Input      *string  `json:"input,omitempty"`
	Inputs     []string `json:"inputs,omitempty"`
	Dimensions *int64   `json:"dimensions,omitempty"`

	ProviderOptions ProviderOptions `json:"provider_options,omitempty"`
}

// Embedding represents a single embedding vector.
type Embedding struct {
	Index  int       `json:"index"`
	Vector []float32 `json:"vector"`
}

// EmbeddingResponse represents the response from an embedding model.
type EmbeddingResponse struct {
	Model      string      `json:"model"`
	Usage      Usage       `json:"usage"`
	Embeddings []Embedding `json:"embeddings"`
}

// ValidateEmbeddingCall validates the embedding request parameters.
func ValidateEmbeddingCall(call EmbeddingCall) error {
	hasInput := call.Input != nil
	hasInputs := len(call.Inputs) > 0

	if hasInput == hasInputs {
		return &Error{
			Title:   "invalid argument",
			Message: "embedding call must set exactly one of input or inputs",
		}
	}

	if hasInput {
		if *call.Input == "" {
			return &Error{
				Title:   "invalid argument",
				Message: "embedding input cannot be empty",
			}
		}
		return nil
	}

	for i, input := range call.Inputs {
		if input == "" {
			return &Error{
				Title:   "invalid argument",
				Message: fmt.Sprintf("embedding inputs[%d] cannot be empty", i),
			}
		}
	}

	return nil
}
