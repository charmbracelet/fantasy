// Package huggingface provides a Fantasy provider for the Hugging Face Inference API.
package huggingface

import (
	"charm.land/fantasy"
	"charm.land/fantasy/providers/openaicompat"
)

const (
	// Name is the provider identifier for Hugging Face.
	Name = "huggingface"
	// BaseURL is the default Hugging Face Inference API base URL.
	BaseURL = "https://router.huggingface.co/v1"
)

// Option configures the Hugging Face provider via OpenAI-compatible options.
type Option = openaicompat.Option

// New creates a new Hugging Face provider using OpenAI-compatible transport/options.
func New(opts ...Option) (fantasy.Provider, error) {
	options := []Option{
		openaicompat.WithName(Name),
		openaicompat.WithBaseURL(BaseURL),
	}
	options = append(options, opts...)
	return openaicompat.New(options...)
}
