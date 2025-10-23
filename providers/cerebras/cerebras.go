// Package cerebras provides a Fantasy provider for the Cerebras Inference API.
package cerebras

import (
	"charm.land/fantasy"
	"charm.land/fantasy/providers/openaicompat"
)

const (
	// Name is the provider identifier for Cerebras.
	Name = "cerebras"
	// BaseURL is the default Cerebras API base URL.
	BaseURL = "https://api.cerebras.ai/v1"
)

// Option configures the Cerebras provider via OpenAI-compatible options.
type Option = openaicompat.Option

// New creates a new Cerebras provider using OpenAI-compatible transport/options.
func New(opts ...Option) (fantasy.Provider, error) {
	options := []Option{
		openaicompat.WithName(Name),
		openaicompat.WithBaseURL(BaseURL),
	}
	options = append(options, opts...)
	return openaicompat.New(options...)
}
