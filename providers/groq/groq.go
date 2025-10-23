// Package groq provides a Fantasy provider for the Groq API.
package groq

import (
	"charm.land/fantasy"
	"charm.land/fantasy/providers/openaicompat"
)

const (
	// Name is the provider identifier for Groq.
	Name = "groq"
	// BaseURL is the default Groq API base URL.
	BaseURL = "https://api.groq.com/openai/v1"
)

// Option configures the Groq provider via OpenAI-compatible options.
type Option = openaicompat.Option

// New creates a new Groq provider using OpenAI-compatible transport/options.
func New(opts ...Option) (fantasy.Provider, error) {
	options := []Option{
		openaicompat.WithName(Name),
		openaicompat.WithBaseURL(BaseURL),
	}
	options = append(options, opts...)
	return openaicompat.New(options...)
}
