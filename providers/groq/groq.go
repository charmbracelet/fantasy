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

var (
	// WithBaseURL is an alias for openaicompat.WithBaseURL.
	WithBaseURL = openaicompat.WithBaseURL
	// WithAPIKey is an alias for openaicompat.WithAPIKey.
	WithAPIKey = openaicompat.WithAPIKey
	// WithHeaders is an alias for openaicompat.WithHeaders.
	WithHeaders = openaicompat.WithHeaders
	// WithHTTPClient is an alias for openaicompat.WithHTTPClient.
	WithHTTPClient = openaicompat.WithHTTPClient
	// WithSDKOptions is an alias for openaicompat.WithSDKOptions.
	WithSDKOptions = openaicompat.WithSDKOptions
)

// New creates a new Groq provider using OpenAI-compatible transport/options.
func New(opts ...Option) (fantasy.Provider, error) {
	options := []Option{
		openaicompat.WithName(Name),
		WithBaseURL(BaseURL),
	}
	options = append(options, opts...)
	return openaicompat.New(options...)
}
