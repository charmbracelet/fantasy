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

// New creates a new Cerebras provider using OpenAI-compatible transport/options.
func New(opts ...Option) (fantasy.Provider, error) {
	options := []Option{
		openaicompat.WithName(Name),
		WithBaseURL(BaseURL),
	}
	options = append(options, opts...)
	return openaicompat.New(options...)
}
