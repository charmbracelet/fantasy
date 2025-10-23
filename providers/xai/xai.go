// Package xai provides a Fantasy provider for the xAI API.
package xai

import (
	"charm.land/fantasy"
	"charm.land/fantasy/providers/openaicompat"
)

const (
	// Name is the provider identifier for xAI.
	Name = "xai"
	// BaseURL is the default xAI API base URL.
	BaseURL = "https://api.x.ai/v1"
)

// Option configures the xAI provider via OpenAI-compatible options.
type Option = openaicompat.Option

// New creates a new xAI provider using OpenAI-compatible transport/options.
func New(opts ...Option) (fantasy.Provider, error) {
	options := []Option{
		openaicompat.WithName(Name),
		openaicompat.WithBaseURL(BaseURL),
	}
	options = append(options, opts...)
	return openaicompat.New(options...)
}
