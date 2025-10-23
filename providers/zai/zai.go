// Package zai provides a Fantasy provider for the Z.ai Coding PaaS API.
package zai

import (
	"charm.land/fantasy"
	"charm.land/fantasy/providers/openaicompat"
)

const (
	// Name is the provider identifier for Z.ai.
	Name = "zai"
	// BaseURL is the default Z.ai Coding PaaS API base URL.
	BaseURL = "https://api.z.ai/api/coding/paas/v4"
)

// Option configures the Z.ai provider via OpenAI-compatible options.
type Option = openaicompat.Option

// New creates a new Z.ai provider using OpenAI-compatible transport/options.
func New(opts ...Option) (fantasy.Provider, error) {
	options := []Option{
		openaicompat.WithName(Name),
		openaicompat.WithBaseURL(BaseURL),
	}
	options = append(options, opts...)
	return openaicompat.New(options...)
}
