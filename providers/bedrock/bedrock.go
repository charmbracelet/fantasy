// Package bedrock provides an implementation of the fantasy AI SDK for AWS Bedrock's language models.
package bedrock

import (
	"charm.land/fantasy"
	"charm.land/fantasy/providers/anthropic"
	"github.com/anthropics/anthropic-sdk-go/option"
)

type options struct {
	skipAuth         bool
	anthropicOptions []anthropic.Option
}

const (
	// Name is the name of the Bedrock provider.
	Name = "bedrock"
)

// Option defines a function that configures Bedrock provider options.
type Option = func(*options)

// New creates a new Bedrock provider with the given options.
func New(opts ...Option) fantasy.Provider {
	var o options
	for _, opt := range opts {
		opt(&o)
	}
	return anthropic.New(
		append(
			o.anthropicOptions,
			anthropic.WithName(Name),
			anthropic.WithBedrock(),
			anthropic.WithSkipAuth(o.skipAuth),
		)...,
	)
}

// WithHeaders sets the headers for the Bedrock provider.
func WithHeaders(headers map[string]string) Option {
	return func(o *options) {
		o.anthropicOptions = append(o.anthropicOptions, anthropic.WithHeaders(headers))
	}
}

// WithHTTPClient sets the HTTP client for the Bedrock provider.
func WithHTTPClient(client option.HTTPClient) Option {
	return func(o *options) {
		o.anthropicOptions = append(o.anthropicOptions, anthropic.WithHTTPClient(client))
	}
}

// WithSkipAuth configures whether to skip authentication for the Bedrock provider.
func WithSkipAuth(skipAuth bool) Option {
	return func(o *options) {
		o.skipAuth = skipAuth
	}
}
