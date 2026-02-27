// Package bedrock provides an implementation of the fantasy AI SDK for AWS Bedrock's language models.
package bedrock

import (
	"charm.land/fantasy"
	"charm.land/fantasy/providers/anthropic"
	"github.com/charmbracelet/anthropic-sdk-go/option"
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
func New(opts ...Option) (fantasy.Provider, error) {
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

// WithAPIKey sets the access token for the Bedrock provider.
func WithAPIKey(apiKey string) Option {
	return func(o *options) {
		o.anthropicOptions = append(o.anthropicOptions, anthropic.WithAPIKey(apiKey))
	}
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

// WithUserAgent sets an explicit User-Agent header, overriding the default and any
// value set via WithHeaders.
func WithUserAgent(ua string) Option {
	return func(o *options) {
		o.anthropicOptions = append(o.anthropicOptions, anthropic.WithUserAgent(ua))
	}
}

// WithModelSegment sets the model segment appended to the default User-Agent.
// The resulting header is "Fantasy/<version> (<model>)". Pass an empty string
// to clear a previously set segment.
func WithModelSegment(model string) Option {
	return func(o *options) {
		o.anthropicOptions = append(o.anthropicOptions, anthropic.WithModelSegment(model))
	}
}

// WithSkipAuth configures whether to skip authentication for the Bedrock provider.
func WithSkipAuth(skipAuth bool) Option {
	return func(o *options) {
		o.skipAuth = skipAuth
	}
}
