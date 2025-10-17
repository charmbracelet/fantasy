// Package azure provides an implementation of the fantasy AI SDK for Azure's language models.
package azure

import (
	"charm.land/fantasy"
	"charm.land/fantasy/providers/openaicompat"
	"github.com/openai/openai-go/v2/azure"
	"github.com/openai/openai-go/v2/option"
)

type options struct {
	baseURL    string
	apiKey     string
	apiVersion string

	openaiOptions []openaicompat.Option
}

const (
	// Name is the name of the Azure provider.
	Name = "azure"
	// defaultAPIVersion is the default API version for Azure.
	defaultAPIVersion = "2025-01-01-preview"
)

// Option defines a function that configures Azure provider options.
type Option = func(*options)

// New creates a new Azure provider with the given options.
func New(opts ...Option) fantasy.Provider {
	o := options{
		apiVersion: defaultAPIVersion,
	}
	for _, opt := range opts {
		opt(&o)
	}
	return openaicompat.New(
		append(
			o.openaiOptions,
			openaicompat.WithName(Name),
			openaicompat.WithSDKOptions(
				azure.WithEndpoint(o.baseURL, o.apiVersion),
				azure.WithAPIKey(o.apiKey),
			),
		)...,
	)
}

// WithBaseURL sets the base URL for the Azure provider.
func WithBaseURL(baseURL string) Option {
	return func(o *options) {
		o.baseURL = baseURL
	}
}

// WithAPIKey sets the API key for the Azure provider.
func WithAPIKey(apiKey string) Option {
	return func(o *options) {
		o.apiKey = apiKey
	}
}

// WithHeaders sets the headers for the Azure provider.
func WithHeaders(headers map[string]string) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openaicompat.WithHeaders(headers))
	}
}

// WithAPIVersion sets the API version for the Azure provider.
func WithAPIVersion(version string) Option {
	return func(o *options) {
		o.apiVersion = version
	}
}

// WithHTTPClient sets the HTTP client for the Azure provider.
func WithHTTPClient(client option.HTTPClient) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openaicompat.WithHTTPClient(client))
	}
}
