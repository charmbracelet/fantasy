// Package azure provides an implementation of the fantasy AI SDK for Azure's language models.
package azure

import (
	"fmt"
	"strings"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openai"
	"github.com/openai/openai-go/v2/azure"
	"github.com/openai/openai-go/v2/option"
)

type options struct {
	baseURL    string
	apiKey     string
	apiVersion string

	openaiOptions []openai.Option
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
func New(opts ...Option) (fantasy.Provider, error) {
	o := options{
		apiVersion: defaultAPIVersion,
	}
	for _, opt := range opts {
		opt(&o)
	}
	return openai.New(
		append(
			o.openaiOptions,
			openai.WithName(Name),
			openai.WithBaseURL(o.baseURL),
			openai.WithSDKOptions(
				azure.WithAPIKey(o.apiKey),
			),
		)...,
	)
}

// WithBaseURL sets the base URL for the Azure provider.
func WithBaseURL(baseURL string) Option {
	return func(o *options) {
		// This tries to find the resource ID and make sure we use the correct URL
		//  azure gives the user multiple urls for different endpoints we make sure to use the correct one
		baseURL = strings.TrimPrefix(baseURL, "https://")
		parts := strings.Split(baseURL, ".")
		if len(parts) >= 2 {
			resourceID := parts[0]
			o.baseURL = fmt.Sprintf("https://%s.openai.azure.com/openai/v1", resourceID)
			return
		}
		// fallback to use the provided url
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
		o.openaiOptions = append(o.openaiOptions, openai.WithHeaders(headers))
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
		o.openaiOptions = append(o.openaiOptions, openai.WithHTTPClient(client))
	}
}

// WithUseResponsesAPI configures the provider to use the responses API for models that support it.
func WithUseResponsesAPI() Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithUseResponsesAPI())
	}
}
