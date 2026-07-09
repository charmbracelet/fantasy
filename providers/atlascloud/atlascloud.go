// Package atlascloud provides an implementation of the fantasy AI SDK for Atlas Cloud's OpenAI-compatible language models.
package atlascloud

import (
	"charm.land/fantasy"
	"charm.land/fantasy/providers/openaicompat"
	"github.com/charmbracelet/openai-go/option"
)

const (
	// DefaultURL is the default URL for the Atlas Cloud LLM API.
	DefaultURL = "https://api.atlascloud.ai/v1"
	// Name is the name of the Atlas Cloud provider.
	Name = "atlascloud"
)

type options struct {
	openaicompatOptions []openaicompat.Option
}

// Option defines a function that configures Atlas Cloud provider options.
type Option = func(*options)

// New creates a new Atlas Cloud provider with the given options.
func New(opts ...Option) (fantasy.Provider, error) {
	providerOptions := options{
		openaicompatOptions: []openaicompat.Option{
			openaicompat.WithName(Name),
			openaicompat.WithBaseURL(DefaultURL),
		},
	}
	for _, o := range opts {
		o(&providerOptions)
	}
	return openaicompat.New(providerOptions.openaicompatOptions...)
}

// WithAPIKey sets the API key for the Atlas Cloud provider.
func WithAPIKey(apiKey string) Option {
	return func(o *options) {
		o.openaicompatOptions = append(o.openaicompatOptions, openaicompat.WithAPIKey(apiKey))
	}
}

// WithBaseURL sets the base URL for the Atlas Cloud provider.
func WithBaseURL(url string) Option {
	return func(o *options) {
		o.openaicompatOptions = append(o.openaicompatOptions, openaicompat.WithBaseURL(url))
	}
}

// WithName sets the name for the Atlas Cloud provider.
func WithName(name string) Option {
	return func(o *options) {
		o.openaicompatOptions = append(o.openaicompatOptions, openaicompat.WithName(name))
	}
}

// WithHeaders sets the headers for the Atlas Cloud provider.
func WithHeaders(headers map[string]string) Option {
	return func(o *options) {
		o.openaicompatOptions = append(o.openaicompatOptions, openaicompat.WithHeaders(headers))
	}
}

// WithHTTPClient sets the HTTP client for the Atlas Cloud provider.
func WithHTTPClient(client option.HTTPClient) Option {
	return func(o *options) {
		o.openaicompatOptions = append(o.openaicompatOptions, openaicompat.WithHTTPClient(client))
	}
}

// WithUserAgent sets an explicit User-Agent header, overriding the default and any
// value set via WithHeaders.
func WithUserAgent(ua string) Option {
	return func(o *options) {
		o.openaicompatOptions = append(o.openaicompatOptions, openaicompat.WithUserAgent(ua))
	}
}

// WithObjectMode sets the object generation mode for the Atlas Cloud provider.
func WithObjectMode(om fantasy.ObjectMode) Option {
	return func(o *options) {
		o.openaicompatOptions = append(o.openaicompatOptions, openaicompat.WithObjectMode(om))
	}
}

// WithUseResponsesAPI configures the provider to use the responses API for models that support it.
func WithUseResponsesAPI() Option {
	return func(o *options) {
		o.openaicompatOptions = append(o.openaicompatOptions, openaicompat.WithUseResponsesAPI())
	}
}

// WithResponsesAPIFunc sets a custom filter for which models use the Responses API.
func WithResponsesAPIFunc(fn func(modelID string) bool) Option {
	return func(o *options) {
		o.openaicompatOptions = append(o.openaicompatOptions, openaicompat.WithResponsesAPIFunc(fn))
	}
}
