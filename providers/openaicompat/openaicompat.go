// Package openaicompat provides an implementation of the fantasy AI SDK for OpenAI-compatible APIs.
package openaicompat

import (
	"charm.land/fantasy"
	"charm.land/fantasy/providers/openai"
	"github.com/openai/openai-go/v2/option"
)

type options struct {
	openaiOptions        []openai.Option
	languageModelOptions []openai.LanguageModelOption
	sdkOptions           []option.RequestOption
}

const (
	// Name is the name of the OpenAI-compatible provider.
	Name = "openai-compat"
)

// Option defines a function that configures OpenAI-compatible provider options.
type Option = func(*options)

// New creates a new OpenAI-compatible provider with the given options.
func New(opts ...Option) fantasy.Provider {
	providerOptions := options{
		openaiOptions: []openai.Option{
			openai.WithName(Name),
		},
		languageModelOptions: []openai.LanguageModelOption{
			openai.WithLanguageModelPrepareCallFunc(PrepareCallFunc),
			openai.WithLanguageModelStreamExtraFunc(StreamExtraFunc),
			openai.WithLanguageModelExtraContentFunc(ExtraContentFunc),
		},
	}
	for _, o := range opts {
		o(&providerOptions)
	}

	providerOptions.openaiOptions = append(
		providerOptions.openaiOptions,
		openai.WithSDKOptions(providerOptions.sdkOptions...),
		openai.WithLanguageModelOptions(providerOptions.languageModelOptions...),
	)
	return openai.New(providerOptions.openaiOptions...)
}

// WithBaseURL sets the base URL for the OpenAI-compatible provider.
func WithBaseURL(url string) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithBaseURL(url))
	}
}

// WithAPIKey sets the API key for the OpenAI-compatible provider.
func WithAPIKey(apiKey string) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithAPIKey(apiKey))
	}
}

// WithName sets the name for the OpenAI-compatible provider.
func WithName(name string) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithName(name))
	}
}

// WithHeaders sets the headers for the OpenAI-compatible provider.
func WithHeaders(headers map[string]string) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithHeaders(headers))
	}
}

// WithHTTPClient sets the HTTP client for the OpenAI-compatible provider.
func WithHTTPClient(client option.HTTPClient) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithHTTPClient(client))
	}
}

// WithSDKOptions sets the SDK options for the OpenAI-compatible provider.
func WithSDKOptions(opts ...option.RequestOption) Option {
	return func(o *options) {
		o.sdkOptions = append(o.sdkOptions, opts...)
	}
}
