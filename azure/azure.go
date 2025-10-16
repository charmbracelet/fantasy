package azure

import (
	"charm.land/fantasy/ai"
	"charm.land/fantasy/openaicompat"
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
	Name              = "azure"
	defaultAPIVersion = "2025-01-01-preview"
)

type Option = func(*options)

func New(opts ...Option) ai.Provider {
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

func WithBaseURL(baseURL string) Option {
	return func(o *options) {
		o.baseURL = baseURL
	}
}

func WithAPIKey(apiKey string) Option {
	return func(o *options) {
		o.apiKey = apiKey
	}
}

func WithHeaders(headers map[string]string) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openaicompat.WithHeaders(headers))
	}
}

func WithAPIVersion(version string) Option {
	return func(o *options) {
		o.apiVersion = version
	}
}

func WithHTTPClient(client option.HTTPClient) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openaicompat.WithHTTPClient(client))
	}
}
