package openaicompat

import (
	"charm.land/fantasy"
	"charm.land/fantasy/openai"
	"github.com/openai/openai-go/v2/option"
)

type options struct {
	openaiOptions        []openai.Option
	languageModelOptions []openai.LanguageModelOption
	sdkOptions           []option.RequestOption
}

const (
	Name = "openai-compat"
)

type Option = func(*options)

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

func WithBaseURL(url string) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithBaseURL(url))
	}
}

func WithAPIKey(apiKey string) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithAPIKey(apiKey))
	}
}

func WithName(name string) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithName(name))
	}
}

func WithHeaders(headers map[string]string) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithHeaders(headers))
	}
}

func WithHTTPClient(client option.HTTPClient) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithHTTPClient(client))
	}
}

func WithSDKOptions(opts ...option.RequestOption) Option {
	return func(o *options) {
		o.sdkOptions = append(o.sdkOptions, opts...)
	}
}
