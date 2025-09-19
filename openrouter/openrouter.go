package openrouter

import (
	"github.com/charmbracelet/fantasy/ai"
	"github.com/charmbracelet/fantasy/openai"
	openaisdk "github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
)

type options struct {
	openaiOptions []openai.Option
}

const (
	DefaultURL = "https://openrouter.ai/api/v1"
)

type Option = func(*options)

func prepareCallWithOptions(model ai.LanguageModel, params *openaisdk.ChatCompletionNewParams, call ai.Call) ([]ai.CallWarning, error) {
	providerOptions := &ProviderOptions{}
	if v, ok := call.ProviderOptions[Name]; ok {
		providerOptions, ok = v.(*ProviderOptions)
		if !ok {
			return nil, ai.NewInvalidArgumentError("providerOptions", "openrouter provider options should be *openrouter.ProviderOptions", nil)
		}
	}
	_ = providerOptions

	// HANDLE OPENROUTER call modification here

	return nil, nil
}

func New(opts ...Option) ai.Provider {
	providerOptions := options{
		openaiOptions: []openai.Option{
			openai.WithBaseURL(DefaultURL),
			openai.WithLanguageModelOptions(
				openai.WithPrepareLanguageModelCallFunc(prepareCallWithOptions),
			),
		},
	}
	for _, o := range opts {
		o(&providerOptions)
	}
	return openai.New(providerOptions.openaiOptions...)
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
