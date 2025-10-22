// Package openrouter provides an implementation of the fantasy AI SDK for OpenRouter's language models.
package openrouter

import (
	"encoding/json"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openai"
	"github.com/openai/openai-go/v2/option"
)

type options struct {
	openaiOptions        []openai.Option
	languageModelOptions []openai.LanguageModelOption
}

const (
	// DefaultURL is the default URL for the OpenRouter API.
	DefaultURL = "https://openrouter.ai/api/v1"
	// Name is the name of the OpenRouter provider.
	Name = "openrouter"
)

// Option defines a function that configures OpenRouter provider options.
type Option = func(*options)

// New creates a new OpenRouter provider with the given options.
func New(opts ...Option) (fantasy.Provider, error) {
	providerOptions := options{
		openaiOptions: []openai.Option{
			openai.WithName(Name),
			openai.WithBaseURL(DefaultURL),
		},
		languageModelOptions: []openai.LanguageModelOption{
			openai.WithLanguageModelPrepareCallFunc(languagePrepareModelCall),
			openai.WithLanguageModelUsageFunc(languageModelUsage),
			openai.WithLanguageModelStreamUsageFunc(languageModelStreamUsage),
			openai.WithLanguageModelStreamExtraFunc(languageModelStreamExtra),
			openai.WithLanguageModelExtraContentFunc(languageModelExtraContent),
			openai.WithLanguageModelToPromptFunc(languageModelToPrompt),
		},
	}
	for _, o := range opts {
		o(&providerOptions)
	}

	providerOptions.openaiOptions = append(providerOptions.openaiOptions, openai.WithLanguageModelOptions(providerOptions.languageModelOptions...))
	return openai.New(providerOptions.openaiOptions...)
}

// WithAPIKey sets the API key for the OpenRouter provider.
func WithAPIKey(apiKey string) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithAPIKey(apiKey))
	}
}

// WithName sets the name for the OpenRouter provider.
func WithName(name string) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithName(name))
	}
}

// WithHeaders sets the headers for the OpenRouter provider.
func WithHeaders(headers map[string]string) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithHeaders(headers))
	}
}

// WithHTTPClient sets the HTTP client for the OpenRouter provider.
func WithHTTPClient(client option.HTTPClient) Option {
	return func(o *options) {
		o.openaiOptions = append(o.openaiOptions, openai.WithHTTPClient(client))
	}
}

func structToMapJSON(s any) (map[string]any, error) {
	var result map[string]any
	jsonBytes, err := json.Marshal(s)
	if err != nil {
		return nil, err
	}
	err = json.Unmarshal(jsonBytes, &result)
	if err != nil {
		return nil, err
	}
	return result, nil
}
