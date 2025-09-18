package openai

import (
	"cmp"
	"maps"

	"github.com/charmbracelet/fantasy/ai"
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
)

const (
	Name       = "openai"
	DefaultURL = "https://api.openai.com/v1"
)

type provider struct {
	options options
}

type options struct {
	baseURL              string
	apiKey               string
	organization         string
	project              string
	name                 string
	headers              map[string]string
	client               option.HTTPClient
	languageModelOptions []LanguageModelOption
}

type Option = func(*options)

func New(opts ...Option) ai.Provider {
	providerOptions := options{
		headers:              map[string]string{},
		languageModelOptions: make([]LanguageModelOption, 0),
	}
	for _, o := range opts {
		o(&providerOptions)
	}

	providerOptions.baseURL = cmp.Or(providerOptions.baseURL, DefaultURL)
	providerOptions.name = cmp.Or(providerOptions.name, Name)

	if providerOptions.organization != "" {
		providerOptions.headers["OpenAi-Organization"] = providerOptions.organization
	}
	if providerOptions.project != "" {
		providerOptions.headers["OpenAi-Project"] = providerOptions.project
	}

	return &provider{options: providerOptions}
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

func WithOrganization(organization string) Option {
	return func(o *options) {
		o.organization = organization
	}
}

func WithProject(project string) Option {
	return func(o *options) {
		o.project = project
	}
}

func WithName(name string) Option {
	return func(o *options) {
		o.name = name
	}
}

func WithHeaders(headers map[string]string) Option {
	return func(o *options) {
		maps.Copy(o.headers, headers)
	}
}

func WithHTTPClient(client option.HTTPClient) Option {
	return func(o *options) {
		o.client = client
	}
}

func WithLanguageModelOptions(opts ...LanguageModelOption) Option {
	return func(o *options) {
		o.languageModelOptions = append(o.languageModelOptions, opts...)
	}
}

// LanguageModel implements ai.Provider.
func (o *provider) LanguageModel(modelID string) (ai.LanguageModel, error) {
	openaiClientOptions := []option.RequestOption{}
	if o.options.apiKey != "" {
		openaiClientOptions = append(openaiClientOptions, option.WithAPIKey(o.options.apiKey))
	}
	if o.options.baseURL != "" {
		openaiClientOptions = append(openaiClientOptions, option.WithBaseURL(o.options.baseURL))
	}

	for key, value := range o.options.headers {
		openaiClientOptions = append(openaiClientOptions, option.WithHeader(key, value))
	}

	if o.options.client != nil {
		openaiClientOptions = append(openaiClientOptions, option.WithHTTPClient(o.options.client))
	}

	return newLanguageModel(
		modelID,
		o.options.name,
		openai.NewClient(openaiClientOptions...),
		o.options.languageModelOptions...,
	), nil
}

func (o *provider) ParseOptions(data map[string]any) (ai.ProviderOptionsData, error) {
	var options ProviderOptions
	if err := ai.ParseOptions(data, &options); err != nil {
		return nil, err
	}
	return &options, nil
}

func (o *provider) Name() string {
	return Name
}
