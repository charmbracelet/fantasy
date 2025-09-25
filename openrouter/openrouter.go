package openrouter

import (
	"encoding/json"

	"github.com/charmbracelet/fantasy/ai"
	"github.com/charmbracelet/fantasy/openai"
	"github.com/openai/openai-go/v2/option"
)

type options struct {
	openaiOptions        []openai.Option
	languageModelOptions []openai.LanguageModelOption
}

const (
	DefaultURL = "https://openrouter.ai/api/v1"
	Name       = "openrouter"
)

type Option = func(*options)

func New(opts ...Option) ai.Provider {
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
			openai.WithLanguageModelMapFinishReasonFunc(languageModelMapFinishReason),
		},
	}
	for _, o := range opts {
		o(&providerOptions)
	}

	providerOptions.openaiOptions = append(providerOptions.openaiOptions, openai.WithLanguageModelOptions(providerOptions.languageModelOptions...))
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

func WithLanguageUniqueToolCallIds() Option {
	return func(l *options) {
		l.languageModelOptions = append(l.languageModelOptions, openai.WithLanguageUniqueToolCallIds())
	}
}

func WithLanguageModelGenerateIDFunc(fn openai.LanguageModelGenerateIDFunc) Option {
	return func(l *options) {
		l.languageModelOptions = append(l.languageModelOptions, openai.WithLanguageModelGenerateIDFunc(fn))
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
