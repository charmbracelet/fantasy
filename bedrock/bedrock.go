package bedrock

import (
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/charmbracelet/fantasy/ai"
	"github.com/charmbracelet/fantasy/anthropic"
)

type options struct {
	skipAuth         bool
	anthropicOptions []anthropic.Option
}

const (
	Name = "bedrock"
)

type Option = func(*options)

func New(opts ...Option) ai.Provider {
	var o options
	for _, opt := range opts {
		opt(&o)
	}
	return anthropic.New(
		append(
			o.anthropicOptions,
			anthropic.WithName(Name),
			anthropic.WithBedrock(),
			anthropic.WithSkipAuth(o.skipAuth),
		)...,
	)
}

func WithHeaders(headers map[string]string) Option {
	return func(o *options) {
		o.anthropicOptions = append(o.anthropicOptions, anthropic.WithHeaders(headers))
	}
}

func WithHTTPClient(client option.HTTPClient) Option {
	return func(o *options) {
		o.anthropicOptions = append(o.anthropicOptions, anthropic.WithHTTPClient(client))
	}
}

func WithSkipAuth(skipAuth bool) Option {
	return func(o *options) {
		o.skipAuth = skipAuth
	}
}
