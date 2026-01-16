// Package bedrock provides an implementation of the fantasy AI SDK for AWS Bedrock's language models.
package bedrock

import (
	"context"
	"fmt"
	"strings"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/anthropic"
	"github.com/charmbracelet/anthropic-sdk-go/option"
)

type options struct {
	skipAuth         bool
	anthropicOptions []anthropic.Option
	headers          map[string]string
	client           option.HTTPClient
}

const (
	// Name is the name of the Bedrock provider.
	Name = "bedrock"
)

// Option defines a function that configures Bedrock provider options.
type Option = func(*options)

type provider struct {
	options           options
	anthropicProvider fantasy.Provider
}

// New creates a new Bedrock provider with the given options.
func New(opts ...Option) (fantasy.Provider, error) {
	var o options
	for _, opt := range opts {
		opt(&o)
	}

	// Create Anthropic provider for anthropic.* models
	anthropicProvider, err := anthropic.New(
		append(
			o.anthropicOptions,
			anthropic.WithName(Name),
			anthropic.WithBedrock(),
			anthropic.WithSkipAuth(o.skipAuth),
		)...,
	)
	if err != nil {
		return nil, err
	}

	return &provider{
		options:           o,
		anthropicProvider: anthropicProvider,
	}, nil
}

// Name returns the provider name.
func (p *provider) Name() string {
	return Name
}

// LanguageModel routes to the appropriate SDK based on model ID prefix.
func (p *provider) LanguageModel(ctx context.Context, modelID string) (fantasy.LanguageModel, error) {
	if strings.HasPrefix(modelID, "anthropic.") {
		// Use Anthropic SDK (existing behavior)
		return p.anthropicProvider.LanguageModel(ctx, modelID)
	} else if strings.HasPrefix(modelID, "amazon.") {
		// Use AWS SDK Converse API (new behavior)
		return p.createNovaModel(ctx, modelID)
	}
	return nil, fmt.Errorf("unsupported model prefix for Bedrock: %s", modelID)
}

// createNovaModel creates a language model instance for Nova models.
// This is a stub that will be implemented in task 5.
func (p *provider) createNovaModel(ctx context.Context, modelID string) (fantasy.LanguageModel, error) {
	return nil, fmt.Errorf("Nova model support not yet implemented")
}

// WithAPIKey sets the access token for the Bedrock provider.
func WithAPIKey(apiKey string) Option {
	return func(o *options) {
		o.anthropicOptions = append(o.anthropicOptions, anthropic.WithAPIKey(apiKey))
	}
}

// WithHeaders sets the headers for the Bedrock provider.
func WithHeaders(headers map[string]string) Option {
	return func(o *options) {
		o.headers = headers
		o.anthropicOptions = append(o.anthropicOptions, anthropic.WithHeaders(headers))
	}
}

// WithHTTPClient sets the HTTP client for the Bedrock provider.
func WithHTTPClient(client option.HTTPClient) Option {
	return func(o *options) {
		o.client = client
		o.anthropicOptions = append(o.anthropicOptions, anthropic.WithHTTPClient(client))
	}
}

// WithSkipAuth configures whether to skip authentication for the Bedrock provider.
func WithSkipAuth(skipAuth bool) Option {
	return func(o *options) {
		o.skipAuth = skipAuth
	}
}
