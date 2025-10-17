// Package openaicompat provides an implementation of the fantasy AI SDK for OpenAI-compatible APIs.
package openaicompat

import (
	"charm.land/fantasy"
	"charm.land/fantasy/providers/openai"
)

// ProviderOptions represents additional options for the OpenAI-compatible provider.
type ProviderOptions struct {
	User            *string                 `json:"user"`
	ReasoningEffort *openai.ReasoningEffort `json:"reasoning_effort"`
}

// ReasoningData represents reasoning data for OpenAI-compatible provider.
type ReasoningData struct {
	ReasoningContent string `json:"reasoning_content"`
}

// Options implements the ProviderOptions interface.
func (*ProviderOptions) Options() {}

// NewProviderOptions creates new provider options for the OpenAI-compatible provider.
func NewProviderOptions(opts *ProviderOptions) fantasy.ProviderOptions {
	return fantasy.ProviderOptions{
		Name: opts,
	}
}

// ParseOptions parses provider options from a map for OpenAI-compatible provider.
func ParseOptions(data map[string]any) (*ProviderOptions, error) {
	var options ProviderOptions
	if err := fantasy.ParseOptions(data, &options); err != nil {
		return nil, err
	}
	return &options, nil
}
