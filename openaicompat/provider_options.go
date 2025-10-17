package openaicompat

import (
	"charm.land/fantasy"
	"charm.land/fantasy/openai"
)

type ProviderOptions struct {
	User            *string                 `json:"user"`
	ReasoningEffort *openai.ReasoningEffort `json:"reasoning_effort"`
}

type ReasoningData struct {
	ReasoningContent string `json:"reasoning_content"`
}

func (*ProviderOptions) Options() {}

func NewProviderOptions(opts *ProviderOptions) fantasy.ProviderOptions {
	return fantasy.ProviderOptions{
		Name: opts,
	}
}

func ParseOptions(data map[string]any) (*ProviderOptions, error) {
	var options ProviderOptions
	if err := fantasy.ParseOptions(data, &options); err != nil {
		return nil, err
	}
	return &options, nil
}
