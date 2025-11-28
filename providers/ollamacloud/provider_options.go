package ollamacloud

import (
	"charm.land/fantasy"
)

// ProviderOptions represents additional options for the Ollama Cloud provider.
type ProviderOptions struct {
	Think *bool `json:"think,omitempty"`
}

// Options implements the ProviderOptions interface.
func (*ProviderOptions) Options() {}

// NewProviderOptions creates new provider options for the Ollama Cloud provider.
func NewProviderOptions(opts *ProviderOptions) fantasy.ProviderOptions {
	return fantasy.ProviderOptions{
		Name: opts,
	}
}

// ParseOptions parses provider options from a map for Ollama Cloud provider.
func ParseOptions(data map[string]any) (*ProviderOptions, error) {
	var options ProviderOptions
	if err := fantasy.ParseOptions(data, &options); err != nil {
		return nil, err
	}
	return &options, nil
}
