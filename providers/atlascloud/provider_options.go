package atlascloud

import (
	"charm.land/fantasy"
	"charm.land/fantasy/providers/openaicompat"
)

// ProviderOptions represents additional options for the Atlas Cloud provider.
type ProviderOptions = openaicompat.ProviderOptions

// NewProviderOptions creates new provider options for Atlas Cloud.
func NewProviderOptions(opts *ProviderOptions) fantasy.ProviderOptions {
	return fantasy.ProviderOptions{
		Name: opts,
	}
}

// ParseOptions parses provider options from a map for Atlas Cloud.
func ParseOptions(data map[string]any) (*ProviderOptions, error) {
	return openaicompat.ParseOptions(data)
}
