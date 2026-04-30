// Package openaicompat provides an implementation of the fantasy AI SDK for OpenAI-compatible APIs.
package openaicompat

import (
	"encoding/json"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openai"
)

// Global type identifiers for OpenAI-compatible provider data.
const (
	TypeProviderOptions = Name + ".options"
)

// Register OpenAI-compatible provider-specific types with the global registry.
func init() {
	fantasy.RegisterProviderType(TypeProviderOptions, func(data []byte) (fantasy.ProviderOptionsData, error) {
		var v ProviderOptions
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, err
		}
		return &v, nil
	})
}

// ProviderOptions represents additional options for the OpenAI-compatible provider.
type ProviderOptions struct {
	User            *string                 `json:"user"`
	ReasoningEffort *openai.ReasoningEffort `json:"reasoning_effort"`
	ExtraBody       map[string]any          `json:"extra_body,omitempty"`
}

// ReasoningData represents reasoning data for OpenAI-compatible provider.
// Some providers use "reasoning_content" (e.g. Avian), others use "reasoning" (e.g. Moonshot AI/Kimi).
type ReasoningData struct {
	ReasoningContent string `json:"reasoning_content"`
	Reasoning        string `json:"reasoning"`
}

// GetReasoningContent returns the reasoning text from whichever field is populated.
func (r ReasoningData) GetReasoningContent() string {
	if r.ReasoningContent != "" {
		return r.ReasoningContent
	}
	return r.Reasoning
}

// Options implements the ProviderOptions interface.
func (*ProviderOptions) Options() {}

// MarshalJSON implements custom JSON marshaling with type info for ProviderOptions.
func (o ProviderOptions) MarshalJSON() ([]byte, error) {
	type plain ProviderOptions
	return fantasy.MarshalProviderType(TypeProviderOptions, plain(o))
}

// UnmarshalJSON implements custom JSON unmarshaling with type info for ProviderOptions.
func (o *ProviderOptions) UnmarshalJSON(data []byte) error {
	type plain ProviderOptions
	var p plain
	if err := fantasy.UnmarshalProviderType(data, &p); err != nil {
		return err
	}
	*o = ProviderOptions(p)
	return nil
}

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
