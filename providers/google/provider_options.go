// Package google provides an implementation of the fantasy AI SDK for Google's language models.
package google

import (
	"encoding/json"

	"charm.land/fantasy"
)

// Global type identifiers for Google-specific provider data.
const (
	TypeProviderOptions = Name + ".options"
)

// ThinkingConfig represents thinking configuration for the Google provider.
type ThinkingConfig struct {
	ThinkingBudget  *int64 `json:"thinking_budget"`
	IncludeThoughts *bool  `json:"include_thoughts"`
}

// SafetySetting represents safety settings for the Google provider.
type SafetySetting struct {
	// 'HARM_CATEGORY_UNSPECIFIED',
	// 'HARM_CATEGORY_HATE_SPEECH',
	// 'HARM_CATEGORY_DANGEROUS_CONTENT',
	// 'HARM_CATEGORY_HARASSMENT',
	// 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
	// 'HARM_CATEGORY_CIVIC_INTEGRITY',
	Category string `json:"category"`

	// 'HARM_BLOCK_THRESHOLD_UNSPECIFIED',
	// 'BLOCK_LOW_AND_ABOVE',
	// 'BLOCK_MEDIUM_AND_ABOVE',
	// 'BLOCK_ONLY_HIGH',
	// 'BLOCK_NONE',
	// 'OFF',
	Threshold string `json:"threshold"`
}

// ProviderOptions represents additional options for the Google provider.
type ProviderOptions struct {
	ThinkingConfig *ThinkingConfig `json:"thinking_config"`

	// Optional.
	// The name of the cached content used as context to serve the prediction.
	// Format: cachedContents/{cachedContent}
	CachedContent string `json:"cached_content"`

	// Optional. A list of unique safety settings for blocking unsafe content.
	SafetySettings []SafetySetting `json:"safety_settings"`
	// 'HARM_BLOCK_THRESHOLD_UNSPECIFIED',
	// 'BLOCK_LOW_AND_ABOVE',
	// 'BLOCK_MEDIUM_AND_ABOVE',
	// 'BLOCK_ONLY_HIGH',
	// 'BLOCK_NONE',
	// 'OFF',
	Threshold string `json:"threshold"`
}

// Options implements the ProviderOptionsData interface for ProviderOptions.
func (o *ProviderOptions) Options() {}

// MarshalJSON implements custom JSON marshaling with type info for ProviderOptions.
func (o ProviderOptions) MarshalJSON() ([]byte, error) {
	type plain ProviderOptions
	raw, err := json.Marshal(plain(o))
	if err != nil {
		return nil, err
	}
	return json.Marshal(struct {
		Type string          `json:"type"`
		Data json.RawMessage `json:"data"`
	}{
		Type: TypeProviderOptions,
		Data: raw,
	})
}

// UnmarshalJSON implements custom JSON unmarshaling with type info for ProviderOptions.
func (o *ProviderOptions) UnmarshalJSON(data []byte) error {
	type plain ProviderOptions
	var oo plain
	err := json.Unmarshal(data, &oo)
	if err != nil {
		return err
	}
	*o = ProviderOptions(oo)
	return nil
}

// ParseOptions parses provider options from a map for the Google provider.
func ParseOptions(data map[string]any) (*ProviderOptions, error) {
	var options ProviderOptions
	if err := fantasy.ParseOptions(data, &options); err != nil {
		return nil, err
	}
	return &options, nil
}

// Register Google provider-specific types with the global registry.
func init() {
	fantasy.RegisterProviderType(TypeProviderOptions, func(data []byte) (fantasy.ProviderOptionsData, error) {
		var v ProviderOptions
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, err
		}
		return &v, nil
	})
}
