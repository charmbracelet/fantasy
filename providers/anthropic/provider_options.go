// Package anthropic provides an implementation of the fantasy AI SDK for Anthropic's language models.
package anthropic

import (
	"encoding/json"

	"charm.land/fantasy"
)

// Global type identifiers for Anthropic-specific provider data.
const (
	TypeProviderOptions         = Name + ".options"
	TypeReasoningOptionMetadata = Name + ".reasoning_metadata"
	TypeProviderCacheControl    = Name + ".cache_control_options"
)

// ProviderOptions represents additional options for the Anthropic provider.
type ProviderOptions struct {
	SendReasoning          *bool                   `json:"send_reasoning"`
	Thinking               *ThinkingProviderOption `json:"thinking"`
	DisableParallelToolUse *bool                   `json:"disable_parallel_tool_use"`
}

// Options implements the ProviderOptions interface.
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

// ThinkingProviderOption represents thinking options for the Anthropic provider.
type ThinkingProviderOption struct {
	BudgetTokens int64 `json:"budget_tokens"`
}

// ReasoningOptionMetadata represents reasoning metadata for the Anthropic provider.
type ReasoningOptionMetadata struct {
	Signature    string `json:"signature"`
	RedactedData string `json:"redacted_data"`
}

// Options implements the ProviderOptions interface.
func (*ReasoningOptionMetadata) Options() {}

// MarshalJSON implements custom JSON marshaling with type info for ReasoningOptionMetadata.
func (m ReasoningOptionMetadata) MarshalJSON() ([]byte, error) {
	type plain ReasoningOptionMetadata
	raw, err := json.Marshal(plain(m))
	if err != nil {
		return nil, err
	}
	return json.Marshal(struct {
		Type string          `json:"type"`
		Data json.RawMessage `json:"data"`
	}{
		Type: TypeReasoningOptionMetadata,
		Data: raw,
	})
}

// UnmarshalJSON implements custom JSON unmarshaling with type info for ReasoningOptionMetadata.
func (m *ReasoningOptionMetadata) UnmarshalJSON(data []byte) error {
	type plain ReasoningOptionMetadata
	var rm plain
	err := json.Unmarshal(data, &rm)
	if err != nil {
		return err
	}
	*m = ReasoningOptionMetadata(rm)
	return nil
}

// ProviderCacheControlOptions represents cache control options for the Anthropic provider.
type ProviderCacheControlOptions struct {
	CacheControl CacheControl `json:"cache_control"`
}

// Options implements the ProviderOptions interface.
func (*ProviderCacheControlOptions) Options() {}

// MarshalJSON implements custom JSON marshaling with type info for ProviderCacheControlOptions.
func (o ProviderCacheControlOptions) MarshalJSON() ([]byte, error) {
	type plain ProviderCacheControlOptions
	raw, err := json.Marshal(plain(o))
	if err != nil {
		return nil, err
	}
	return json.Marshal(struct {
		Type string          `json:"type"`
		Data json.RawMessage `json:"data"`
	}{
		Type: TypeProviderCacheControl,
		Data: raw,
	})
}

// UnmarshalJSON implements custom JSON unmarshaling with type info for ProviderCacheControlOptions.
func (o *ProviderCacheControlOptions) UnmarshalJSON(data []byte) error {
	type plain ProviderCacheControlOptions
	var cc plain
	err := json.Unmarshal(data, &cc)
	if err != nil {
		return err
	}
	*o = ProviderCacheControlOptions(cc)
	return nil
}

// CacheControl represents cache control settings for the Anthropic provider.
type CacheControl struct {
	Type string `json:"type"`
}

// NewProviderOptions creates new provider options for the Anthropic provider.
func NewProviderOptions(opts *ProviderOptions) fantasy.ProviderOptions {
	return fantasy.ProviderOptions{
		Name: opts,
	}
}

// NewProviderCacheControlOptions creates new cache control options for the Anthropic provider.
func NewProviderCacheControlOptions(opts *ProviderCacheControlOptions) fantasy.ProviderOptions {
	return fantasy.ProviderOptions{
		Name: opts,
	}
}

// ParseOptions parses provider options from a map for the Anthropic provider.
func ParseOptions(data map[string]any) (*ProviderOptions, error) {
	var options ProviderOptions
	if err := fantasy.ParseOptions(data, &options); err != nil {
		return nil, err
	}
	return &options, nil
}

// Register Anthropic provider-specific types with the global registry.
func init() {
	fantasy.RegisterProviderType(TypeProviderOptions, func(data []byte) (fantasy.ProviderOptionsData, error) {
		var v ProviderOptions
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, err
		}
		return &v, nil
	})
	fantasy.RegisterProviderType(TypeReasoningOptionMetadata, func(data []byte) (fantasy.ProviderOptionsData, error) {
		var v ReasoningOptionMetadata
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, err
		}
		return &v, nil
	})
	fantasy.RegisterProviderType(TypeProviderCacheControl, func(data []byte) (fantasy.ProviderOptionsData, error) {
		var v ProviderCacheControlOptions
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, err
		}
		return &v, nil
	})
}
