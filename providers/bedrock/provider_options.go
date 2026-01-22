// Package bedrock provides an implementation of the fantasy AI SDK for AWS Bedrock models.
package bedrock

import (
	"encoding/json"

	"charm.land/fantasy"
)

// Global type identifiers for Bedrock-specific provider data.
const (
	TypeProviderOptions         = Name + ".options"
	TypeReasoningOptionMetadata = Name + ".reasoning_metadata"
)

// Register Bedrock provider-specific types with the global registry.
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
}

// ProviderOptions represents additional options for the Bedrock provider.
type ProviderOptions struct {
	// Thinking enables extended thinking/reasoning for models that support it.
	Thinking *ThinkingProviderOption `json:"thinking"`
}

// Options implements the ProviderOptions interface.
func (o *ProviderOptions) Options() {}

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

// ThinkingProviderOption represents thinking options for the Bedrock provider.
type ThinkingProviderOption struct {
	// BudgetTokens sets the maximum number of tokens for reasoning output.
	BudgetTokens int64 `json:"budget_tokens"`
}

// ReasoningOptionMetadata represents reasoning metadata for the Bedrock provider.
type ReasoningOptionMetadata struct {
	// Signature contains the reasoning signature if provided by the model.
	Signature string `json:"signature,omitempty"`
	// RedactedData contains redacted reasoning data if the model redacted content.
	RedactedData string `json:"redacted_data,omitempty"`
}

// Options implements the ProviderOptions interface.
func (*ReasoningOptionMetadata) Options() {}

// MarshalJSON implements custom JSON marshaling with type info for ReasoningOptionMetadata.
func (m ReasoningOptionMetadata) MarshalJSON() ([]byte, error) {
	type plain ReasoningOptionMetadata
	return fantasy.MarshalProviderType(TypeReasoningOptionMetadata, plain(m))
}

// UnmarshalJSON implements custom JSON unmarshaling with type info for ReasoningOptionMetadata.
func (m *ReasoningOptionMetadata) UnmarshalJSON(data []byte) error {
	type plain ReasoningOptionMetadata
	var p plain
	if err := fantasy.UnmarshalProviderType(data, &p); err != nil {
		return err
	}
	*m = ReasoningOptionMetadata(p)
	return nil
}

// NewProviderOptions creates new provider options for the Bedrock provider.
func NewProviderOptions(opts *ProviderOptions) fantasy.ProviderOptions {
	return fantasy.ProviderOptions{
		Name: opts,
	}
}

// ParseOptions parses provider options from a map for the Bedrock provider.
func ParseOptions(data map[string]any) (*ProviderOptions, error) {
	var options ProviderOptions
	if err := fantasy.ParseOptions(data, &options); err != nil {
		return nil, err
	}
	return &options, nil
}
