// Package openaicompat provides an implementation of the fantasy AI SDK for OpenAI-compatible APIs.
package openaicompat

import (
	"encoding/json"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openai"
)

// Global type identifiers for OpenAI-compatible provider data.
const (
	TypeProviderOptions    = Name + ".options"
	TypeContentExtraFields = Name + ".content_extra_fields"
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
	fantasy.RegisterProviderType(TypeContentExtraFields, func(data []byte) (fantasy.ProviderOptionsData, error) {
		var v ContentExtraFields
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

// ContentExtraFields injects non-standard JSON fields onto an individual message
// content block. It is the content-block analogue of ProviderOptions.ExtraBody
// (which injects fields onto the request root), and exists to support provider
// extensions that the OpenAI API format does not model.
//
// When a content part carries this option, its content is automatically serialized
// in array form (rather than as a plain string), and the given fields are merged
// onto the text block.
//
// The primary use case is explicit prompt caching on Alibaba Cloud Qwen 3.6 models,
// which expect a non-standard cache_control field on cached content blocks:
//
//	textPart := fantasy.TextPart{
//	    Text: "Large system prompt or document (min 1024 tokens)...",
//	    ProviderOptions: fantasy.ProviderOptions{
//	        "openai-compat": &openaicompat.ContentExtraFields{
//	            Fields: map[string]any{
//	                "cache_control": map[string]string{"type": "ephemeral"},
//	            },
//	        },
//	    },
//	}
//
// Providers that do not recognize the injected fields will ignore them.
type ContentExtraFields struct {
	Fields map[string]any `json:"fields"`
}

// Options implements the ProviderOptions interface.
func (*ContentExtraFields) Options() {}

// MarshalJSON implements custom JSON marshaling with type info for ContentExtraFields.
func (o ContentExtraFields) MarshalJSON() ([]byte, error) {
	type plain ContentExtraFields
	return fantasy.MarshalProviderType(TypeContentExtraFields, plain(o))
}

// UnmarshalJSON implements custom JSON unmarshaling with type info for ContentExtraFields.
func (o *ContentExtraFields) UnmarshalJSON(data []byte) error {
	type plain ContentExtraFields
	var p plain
	if err := fantasy.UnmarshalProviderType(data, &p); err != nil {
		return err
	}
	*o = ContentExtraFields(p)
	return nil
}

// GetContentExtraFields returns the extra content-block fields from provider options,
// or nil if none are set.
func GetContentExtraFields(providerOptions fantasy.ProviderOptions) map[string]any {
	if options, ok := providerOptions[Name]; ok {
		if options, ok := options.(*ContentExtraFields); ok {
			return options.Fields
		}
	}
	return nil
}

// hasContentExtraFields reports whether extra content-block fields are present on
// either the part-level or message-level provider options.
func hasContentExtraFields(partOpts, msgOpts fantasy.ProviderOptions) bool {
	return len(GetContentExtraFields(partOpts)) > 0 || len(GetContentExtraFields(msgOpts)) > 0
}
