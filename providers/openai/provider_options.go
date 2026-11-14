// Package openai provides an implementation of the fantasy AI SDK for OpenAI's language models.
package openai

import (
	"encoding/json"

	"charm.land/fantasy"
	"github.com/charmbracelet/openai-go"
	"github.com/charmbracelet/openai-go/packages/respjson"
)

// ReasoningEffort represents the reasoning effort level for OpenAI models.
type ReasoningEffort string

const (
	// ReasoningEffortNone represents ReasoningEffortNone reasoning effort.
	ReasoningEffortNone ReasoningEffort = "none"
	// ReasoningEffortMinimal represents minimal reasoning effort.
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	// ReasoningEffortLow represents low reasoning effort.
	ReasoningEffortLow ReasoningEffort = "low"
	// ReasoningEffortMedium represents medium reasoning effort.
	ReasoningEffortMedium ReasoningEffort = "medium"
	// ReasoningEffortHigh represents high reasoning effort.
	ReasoningEffortHigh ReasoningEffort = "high"
	// ReasoningEffortXHigh represents extra-high reasoning effort.
	ReasoningEffortXHigh ReasoningEffort = "xhigh"
	// ReasoningEffortMax represents maximum reasoning effort.
	ReasoningEffortMax ReasoningEffort = "max"
)

// Global type identifiers for OpenAI-specific provider data.
const (
	TypeProviderOptions      = Name + ".options"
	TypeProviderFileOptions  = Name + ".file_options"
	TypeProviderMetadata     = Name + ".metadata"
	TypeToolCallExtraContent = Name + ".tool_call_extra_content"
)

// Register OpenAI provider-specific types with the global registry.
func init() {
	fantasy.RegisterProviderType(TypeProviderOptions, func(data []byte) (fantasy.ProviderOptionsData, error) {
		var v ProviderOptions
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, err
		}
		return &v, nil
	})
	fantasy.RegisterProviderType(TypeProviderFileOptions, func(data []byte) (fantasy.ProviderOptionsData, error) {
		var v ProviderFileOptions
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, err
		}
		return &v, nil
	})
	fantasy.RegisterProviderType(TypeProviderMetadata, func(data []byte) (fantasy.ProviderOptionsData, error) {
		var v ProviderMetadata
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, err
		}
		return &v, nil
	})
	fantasy.RegisterProviderType(TypeToolCallExtraContent, func(data []byte) (fantasy.ProviderOptionsData, error) {
		var v ToolCallExtraContent
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, err
		}
		return &v, nil
	})
}

// ProviderMetadata represents additional metadata from OpenAI provider.
type ProviderMetadata struct {
	Logprobs                 []openai.ChatCompletionTokenLogprob `json:"logprobs"`
	AcceptedPredictionTokens int64                               `json:"accepted_prediction_tokens"`
	RejectedPredictionTokens int64                               `json:"rejected_prediction_tokens"`
	// ExtraFields captures non-standard fields from the usage object.
	// Keys are field names, values are raw JSON.
	ExtraFields map[string]json.RawMessage `json:"extra_fields,omitempty"`
}

// ExtraField parses an extra usage field into the provided target.
// Returns false if the field is not present or cannot be parsed.
func (m *ProviderMetadata) ExtraField(key string, target any) bool {
	if m == nil || m.ExtraFields == nil {
		return false
	}
	raw, ok := m.ExtraFields[key]
	if !ok {
		return false
	}
	return json.Unmarshal(raw, target) == nil
}

// ExtractExtraFields reads non-standard fields from the SDK's
// ExtraFields map and returns them as a map of raw JSON values.
func ExtractExtraFields(extraFields map[string]respjson.Field) map[string]json.RawMessage {
	if len(extraFields) == 0 {
		return nil
	}
	ext := make(map[string]json.RawMessage, len(extraFields))
	for k, f := range extraFields {
		if raw := f.Raw(); raw != "" && raw != "null" {
			ext[k] = json.RawMessage(raw)
		}
	}
	if len(ext) == 0 {
		return nil
	}
	return ext
}

// Options implements the ProviderOptions interface.
func (*ProviderMetadata) Options() {}

// MarshalJSON implements custom JSON marshaling with type info for ProviderMetadata.
func (m ProviderMetadata) MarshalJSON() ([]byte, error) {
	type plain ProviderMetadata
	return fantasy.MarshalProviderType(TypeProviderMetadata, plain(m))
}

// UnmarshalJSON implements custom JSON unmarshaling with type info for ProviderMetadata.
func (m *ProviderMetadata) UnmarshalJSON(data []byte) error {
	type plain ProviderMetadata
	var p plain
	if err := fantasy.UnmarshalProviderType(data, &p); err != nil {
		return err
	}
	*m = ProviderMetadata(p)
	return nil
}

// ToolCallExtraContent carries the provider-specific extra_content object
// attached to an assistant tool call. Gemini 3 models put a thought_signature
// there and require it to be replayed verbatim on the same tool call in later
// requests; dropping it makes the follow-up turn fail with "Function call is
// missing a thought_signature". It is opaque to fantasy, so keep it as raw
// JSON.
type ToolCallExtraContent struct {
	ExtraContent json.RawMessage `json:"extra_content,omitempty"`
}

// Options implements the ProviderOptionsData interface.
func (*ToolCallExtraContent) Options() {}

// MarshalJSON implements custom JSON marshaling with type info for ToolCallExtraContent.
func (m ToolCallExtraContent) MarshalJSON() ([]byte, error) {
	type plain ToolCallExtraContent
	return fantasy.MarshalProviderType(TypeToolCallExtraContent, plain(m))
}

// UnmarshalJSON implements custom JSON unmarshaling with type info for ToolCallExtraContent.
func (m *ToolCallExtraContent) UnmarshalJSON(data []byte) error {
	type plain ToolCallExtraContent
	var p plain
	if err := fantasy.UnmarshalProviderType(data, &p); err != nil {
		return err
	}
	*m = ToolCallExtraContent(p)
	return nil
}

// ToolCallExtraFields returns the provider-specific extra fields to attach to
// a replayed tool call param, or nil when the part carries no extra_content.
// Gemini 3 models reject the follow-up turn unless the thought_signature that
// arrived with the tool call is echoed back on that exact call.
func ToolCallExtraFields(options fantasy.ProviderOptions) map[string]any {
	extra, ok := options[Name].(*ToolCallExtraContent)
	if !ok || len(extra.ExtraContent) == 0 {
		return nil
	}
	return map[string]any{"extra_content": extra.ExtraContent}
}

// ProviderOptions represents additional options for OpenAI provider.
type ProviderOptions struct {
	LogitBias           map[string]int64 `json:"logit_bias"`
	LogProbs            *bool            `json:"log_probs"`
	TopLogProbs         *int64           `json:"top_log_probs"`
	ParallelToolCalls   *bool            `json:"parallel_tool_calls"`
	User                *string          `json:"user"`
	ReasoningEffort     *ReasoningEffort `json:"reasoning_effort"`
	MaxCompletionTokens *int64           `json:"max_completion_tokens"`
	TextVerbosity       *string          `json:"text_verbosity"`
	Prediction          map[string]any   `json:"prediction"`
	Store               *bool            `json:"store"`
	Metadata            map[string]any   `json:"metadata"`
	PromptCacheKey      *string          `json:"prompt_cache_key"`
	SafetyIdentifier    *string          `json:"safety_identifier"`
	ServiceTier         *string          `json:"service_tier"`
	StructuredOutputs   *bool            `json:"structured_outputs"`
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

// ProviderFileOptions represents file options for OpenAI provider.
type ProviderFileOptions struct {
	ImageDetail string `json:"image_detail"`
}

// Options implements the ProviderOptions interface.
func (*ProviderFileOptions) Options() {}

// MarshalJSON implements custom JSON marshaling with type info for ProviderFileOptions.
func (o ProviderFileOptions) MarshalJSON() ([]byte, error) {
	type plain ProviderFileOptions
	return fantasy.MarshalProviderType(TypeProviderFileOptions, plain(o))
}

// UnmarshalJSON implements custom JSON unmarshaling with type info for ProviderFileOptions.
func (o *ProviderFileOptions) UnmarshalJSON(data []byte) error {
	type plain ProviderFileOptions
	var p plain
	if err := fantasy.UnmarshalProviderType(data, &p); err != nil {
		return err
	}
	*o = ProviderFileOptions(p)
	return nil
}

// ReasoningEffortOption creates a pointer to a ReasoningEffort value.
//
//go:fix inline
func ReasoningEffortOption(e ReasoningEffort) *ReasoningEffort {
	return new(e)
}

// NewProviderOptions creates new provider options for OpenAI.
func NewProviderOptions(opts *ProviderOptions) fantasy.ProviderOptions {
	return fantasy.ProviderOptions{
		Name: opts,
	}
}

// NewProviderFileOptions creates new file options for OpenAI.
func NewProviderFileOptions(opts *ProviderFileOptions) fantasy.ProviderOptions {
	return fantasy.ProviderOptions{
		Name: opts,
	}
}

// ParseOptions parses provider options from a map.
func ParseOptions(data map[string]any) (*ProviderOptions, error) {
	var options ProviderOptions
	if err := fantasy.ParseOptions(data, &options); err != nil {
		return nil, err
	}
	return &options, nil
}
