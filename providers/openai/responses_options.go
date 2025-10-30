// Package openai provides an implementation of the fantasy AI SDK for OpenAI's language models.
package openai

import (
	"encoding/json"
	"slices"

	"charm.land/fantasy"
)

// Global type identifiers for OpenAI Responses API-specific data.
const (
	TypeResponsesProviderOptions   = Name + ".responses.options"
	TypeResponsesReasoningMetadata = Name + ".responses.reasoning_metadata"
)

// ResponsesReasoningMetadata represents reasoning metadata for OpenAI Responses API.
type ResponsesReasoningMetadata struct {
	ItemID           string   `json:"item_id"`
	EncryptedContent *string  `json:"encrypted_content"`
	Summary          []string `json:"summary"`
}

// Options implements the ProviderOptions interface.
func (*ResponsesReasoningMetadata) Options() {}

// MarshalJSON implements custom JSON marshaling with type info for ResponsesReasoningMetadata.
func (m ResponsesReasoningMetadata) MarshalJSON() ([]byte, error) {
	type plain ResponsesReasoningMetadata
	raw, err := json.Marshal(plain(m))
	if err != nil {
		return nil, err
	}
	return json.Marshal(struct {
		Type string          `json:"type"`
		Data json.RawMessage `json:"data"`
	}{
		Type: TypeResponsesReasoningMetadata,
		Data: raw,
	})
}

// UnmarshalJSON implements custom JSON unmarshaling with type info for ResponsesReasoningMetadata.
func (m *ResponsesReasoningMetadata) UnmarshalJSON(data []byte) error {
	type plain ResponsesReasoningMetadata
	var rm plain
	err := json.Unmarshal(data, &rm)
	if err != nil {
		return err
	}
	*m = ResponsesReasoningMetadata(rm)
	return nil
}

// IncludeType represents the type of content to include for OpenAI Responses API.
type IncludeType string

const (
	// IncludeReasoningEncryptedContent includes encrypted reasoning content.
	IncludeReasoningEncryptedContent IncludeType = "reasoning.encrypted_content"
	// IncludeFileSearchCallResults includes file search call results.
	IncludeFileSearchCallResults IncludeType = "file_search_call.results"
	// IncludeMessageOutputTextLogprobs includes message output text log probabilities.
	IncludeMessageOutputTextLogprobs IncludeType = "message.output_text.logprobs"
)

// ServiceTier represents the service tier for OpenAI Responses API.
type ServiceTier string

const (
	// ServiceTierAuto represents the auto service tier.
	ServiceTierAuto ServiceTier = "auto"
	// ServiceTierFlex represents the flex service tier.
	ServiceTierFlex ServiceTier = "flex"
	// ServiceTierPriority represents the priority service tier.
	ServiceTierPriority ServiceTier = "priority"
)

// TextVerbosity represents the text verbosity level for OpenAI Responses API.
type TextVerbosity string

const (
	// TextVerbosityLow represents low text verbosity.
	TextVerbosityLow TextVerbosity = "low"
	// TextVerbosityMedium represents medium text verbosity.
	TextVerbosityMedium TextVerbosity = "medium"
	// TextVerbosityHigh represents high text verbosity.
	TextVerbosityHigh TextVerbosity = "high"
)

// ResponsesProviderOptions represents additional options for OpenAI Responses API.
type ResponsesProviderOptions struct {
	Include           []IncludeType    `json:"include"`
	Instructions      *string          `json:"instructions"`
	Logprobs          any              `json:"logprobs"`
	MaxToolCalls      *int64           `json:"max_tool_calls"`
	Metadata          map[string]any   `json:"metadata"`
	ParallelToolCalls *bool            `json:"parallel_tool_calls"`
	PromptCacheKey    *string          `json:"prompt_cache_key"`
	ReasoningEffort   *ReasoningEffort `json:"reasoning_effort"`
	ReasoningSummary  *string          `json:"reasoning_summary"`
	SafetyIdentifier  *string          `json:"safety_identifier"`
	ServiceTier       *ServiceTier     `json:"service_tier"`
	StrictJSONSchema  *bool            `json:"strict_json_schema"`
	TextVerbosity     *TextVerbosity   `json:"text_verbosity"`
	User              *string          `json:"user"`
}

// Options implements the ProviderOptions interface.
func (*ResponsesProviderOptions) Options() {}

// MarshalJSON implements custom JSON marshaling with type info for ResponsesProviderOptions.
func (o ResponsesProviderOptions) MarshalJSON() ([]byte, error) {
	type plain ResponsesProviderOptions
	raw, err := json.Marshal(plain(o))
	if err != nil {
		return nil, err
	}
	return json.Marshal(struct {
		Type string          `json:"type"`
		Data json.RawMessage `json:"data"`
	}{
		Type: TypeResponsesProviderOptions,
		Data: raw,
	})
}

// UnmarshalJSON implements custom JSON unmarshaling with type info for ResponsesProviderOptions.
func (o *ResponsesProviderOptions) UnmarshalJSON(data []byte) error {
	type plain ResponsesProviderOptions
	var ro plain
	err := json.Unmarshal(data, &ro)
	if err != nil {
		return err
	}
	*o = ResponsesProviderOptions(ro)
	return nil
}

// responsesReasoningModelIds lists the model IDs that support reasoning for OpenAI Responses API.
var responsesReasoningModelIDs = []string{
	"o1",
	"o1-2024-12-17",
	"o3-mini",
	"o3-mini-2025-01-31",
	"o3",
	"o3-2025-04-16",
	"o4-mini",
	"o4-mini-2025-04-16",
	"codex-mini-latest",
	"gpt-5",
	"gpt-5-2025-08-07",
	"gpt-5-mini",
	"gpt-5-mini-2025-08-07",
	"gpt-5-nano",
	"gpt-5-nano-2025-08-07",
	"gpt-5-codex",
}

// responsesModelIds lists all model IDs for OpenAI Responses API.
var responsesModelIDs = append([]string{
	"gpt-4.1",
	"gpt-4.1-2025-04-14",
	"gpt-4.1-mini",
	"gpt-4.1-mini-2025-04-14",
	"gpt-4.1-nano",
	"gpt-4.1-nano-2025-04-14",
	"gpt-4o",
	"gpt-4o-2024-05-13",
	"gpt-4o-2024-08-06",
	"gpt-4o-2024-11-20",
	"gpt-4o-mini",
	"gpt-4o-mini-2024-07-18",
	"gpt-4-turbo",
	"gpt-4-turbo-2024-04-09",
	"gpt-4-turbo-preview",
	"gpt-4-0125-preview",
	"gpt-4-1106-preview",
	"gpt-4",
	"gpt-4-0613",
	"gpt-4.5-preview",
	"gpt-4.5-preview-2025-02-27",
	"gpt-3.5-turbo-0125",
	"gpt-3.5-turbo",
	"gpt-3.5-turbo-1106",
	"chatgpt-4o-latest",
	"gpt-5-chat-latest",
}, responsesReasoningModelIDs...)

// NewResponsesProviderOptions creates new provider options for OpenAI Responses API.
func NewResponsesProviderOptions(opts *ResponsesProviderOptions) fantasy.ProviderOptions {
	return fantasy.ProviderOptions{
		Name: opts,
	}
}

// ParseResponsesOptions parses provider options from a map for OpenAI Responses API.
func ParseResponsesOptions(data map[string]any) (*ResponsesProviderOptions, error) {
	var options ResponsesProviderOptions
	if err := fantasy.ParseOptions(data, &options); err != nil {
		return nil, err
	}
	return &options, nil
}

// IsResponsesModel checks if a model ID is a Responses API model for OpenAI.
func IsResponsesModel(modelID string) bool {
	return slices.Contains(responsesModelIDs, modelID)
}

// IsResponsesReasoningModel checks if a model ID is a Responses API reasoning model for OpenAI.
func IsResponsesReasoningModel(modelID string) bool {
	return slices.Contains(responsesReasoningModelIDs, modelID)
}

// Register OpenAI Responses API-specific types with the global registry.
func init() {
	fantasy.RegisterProviderType(TypeResponsesProviderOptions, func(data []byte) (fantasy.ProviderOptionsData, error) {
		var v ResponsesProviderOptions
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, err
		}
		return &v, nil
	})
	fantasy.RegisterProviderType(TypeResponsesReasoningMetadata, func(data []byte) (fantasy.ProviderOptionsData, error) {
		var v ResponsesReasoningMetadata
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, err
		}
		return &v, nil
	})
}
