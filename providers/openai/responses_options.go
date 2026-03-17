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
	TypeWebSearchCallMetadata      = Name + ".responses.web_search_call_metadata"
)

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
	fantasy.RegisterProviderType(TypeWebSearchCallMetadata, func(data []byte) (fantasy.ProviderOptionsData, error) {
		var v WebSearchCallMetadata
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, err
		}
		return &v, nil
	})
}

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
	return fantasy.MarshalProviderType(TypeResponsesReasoningMetadata, plain(m))
}

// UnmarshalJSON implements custom JSON unmarshaling with type info for ResponsesReasoningMetadata.
func (m *ResponsesReasoningMetadata) UnmarshalJSON(data []byte) error {
	type plain ResponsesReasoningMetadata
	var p plain
	if err := fantasy.UnmarshalProviderType(data, &p); err != nil {
		return err
	}
	*m = ResponsesReasoningMetadata(p)
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
	return fantasy.MarshalProviderType(TypeResponsesProviderOptions, plain(o))
}

// UnmarshalJSON implements custom JSON unmarshaling with type info for ResponsesProviderOptions.
func (o *ResponsesProviderOptions) UnmarshalJSON(data []byte) error {
	type plain ResponsesProviderOptions
	var p plain
	if err := fantasy.UnmarshalProviderType(data, &p); err != nil {
		return err
	}
	*o = ResponsesProviderOptions(p)
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
	"gpt-5-chat",
	"gpt-5-pro",
	"gpt-5.1",
	"gpt-5.1-codex",
	"gpt-5.1-codex-max",
	"gpt-5.1-codex-mini",
	"gpt-5.1-chat",
	"gpt-5.2",
	"gpt-5.2-codex",
	"gpt-5.3",
	"gpt-5.3-codex",
	"gpt-5.4",
	"gpt-5.4-pro",
	"gpt-5.4-mini",
	"gpt-5.4-nano",
	"gpt-5.4-codex",
	"gpt-oss-120b",
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

// SearchContextSize controls how much context window space the
// web search tool uses. Maps to the OpenAI API's
// search_context_size parameter.
type SearchContextSize string

const (
	// SearchContextSizeLow uses minimal context for search results.
	SearchContextSizeLow SearchContextSize = "low"
	// SearchContextSizeMedium is the default context size.
	SearchContextSizeMedium SearchContextSize = "medium"
	// SearchContextSizeHigh uses maximal context for search results.
	SearchContextSizeHigh SearchContextSize = "high"
)

// WebSearchUserLocation provides geographic context for more
// relevant web search results.
type WebSearchUserLocation struct {
	City     string `json:"city,omitempty"`
	Region   string `json:"region,omitempty"`
	Country  string `json:"country,omitempty"`
	Timezone string `json:"timezone,omitempty"`
}

// WebSearchToolOptions configures the OpenAI web search tool.
type WebSearchToolOptions struct {
	// SearchContextSize controls the amount of context window
	// space used for search results. Defaults to medium.
	SearchContextSize SearchContextSize
	// AllowedDomains restricts search results to these domains.
	// Subdomains are included automatically.
	AllowedDomains []string
	// UserLocation provides geographic context for more
	// relevant search results.
	UserLocation *WebSearchUserLocation
}

// WebSearchTool creates a provider-defined web search tool for
// OpenAI models. Pass nil for default options.
func WebSearchTool(opts *WebSearchToolOptions) fantasy.ProviderDefinedTool {
	tool := fantasy.ProviderDefinedTool{
		ID:   "web_search",
		Name: "web_search",
	}
	if opts == nil {
		return tool
	}
	args := map[string]any{}
	if opts.SearchContextSize != "" {
		args["search_context_size"] = opts.SearchContextSize
	}
	if len(opts.AllowedDomains) > 0 {
		args["allowed_domains"] = opts.AllowedDomains
	}
	if opts.UserLocation != nil {
		args["user_location"] = opts.UserLocation
	}
	if len(args) > 0 {
		tool.Args = args
	}
	return tool
}

// WebSearchSource represents a single source from a web search action.
type WebSearchSource struct {
	Type string `json:"type"`
	URL  string `json:"url"`
}

// WebSearchAction represents the action taken during a web search call.
type WebSearchAction struct {
	// Type is the kind of action: "search", "open_page", or "find".
	Type string `json:"type"`
	// Query is the search query (present when Type is "search").
	Query string `json:"query,omitempty"`
	// Sources are the results returned by the search.
	Sources []WebSearchSource `json:"sources,omitempty"`
}

// WebSearchCallMetadata stores structured data from a web_search_call
// output item for round-tripping through multi-turn conversations.
// The ItemID is used with item_reference for efficient round-tripping
// when response storage is enabled.
type WebSearchCallMetadata struct {
	// ItemID is the server-side ID of the web_search_call output item.
	ItemID string `json:"item_id"`
	// Action contains the structured action data from the search.
	Action *WebSearchAction `json:"action,omitempty"`
}

// Options implements the ProviderOptionsData interface.
func (*WebSearchCallMetadata) Options() {}

// MarshalJSON implements custom JSON marshaling with type info.
func (m WebSearchCallMetadata) MarshalJSON() ([]byte, error) {
	type plain WebSearchCallMetadata
	return fantasy.MarshalProviderType(TypeWebSearchCallMetadata, plain(m))
}

// UnmarshalJSON implements custom JSON unmarshaling with type info.
func (m *WebSearchCallMetadata) UnmarshalJSON(data []byte) error {
	type plain WebSearchCallMetadata
	var p plain
	if err := fantasy.UnmarshalProviderType(data, &p); err != nil {
		return err
	}
	*m = WebSearchCallMetadata(p)
	return nil
}
