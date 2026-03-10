// Package anthropic provides an implementation of the fantasy AI SDK for Anthropic's language models.
package anthropic

import (
	"encoding/json"

	"charm.land/fantasy"
)

// Effort represents the output effort level for Anthropic models.
//
// This maps to Messages API `output_config.effort`.
type Effort string

const (
	// EffortLow represents low output effort.
	EffortLow Effort = "low"
	// EffortMedium represents medium output effort.
	EffortMedium Effort = "medium"
	// EffortHigh represents high output effort.
	EffortHigh Effort = "high"
	// EffortMax represents maximum output effort.
	EffortMax Effort = "max"
)

// Global type identifiers for Anthropic-specific provider data.
const (
	TypeProviderOptions         = Name + ".options"
	TypeReasoningOptionMetadata = Name + ".reasoning_metadata"
	TypeProviderCacheControl    = Name + ".cache_control_options"
	TypeWebSearchResultMetadata = Name + ".web_search_result_metadata"
)

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
	fantasy.RegisterProviderType(TypeWebSearchResultMetadata, func(data []byte) (fantasy.ProviderOptionsData, error) {
		var v WebSearchResultMetadata
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, err
		}
		return &v, nil
	})
}

// ProviderOptions represents additional options for the Anthropic provider.
type ProviderOptions struct {
	SendReasoning          *bool                   `json:"send_reasoning"`
	Thinking               *ThinkingProviderOption `json:"thinking"`
	Effort                 *Effort                 `json:"effort"`
	DisableParallelToolUse *bool                   `json:"disable_parallel_tool_use"`
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

// ProviderCacheControlOptions represents cache control options for the Anthropic provider.
type ProviderCacheControlOptions struct {
	CacheControl CacheControl `json:"cache_control"`
}

// Options implements the ProviderOptions interface.
func (*ProviderCacheControlOptions) Options() {}

// MarshalJSON implements custom JSON marshaling with type info for ProviderCacheControlOptions.
func (o ProviderCacheControlOptions) MarshalJSON() ([]byte, error) {
	type plain ProviderCacheControlOptions
	return fantasy.MarshalProviderType(TypeProviderCacheControl, plain(o))
}

// UnmarshalJSON implements custom JSON unmarshaling with type info for ProviderCacheControlOptions.
func (o *ProviderCacheControlOptions) UnmarshalJSON(data []byte) error {
	type plain ProviderCacheControlOptions
	var p plain
	if err := fantasy.UnmarshalProviderType(data, &p); err != nil {
		return err
	}
	*o = ProviderCacheControlOptions(p)
	return nil
}

// WebSearchResultItem represents a single web search result for round-tripping.
type WebSearchResultItem struct {
	URL              string `json:"url"`
	Title            string `json:"title"`
	EncryptedContent string `json:"encrypted_content"`
	// PageAge may be empty when the API does not return age info.
	PageAge string `json:"page_age,omitempty"`
}

// WebSearchResultMetadata stores web search results from Anthropic's
// server-executed web_search tool. The structured data (especially
// EncryptedContent) must be preserved for multi-turn conversations.
type WebSearchResultMetadata struct {
	Results []WebSearchResultItem `json:"results"`
}

// Options implements the ProviderOptions interface.
func (*WebSearchResultMetadata) Options() {}

// MarshalJSON implements custom JSON marshaling with type info for WebSearchResultMetadata.
func (m WebSearchResultMetadata) MarshalJSON() ([]byte, error) {
	type plain WebSearchResultMetadata
	return fantasy.MarshalProviderType(TypeWebSearchResultMetadata, plain(m))
}

// UnmarshalJSON implements custom JSON unmarshaling with type info for WebSearchResultMetadata.
func (m *WebSearchResultMetadata) UnmarshalJSON(data []byte) error {
	type plain WebSearchResultMetadata
	var p plain
	if err := fantasy.UnmarshalProviderType(data, &p); err != nil {
		return err
	}
	*m = WebSearchResultMetadata(p)
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

// UserLocation provides geographic context for web search results.
type UserLocation struct {
	City     string `json:"city,omitempty"`
	Region   string `json:"region,omitempty"`
	Country  string `json:"country,omitempty"`
	Timezone string `json:"timezone,omitempty"`
}

// WebSearchToolOptions configures the Anthropic web search tool.
type WebSearchToolOptions struct {
	// MaxUses limits the number of web searches the model can
	// perform within a single API request. Zero means no limit.
	MaxUses int64
	// AllowedDomains restricts results to these domains. Cannot
	// be used together with BlockedDomains.
	AllowedDomains []string
	// BlockedDomains excludes these domains from results. Cannot
	// be used together with AllowedDomains.
	BlockedDomains []string
	// UserLocation provides geographic context for more relevant
	// search results.
	UserLocation *UserLocation
}

// WebSearchTool creates a provider-defined web search tool for
// Anthropic models. Pass nil for default options.
func WebSearchTool(opts *WebSearchToolOptions) fantasy.ProviderDefinedTool {
	tool := fantasy.ProviderDefinedTool{
		ID:   "web_search",
		Name: "web_search",
	}
	if opts == nil {
		return tool
	}
	args := map[string]any{}
	if opts.MaxUses > 0 {
		args["max_uses"] = opts.MaxUses
	}
	if len(opts.AllowedDomains) > 0 {
		args["allowed_domains"] = opts.AllowedDomains
	}
	if len(opts.BlockedDomains) > 0 {
		args["blocked_domains"] = opts.BlockedDomains
	}
	if opts.UserLocation != nil {
		args["user_location"] = opts.UserLocation
	}
	if len(args) > 0 {
		tool.Args = args
	}
	return tool
}
