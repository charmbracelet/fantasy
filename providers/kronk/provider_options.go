package kronk

import (
	"encoding/json"

	"charm.land/fantasy"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
)

// Global type identifiers for Kronk-specific provider data.
const (
	TypeProviderOptions  = Name + ".options"
	TypeProviderMetadata = Name + ".metadata"
)

// Register Kronk provider-specific types with the global registry.
func init() {
	fantasy.RegisterProviderType(TypeProviderOptions, func(data []byte) (fantasy.ProviderOptionsData, error) {
		var v ProviderOptions
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
}

// ProviderMetadata represents additional metadata from Kronk provider.
type ProviderMetadata struct {
	Model               string  `json:"model"`
	ModelDescription    string  `json:"model_description,omitempty"`
	ModelType           string  `json:"model_type,omitempty"`
	SystemFingerprint   string  `json:"system_fingerprint,omitempty"`
	HasProjection       bool    `json:"has_projection"`
	ModelSizeBytes      uint64  `json:"model_size_bytes,omitempty"`
	VRAMBytes           int64   `json:"vram_bytes,omitempty"`
	SlotMemoryBytes     int64   `json:"slot_memory_bytes,omitempty"`
	CachedInputTokens   int64   `json:"cached_input_tokens,omitempty"`
	OutputTokens        int64   `json:"output_tokens"`
	TokensPerSecond     float64 `json:"tokens_per_second"`
	TimeToFirstTokenMS  float64 `json:"time_to_first_token_ms"`
	DraftTokens         int64   `json:"draft_tokens,omitempty"`
	DraftAcceptedTokens int64   `json:"draft_accepted_tokens,omitempty"`
	DraftAcceptanceRate float64 `json:"draft_acceptance_rate,omitempty"`
	DraftCoverage       float64 `json:"draft_coverage,omitempty"`
	DraftDisableReason  string  `json:"draft_disable_reason,omitempty"`
}

func newProviderMetadata(info model.ModelInfo) *ProviderMetadata {
	return &ProviderMetadata{
		Model:            info.ID,
		ModelDescription: info.Desc,
		ModelType:        info.Type.String(),
		HasProjection:    info.HasProjection,
		ModelSizeBytes:   info.Size,
		VRAMBytes:        info.VRAMTotal,
		SlotMemoryBytes:  info.SlotMemory,
	}
}

func (m *ProviderMetadata) update(response model.ChatResponse) {
	if response.Model != "" {
		m.Model = response.Model
	}
	if response.SystemFingerprint != "" {
		m.SystemFingerprint = response.SystemFingerprint
	}
	if response.Usage == nil {
		return
	}

	m.CachedInputTokens = int64(response.Usage.PromptTokensDetails.CachedTokens)
	m.OutputTokens = int64(response.Usage.CompletionTokens)
	m.TokensPerSecond = response.Usage.TokensPerSecond
	m.TimeToFirstTokenMS = response.Usage.TimeToFirstTokenMS
	m.DraftTokens = int64(response.Usage.DraftTokens)
	m.DraftAcceptedTokens = int64(response.Usage.DraftAcceptedTokens)
	m.DraftAcceptanceRate = response.Usage.DraftAcceptanceRate
	m.DraftCoverage = response.Usage.DraftCoverage
	m.DraftDisableReason = response.Usage.DraftDisableReason
}

// Options implements the ProviderOptionsData interface.
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

// ProviderOptions represents additional options for Kronk provider.
type ProviderOptions struct {
	AdaptivePDecay  *float64 `json:"adaptive_p_decay"`
	AdaptivePTarget *float64 `json:"adaptive_p_target"`
	DryAllowedLen   *int64   `json:"dry_allowed_length"`
	DryBase         *float64 `json:"dry_base"`
	DryMultiplier   *float64 `json:"dry_multiplier"`
	DryPenaltyLast  *int64   `json:"dry_penalty_last_n"`
	Grammar         *string  `json:"grammar"`
	Logprobs        *bool    `json:"logprobs"`
	MinP            *float64 `json:"min_p"`
	NumPredict      *int64   `json:"num_predict"`
	ReasoningEffort *string  `json:"reasoning_effort"`
	RepeatLastN     *int64   `json:"repeat_last_n"`
	RepeatPenalty   *float64 `json:"repeat_penalty"`
	Seed            *int64   `json:"seed"`
	Stop            []string `json:"stop"`
	Thinking        *bool    `json:"enable_thinking"`
	TopK            *int64   `json:"top_k"`
	TopLogprobs     *int64   `json:"top_logprobs"`
	XtcMinKeep      *int64   `json:"xtc_min_keep"`
	XtcProbability  *float64 `json:"xtc_probability"`
	XtcThreshold    *float64 `json:"xtc_threshold"`
}

// Options implements the ProviderOptionsData interface.
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

// NewProviderOptions creates new provider options for Kronk.
func NewProviderOptions(opts *ProviderOptions) fantasy.ProviderOptions {
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
