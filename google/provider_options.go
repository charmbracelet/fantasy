package google

const Name = "google"

type ThinkingConfig struct {
	ThinkingBudget  *int64 `json:"thinking_budget"`
	IncludeThoughts *bool  `json:"include_thoughts"`
}

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

func (o *ProviderOptions) Options() {}
