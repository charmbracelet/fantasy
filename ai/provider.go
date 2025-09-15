package ai

type Provider interface {
	Name() string
	LanguageModel(modelID string) (LanguageModel, error)
	ParseOptions(data map[string]any) (ProviderOptionsData, error)
}
