package ai

type Provider interface {
	Name() string
	LanguageModel(modelID string) (LanguageModel, error)
	// TODO: add other model types when needed

	OptionsFromMap(data map[string]any) (ProviderOptionsData, error)
}
