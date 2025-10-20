package fantasy

// Provider represents a provider of language models.
type Provider interface {
	Name() string
	LanguageModel(modelID string) (LanguageModel, error)
}
