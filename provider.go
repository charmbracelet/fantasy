package ai

type Provider interface {
	LanguageModel(modelID string) (LanguageModel, error)
	// TODO: add other model types when needed
}
