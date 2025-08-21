package ai

type Provider interface {
	LanguageModel(modelID string) LanguageModel
	// TODO: add other model types when needed
}
