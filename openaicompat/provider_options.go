package openaicompat

import (
	"github.com/charmbracelet/fantasy/ai"
	"github.com/charmbracelet/fantasy/openai"
)

type ProviderOptions struct {
	User            *string                 `json:"user"`
	ReasoningEffort *openai.ReasoningEffort `json:"reasoning_effort"`
}

func (*ProviderOptions) Options() {}

func NewProviderOptions(opts *ProviderOptions) ai.ProviderOptions {
	return ai.ProviderOptions{
		Name: opts,
	}
}
