package cerebras

import (
	"charm.land/fantasy"
	"charm.land/fantasy/providers/openaicompat"
)

const (
	Name = "cerebras"
	BaseURL = "https://api.cerebras.ai/v1"
)

type Option = openaicompat.Option

func New(opts ...Option) (fantasy.Provider, error) {
	options := []Option{
		openaicompat.WithName(Name),
		openaicompat.WithBaseURL(BaseURL),
	}
	options = append(options, opts...)
	return openaicompat.New(options...)
}
