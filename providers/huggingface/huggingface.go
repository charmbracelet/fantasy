package huggingface

import (
	"charm.land/fantasy"
	"charm.land/fantasy/providers/openaicompat"
)

const (
	Name = "huggingface"
	BaseURL = "https://router.huggingface.co/v1"
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
