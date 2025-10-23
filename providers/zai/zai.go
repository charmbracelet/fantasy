package zai

import (
	"charm.land/fantasy"
	"charm.land/fantasy/providers/openaicompat"
)

const (
	Name = "zai"
	BaseURL = "https://api.z.ai/api/coding/paas/v4"
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
