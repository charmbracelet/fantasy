package yzma

import (
	"context"
	"errors"
	"os"

	"charm.land/fantasy"
	"github.com/hybridgroup/yzma/pkg/llama"
)

const (
	// Name is the name of the yzma provider.
	Name = "yzma"
)

type options struct {
}

// Option defines a function that configures yzma provider options.
type Option = func(*options)

type yzmaProvider struct {
}

// New creates a new yzma provider with the given options.
func New(opts ...Option) (fantasy.Provider, error) {
	libPath := os.Getenv("YZMA_LIB")
	if libPath == "" {
		return nil, errors.New("no path to yzma libs")
	}

	if err := llama.Load(libPath); err != nil {
		return nil, err
	}

	llama.LogSet(llama.LogSilent())
	llama.Init()

	return &yzmaProvider{}, nil
}

func (p *yzmaProvider) Close() {
	llama.BackendFree()
}

func (p *yzmaProvider) Name() string {
	return Name
}

// LanguageModel implements fantasy.Provider.
func (p *yzmaProvider) LanguageModel(ctx context.Context, modelID string) (fantasy.LanguageModel, error) {
	model, err := newModel(modelID)
	if err != nil {
		return nil, err
	}
	return model, nil
}
