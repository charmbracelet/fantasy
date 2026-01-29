// Package kronk provides an implementation of the fantasy AI SDK for local
// models using the Kronk SDK.
package kronk

import (
	"context"
	"fmt"
	"sync"

	"charm.land/fantasy"
	"github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/tools/libs"
	"github.com/ardanlabs/kronk/sdk/tools/models"
	"github.com/ardanlabs/kronk/sdk/tools/templates"
)

const (
	// Name is the name of the Kronk provider.
	Name = "kronk"
)

type provider struct {
	options options
	mu      sync.Mutex
	kronks  map[string]*kronk.Kronk
}

// New creates a new Kronk provider with the given options.
func New(opts ...Option) (fantasy.Provider, error) {
	providerOptions := options{
		languageModelOptions: make([]LanguageModelOption, 0),
	}

	for _, o := range opts {
		o(&providerOptions)
	}

	if providerOptions.name == "" {
		providerOptions.name = Name
	}

	p := provider{
		options: providerOptions,
		kronks:  make(map[string]*kronk.Kronk),
	}

	return &p, nil
}

// Name implements fantasy.Provider.
func (p *provider) Name() string {
	return p.options.name
}

// LanguageModel implements fantasy.Provider.
// The modelURL parameter should be a URL to a GGUF model file (e.g., from Hugging Face).
func (p *provider) LanguageModel(ctx context.Context, modelURL string) (fantasy.LanguageModel, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if krn, ok := p.kronks[modelURL]; ok {
		opts := append(p.options.languageModelOptions, WithLanguageModelObjectMode(p.options.objectMode))
		return newLanguageModel(modelURL, p.options.name, krn, opts...), nil
	}

	mp, err := p.installSystem(ctx, modelURL)
	if err != nil {
		return nil, fmt.Errorf("failed to install system: %w", err)
	}

	krn, err := p.newKronk(mp)
	if err != nil {
		return nil, fmt.Errorf("failed to create kronk instance: %w", err)
	}

	p.kronks[modelURL] = krn

	opts := append(p.options.languageModelOptions, WithLanguageModelObjectMode(p.options.objectMode))

	return newLanguageModel(modelURL, p.options.name, krn, opts...), nil
}

// Close unloads all Kronk instances. Call this when done with the provider.
func (p *provider) Close(ctx context.Context) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	var errs []error

	for url, krn := range p.kronks {
		if err := krn.Unload(ctx); err != nil {
			errs = append(errs, fmt.Errorf("failed to unload model %s: %w", url, err))
		}

		delete(p.kronks, url)
	}

	if len(errs) > 0 {
		return errs[0]
	}

	return nil
}

func (p *provider) installSystem(ctx context.Context, modelURL string) (models.Path, error) {
	logger := p.options.logger
	if logger == nil {
		logger = func(context.Context, string, ...any) {}
	}

	lbs, err := libs.New()
	if err != nil {
		return models.Path{}, fmt.Errorf("unable to create libs: %w", err)
	}

	if _, err := lbs.Download(ctx, libs.Logger(logger)); err != nil {
		return models.Path{}, fmt.Errorf("unable to install llama.cpp: %w", err)
	}

	tmpls, err := templates.New()
	if err != nil {
		return models.Path{}, fmt.Errorf("unable to create template system: %w", err)
	}

	if err := tmpls.Download(ctx); err != nil {
		return models.Path{}, fmt.Errorf("unable to download templates: %w", err)
	}

	if err := tmpls.Catalog().Download(ctx); err != nil {
		return models.Path{}, fmt.Errorf("unable to download catalog: %w", err)
	}

	mdls, err := models.New()
	if err != nil {
		return models.Path{}, fmt.Errorf("unable to create models: %w", err)
	}

	mp, err := mdls.Download(ctx, models.Logger(logger), modelURL, "")
	if err != nil {
		return models.Path{}, fmt.Errorf("unable to install model: %w", err)
	}

	return mp, nil
}

func (p *provider) newKronk(mp models.Path) (*kronk.Kronk, error) {
	if err := kronk.Init(); err != nil {
		return nil, fmt.Errorf("unable to init kronk: %w", err)
	}

	cfg := p.options.modelConfig
	cfg.ModelFiles = mp.ModelFiles

	krn, err := kronk.New(cfg)
	if err != nil {
		return nil, fmt.Errorf("unable to create inference model: %w", err)
	}

	return krn, nil
}
