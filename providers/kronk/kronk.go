// Package kronk provides an implementation of the fantasy AI SDK for local
// models using the Kronk SDK.
package kronk

import (
	"context"
	"fmt"
	"sync"

	"charm.land/fantasy"
	"github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
	"github.com/ardanlabs/kronk/sdk/tools/libs"
	"github.com/ardanlabs/kronk/sdk/tools/models"
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

// Provider is a Fantasy provider that owns local Kronk model resources.
type Provider interface {
	fantasy.Provider
	Close(context.Context) error
}

var _ Provider = (*provider)(nil)

// New creates a new Kronk provider with the given options.
func New(opts ...Option) (fantasy.Provider, error) {
	return NewProvider(opts...)
}

// NewProvider creates a Kronk provider with direct access to Close.
func NewProvider(opts ...Option) (Provider, error) {
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
		return nil, toOperationErr("kronk installation failed", err)
	}

	krn, err := p.newKronk(mp)
	if err != nil {
		return nil, toOperationErr("kronk model loading failed", err)
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

func (p *provider) installSystem(ctx context.Context, modelSource string) (models.Path, error) {
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

	mdls, err := models.New()
	if err != nil {
		return models.Path{}, fmt.Errorf("unable to create models: %w", err)
	}

	mp, err := mdls.Download(ctx, models.Logger(logger), modelSource)
	if err != nil {
		return models.Path{}, fmt.Errorf("unable to install model: %w", err)
	}

	return mp, nil
}

func (p *provider) newKronk(mp models.Path) (*kronk.Kronk, error) {
	if err := kronk.Init(); err != nil {
		return nil, fmt.Errorf("unable to init kronk: %w", err)
	}

	cfg := p.modelConfig(mp)

	krn, err := kronk.New(model.WithConfig(cfg))
	if err != nil {
		return nil, fmt.Errorf("unable to create inference model: %w", err)
	}

	return krn, nil
}

func (p *provider) modelConfig(mp models.Path) model.Config {
	cfg := model.NewConfig(p.options.modelOptions...)
	cfg.ModelFiles = mp.ModelFiles
	if cfg.ProjFile == "" {
		cfg.ProjFile = mp.ProjFile
	}
	if cfg.MTPDrafterFile == "" {
		cfg.MTPDrafterFile = mp.MTPFile
	}

	return cfg
}
