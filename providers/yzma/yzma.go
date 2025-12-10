package yzma

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"charm.land/fantasy"
	"github.com/hybridgroup/yzma/pkg/download"
	"github.com/hybridgroup/yzma/pkg/llama"
)

const (
	// Name is the name of the yzma provider.
	Name = "yzma"
)

type options struct {
	libPath    string
	modelsPath string
}

// Option defines a function that configures yzma provider options.
type Option = func(*options)

type yzmaProvider struct {
	options options
}

// New creates a new yzma provider with the given options.
func New(opts ...Option) (fantasy.Provider, error) {
	providerOptions := options{
		libPath:    os.Getenv("YZMA_LIB"),
		modelsPath: os.Getenv("YZMA_MODELS"),
	}
	for _, o := range opts {
		o(&providerOptions)
	}

	if err := ensureInstalled(providerOptions.libPath); err != nil {
		return nil, err
	}

	if err := llama.Load(providerOptions.libPath); err != nil {
		return nil, err
	}

	llama.LogSet(llama.LogSilent())
	llama.Init()

	return &yzmaProvider{
		options: providerOptions,
	}, nil
}

// WithLibraryPath sets the path to the yzma library files.
func WithLibraryPath(libPath string) Option {
	return func(o *options) {
		o.libPath = libPath
	}
}

// WithModelsPath sets the path to the yzma model files.
func WithModelsPath(modelsPath string) Option {
	return func(o *options) {
		o.modelsPath = modelsPath
	}
}

// Close closes the yzma provider and frees resources.
func (p *yzmaProvider) Close() {
	llama.BackendFree()
}

// Name returns the name of the yzma provider.
func (p *yzmaProvider) Name() string {
	return Name
}

// LanguageModel implements fantasy.Provider.
func (p *yzmaProvider) LanguageModel(ctx context.Context, modelID string) (fantasy.LanguageModel, error) {
	model, err := newModel(modelID, p.options.modelsPath)
	if err != nil {
		return nil, err
	}
	return model, nil
}

// ensureInstalled checks if the required yzma library is installed at the given path,
// and downloads it if not present.
func ensureInstalled(libPath string) error {
	if libPath == "" {
		return fmt.Errorf("yzma library path not specified")
	}

	libfile := filepath.Join(libPath, download.LibraryName(runtime.GOOS))

	if _, err := os.Stat(libfile); !os.IsNotExist(err) {
		return nil
	}

	version, err := download.LlamaLatestVersion()
	if err != nil {
		fmt.Println("could not obtain latest version:", err.Error())
		return err
	}

	if err := download.Get(runtime.GOARCH, runtime.GOOS, "cpu", version, libPath); err != nil {
		fmt.Println("failed to download llama.cpp:", err.Error())
		return err
	}

	return nil
}
