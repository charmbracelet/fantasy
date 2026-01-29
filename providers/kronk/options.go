package kronk

import (
	"context"
	"fmt"

	"charm.land/fantasy"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
)

// Option defines a function that configures Kronk provider options.
type Option func(*options)

// Logger is the function signature for logging download progress.
type Logger func(ctx context.Context, msg string, args ...any)

type options struct {
	name                 string
	modelConfig          model.Config
	logger               Logger
	objectMode           fantasy.ObjectMode
	languageModelOptions []LanguageModelOption
}

// WithName sets the name for the Kronk provider.
func WithName(name string) Option {
	return func(o *options) {
		o.name = name
	}
}

// WithModelConfig sets additional model configuration options.
func WithModelConfig(cfg model.Config) Option {
	return func(o *options) {
		o.modelConfig = cfg
	}
}

// WithLogger sets the logger function for download progress.
func WithLogger(logger Logger) Option {
	return func(o *options) {
		o.logger = logger
	}
}

// WithLanguageModelOptions sets the language model options for the Kronk provider.
func WithLanguageModelOptions(opts ...LanguageModelOption) Option {
	return func(o *options) {
		o.languageModelOptions = append(o.languageModelOptions, opts...)
	}
}

// WithObjectMode sets the object generation mode.
func WithObjectMode(om fantasy.ObjectMode) Option {
	return func(o *options) {
		o.objectMode = om
	}
}

// FmtLogger is a simple logger that prints to stdout using fmt.Printf.
func FmtLogger(_ context.Context, msg string, args ...any) {
	fmt.Printf("%s:", msg)

	for i := 0; i < len(args); i += 2 {
		if i+1 < len(args) {
			fmt.Printf(" %v[%v]", args[i], args[i+1])
		}
	}

	fmt.Println()
}
