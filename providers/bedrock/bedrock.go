// Package bedrock provides an implementation of the fantasy AI SDK for AWS Bedrock's language models.
package bedrock

import (
	"cmp"
	"context"
	"regexp"
	"strings"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/anthropic"
	"charm.land/fantasy/providers/openai"
	"github.com/anthropics/anthropic-sdk-go/option"
	openaibedrock "github.com/charmbracelet/openai-go/bedrock"
	openaioption "github.com/charmbracelet/openai-go/option"
)

type options struct {
	apiKey    string
	baseURL   string
	region    string
	headers   map[string]string
	userAgent string
	client    option.HTTPClient
	skipAuth  bool
}

const (
	// Name is the name of the Bedrock provider.
	Name = "bedrock"

	defaultRegion = "us-east-1"
)

// openaiModelPattern matches the OpenAI models Bedrock hosts, with or without
// a cross-Region inference profile prefix (us., global., ...).
var openaiModelPattern = regexp.MustCompile(`(?:^|\.)openai\.`)

// Option defines a function that configures Bedrock provider options.
type Option = func(*options)

type provider struct {
	options   options
	anthropic fantasy.Provider
}

// New creates a new Bedrock provider. Anthropic models are served through the
// Anthropic SDK and OpenAI models through Bedrock's OpenAI-compatible endpoint.
func New(opts ...Option) (fantasy.Provider, error) {
	var o options
	for _, opt := range opts {
		opt(&o)
	}

	anthropicOpts := []anthropic.Option{
		anthropic.WithName(Name),
		anthropic.WithBedrock(),
		anthropic.WithSkipAuth(o.skipAuth),
		anthropic.WithBedrockRegion(o.region),
	}
	if o.apiKey != "" {
		anthropicOpts = append(anthropicOpts, anthropic.WithAPIKey(o.apiKey))
	}
	if o.baseURL != "" {
		anthropicOpts = append(anthropicOpts, anthropic.WithBaseURL(o.baseURL))
	}
	if o.headers != nil {
		anthropicOpts = append(anthropicOpts, anthropic.WithHeaders(o.headers))
	}
	if o.userAgent != "" {
		anthropicOpts = append(anthropicOpts, anthropic.WithUserAgent(o.userAgent))
	}
	if o.client != nil {
		anthropicOpts = append(anthropicOpts, anthropic.WithHTTPClient(o.client))
	}

	anthropicProvider, err := anthropic.New(anthropicOpts...)
	if err != nil {
		return nil, err
	}
	return &provider{options: o, anthropic: anthropicProvider}, nil
}

func (p *provider) Name() string {
	return Name
}

func (p *provider) LanguageModel(ctx context.Context, modelID string) (fantasy.LanguageModel, error) {
	if !openaiModelPattern.MatchString(modelID) {
		return p.anthropic.LanguageModel(ctx, modelID)
	}
	openaiProvider, err := p.openaiProvider(ctx)
	if err != nil {
		return nil, err
	}
	return openaiProvider.LanguageModel(ctx, modelID)
}

// openaiProvider serves GPT models through the Responses API and gpt-oss
// through Chat Completions, the only OpenAI API Bedrock offers it on.
func (p *provider) openaiProvider(ctx context.Context) (fantasy.Provider, error) {
	cfg := openaibedrock.Config{
		Endpoint:  openaibedrock.EndpointRuntime,
		APIKey:    p.options.apiKey,
		AWSRegion: p.options.region,
		SkipAuth:  p.options.skipAuth && p.options.apiKey == "",
	}
	if cfg.APIKey != "" {
		cfg.AWSRegion = cmp.Or(cfg.AWSRegion, defaultRegion)
	}
	if p.options.baseURL != "" {
		cfg.BaseURL = strings.TrimSuffix(p.options.baseURL, "/") + "/openai/v1"
	}

	var transport []openaioption.RequestOption
	if p.options.client != nil {
		transport = append(transport, openaioption.WithHTTPClient(p.options.client))
	}
	client, err := openaibedrock.NewClient(ctx, cfg, transport...)
	if err != nil {
		return nil, err
	}

	opts := []openai.Option{
		openai.WithName(Name),
		openai.WithSDKOptions(client.Options...),
		openai.WithUseResponsesAPI(),
		openai.WithResponsesAPIFunc(func(modelID string) bool {
			return !strings.Contains(modelID, "gpt-oss")
		}),
	}
	if p.options.headers != nil {
		opts = append(opts, openai.WithHeaders(p.options.headers))
	}
	if p.options.userAgent != "" {
		opts = append(opts, openai.WithUserAgent(p.options.userAgent))
	}
	return openai.New(opts...)
}

// WithAPIKey sets the Bedrock API key. Without one, AWS_BEARER_TOKEN_BEDROCK or
// the AWS default credential chain is used.
func WithAPIKey(apiKey string) Option {
	return func(o *options) {
		o.apiKey = apiKey
	}
}

// WithHeaders sets the headers for the Bedrock provider.
func WithHeaders(headers map[string]string) Option {
	return func(o *options) {
		o.headers = headers
	}
}

// WithHTTPClient sets the HTTP client for the Bedrock provider.
func WithHTTPClient(client option.HTTPClient) Option {
	return func(o *options) {
		o.client = client
	}
}

// WithUserAgent sets an explicit User-Agent header, overriding the default and any
// value set via WithHeaders.
func WithUserAgent(ua string) Option {
	return func(o *options) {
		o.userAgent = ua
	}
}

// WithBaseURL sets the bedrock-runtime base URL, such as a VPC endpoint.
func WithBaseURL(baseURL string) Option {
	return func(o *options) {
		o.baseURL = baseURL
	}
}

// WithSkipAuth configures whether to skip authentication for the Bedrock provider.
func WithSkipAuth(skipAuth bool) Option {
	return func(o *options) {
		o.skipAuth = skipAuth
	}
}

// WithRegion sets the AWS region for the Bedrock provider.
func WithRegion(region string) Option {
	return func(o *options) {
		o.region = region
	}
}
