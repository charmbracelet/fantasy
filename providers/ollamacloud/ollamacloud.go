// Package ollamacloud provides an implementation of the fantasy AI SDK for Ollama Cloud's language models.
package ollamacloud

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"

	"charm.land/fantasy"
)

const (
	// Name is the name of the Ollama Cloud provider.
	Name = "ollama-cloud"
	// DefaultURL is the default URL for the Ollama Cloud API.
	DefaultURL = "https://ollama.com"
)

type options struct {
	baseURL    string
	apiKey     string
	name       string
	headers    map[string]string
	httpClient *http.Client
}

type provider struct {
	options options
}

// Option defines a function that configures Ollama Cloud provider options.
type Option = func(*options)

// New creates a new Ollama Cloud provider with the given options.
func New(opts ...Option) (fantasy.Provider, error) {
	providerOptions := options{
		headers:    map[string]string{},
		httpClient: &http.Client{},
	}
	for _, o := range opts {
		o(&providerOptions)
	}

	providerOptions.baseURL = cmp.Or(providerOptions.baseURL, DefaultURL)
	providerOptions.name = cmp.Or(providerOptions.name, Name)

	return &provider{options: providerOptions}, nil
}

// WithBaseURL sets the base URL for the Ollama Cloud provider.
func WithBaseURL(baseURL string) Option {
	return func(o *options) {
		o.baseURL = baseURL
	}
}

// WithAPIKey sets the API key for the Ollama Cloud provider.
func WithAPIKey(apiKey string) Option {
	return func(o *options) {
		o.apiKey = apiKey
	}
}

// WithName sets the name for the Ollama Cloud provider.
func WithName(name string) Option {
	return func(o *options) {
		o.name = name
	}
}

// WithHeaders sets the headers for the Ollama Cloud provider.
func WithHeaders(headers map[string]string) Option {
	return func(o *options) {
		maps.Copy(o.headers, headers)
	}
}

// WithHTTPClient sets the HTTP client for the Ollama Cloud provider.
func WithHTTPClient(client *http.Client) Option {
	return func(o *options) {
		o.httpClient = client
	}
}

func (p *provider) LanguageModel(ctx context.Context, modelID string) (fantasy.LanguageModel, error) {
	return &languageModel{
		provider: p,
		modelID:  modelID,
	}, nil
}

func (p *provider) Name() string {
	return p.options.name
}

// doRequest makes an HTTP request to the Ollama Cloud API.
func (p *provider) doRequest(ctx context.Context, reqBody any) (*http.Response, error) {
	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fantasy.NewInvalidArgumentError("request body", "failed to marshal request", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", p.options.baseURL+"/api/chat", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	if p.options.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.options.apiKey)
	}
	for k, v := range p.options.headers {
		req.Header.Set(k, v)
	}

	resp, err := p.options.httpClient.Do(req)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fantasy.NewAIError(fmt.Sprintf("HTTP %d", resp.StatusCode), fmt.Sprintf("API error: %s", string(body)), nil)
	}

	return resp, nil
}
