// Package synthetic provides an implementation of the fantasy AI SDK for
// Synthetic (https://synthetic.new), wrapping its OpenAI-compatible API.
//
// Its distinguishing feature is quota-aware rate limiting. When the
// subscription quota is exhausted, Synthetic rejects requests without naming
// a retry moment, and the raw 429 reads as an opaque "too many requests".
// This provider then consults the /v2/quotas endpoint — which Synthetic does
// not count against the subscription — and records the renewal moment on the
// error, so the retry middleware waits for it when it fits its budget, and
// callers that give up still learn when a retry can succeed.
package synthetic

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openaicompat"
)

const (
	// DefaultURL is the default URL for the Synthetic OpenAI-compatible API.
	DefaultURL = "https://api.synthetic.new/openai/v1"

	// Name is the name of the Synthetic provider.
	Name = "synthetic"

	// quotasPath is the quota endpoint, served from the API origin rather
	// than the OpenAI-compatible base URL.
	quotasPath = "/v2/quotas"

	// probeTimeout bounds the quota lookup that enriches a rate-limit
	// error; it must never stall a failing request longer than the failure
	// itself already has.
	probeTimeout = 5 * time.Second
)

// Option defines a function that configures the Synthetic provider.
type Option = func(*options)

type options struct {
	openaicompatOptions []openaicompat.Option
	apiKey              string
	baseURL             string
	quotaURL            string
	httpClient          *http.Client
}

// New creates a new Synthetic provider.
func New(opts ...Option) (fantasy.Provider, error) {
	providerOptions := options{
		baseURL: DefaultURL,
	}
	for _, o := range opts {
		o(&providerOptions)
	}

	innerOptions := append([]openaicompat.Option{
		openaicompat.WithName(Name),
		openaicompat.WithBaseURL(providerOptions.baseURL),
	}, providerOptions.openaicompatOptions...)
	if providerOptions.apiKey != "" {
		innerOptions = append(innerOptions, openaicompat.WithAPIKey(providerOptions.apiKey))
	}
	if providerOptions.httpClient != nil {
		innerOptions = append(innerOptions, openaicompat.WithHTTPClient(providerOptions.httpClient))
	}

	inner, err := openaicompat.New(innerOptions...)
	if err != nil {
		return nil, err
	}

	return &provider{
		inner:  inner,
		prober: newQuotaProber(providerOptions.quotaURL, providerOptions.baseURL, providerOptions.apiKey, providerOptions.httpClient),
	}, nil
}

// WithAPIKey sets the API key for both the model API and the quota endpoint.
func WithAPIKey(apiKey string) Option {
	return func(o *options) {
		o.apiKey = apiKey
	}
}

// WithBaseURL sets the base URL for the Synthetic API. The quota endpoint is
// derived from its origin unless WithQuotaURL overrides it.
func WithBaseURL(rawURL string) Option {
	return func(o *options) {
		o.baseURL = rawURL
	}
}

// WithQuotaURL sets the quota endpoint outright, for proxies that do not
// preserve the Synthetic origin. An empty string restores the derived
// default.
func WithQuotaURL(rawURL string) Option {
	return func(o *options) {
		o.quotaURL = rawURL
	}
}

// WithHTTPClient sets the HTTP client used for both model calls and the
// quota endpoint.
func WithHTTPClient(client *http.Client) Option {
	return func(o *options) {
		o.httpClient = client
	}
}

// WithOpenAICompatOptions passes options through to the underlying
// OpenAI-compatible provider, for anything this package does not surface
// itself.
func WithOpenAICompatOptions(opts ...openaicompat.Option) Option {
	return func(o *options) {
		o.openaicompatOptions = append(o.openaicompatOptions, opts...)
	}
}

// provider wraps the OpenAI-compatible provider so every language model it
// hands out enriches quota errors with the renewal deadline.
type provider struct {
	inner  fantasy.Provider
	prober *quotaProber
}

func (p *provider) Name() string {
	return Name
}

func (p *provider) LanguageModel(ctx context.Context, modelID string) (fantasy.LanguageModel, error) {
	inner, err := p.inner.LanguageModel(ctx, modelID)
	if err != nil {
		return nil, err
	}
	return &languageModel{inner: inner, prober: p.prober}, nil
}

// languageModel delegates every call to the OpenAI-compatible model and
// fills in the quota renewal deadline on errors that report an exhausted
// subscription.
type languageModel struct {
	inner  fantasy.LanguageModel
	prober *quotaProber
}

func (m *languageModel) Generate(ctx context.Context, call fantasy.Call) (*fantasy.Response, error) {
	response, err := m.inner.Generate(ctx, call)
	return response, m.enrichQuotaError(ctx, err)
}

func (m *languageModel) Stream(ctx context.Context, call fantasy.Call) (fantasy.StreamResponse, error) {
	response, err := m.inner.Stream(ctx, call)
	return response, m.enrichQuotaError(ctx, err)
}

func (m *languageModel) GenerateObject(ctx context.Context, call fantasy.ObjectCall) (*fantasy.ObjectResponse, error) {
	response, err := m.inner.GenerateObject(ctx, call)
	return response, m.enrichQuotaError(ctx, err)
}

func (m *languageModel) StreamObject(ctx context.Context, call fantasy.ObjectCall) (fantasy.ObjectStreamResponse, error) {
	response, err := m.inner.StreamObject(ctx, call)
	return response, m.enrichQuotaError(ctx, err)
}

func (m *languageModel) Provider() string {
	return m.inner.Provider()
}

func (m *languageModel) Model() string {
	return m.inner.Model()
}

// enrichQuotaError consults /v2/quotas when a request failed on an exhausted
// subscription quota and no retry deadline is known, recording the renewal
// moment and usage on the error. It is best-effort: a failed lookup leaves
// the error untouched, and an error that already carries a deadline (or no
// quota exhaustion at all) passes through unchanged.
func (m *languageModel) enrichQuotaError(ctx context.Context, err error) error {
	if err == nil {
		return nil
	}

	providerErr, ok := errors.AsType[*fantasy.ProviderError](err)
	if !ok || !providerErr.QuotaExhausted || !providerErr.RetryAt.IsZero() {
		return err
	}

	quota, probeErr := m.prober.subscription(ctx)
	if probeErr != nil || quota.RenewsAt.IsZero() {
		return err
	}

	providerErr.RetryAt = quota.RenewsAt
	providerErr.Message = fmt.Sprintf(
		"You've exceeded your subscription rate limits (%d of %d requests used). "+
			"Upgrade, or view your usage at https://synthetic.new/billing",
		quota.Requests, quota.Limit,
	)

	return err
}

// subscriptionQuota is the subscription entry of the /v2/quotas response.
type subscriptionQuota struct {
	Limit    int
	Requests int
	RenewsAt time.Time
}

// quotaProber reads the Synthetic quota endpoint.
type quotaProber struct {
	url    string
	apiKey string
	client *http.Client
}

func newQuotaProber(quotaURL, baseURL, apiKey string, client *http.Client) *quotaProber {
	if client == nil {
		client = http.DefaultClient
	}
	return &quotaProber{
		url:    cmp.Or(quotaURL, quotaURLForBaseURL(baseURL)),
		apiKey: apiKey,
		client: client,
	}
}

// quotaURLForBaseURL derives the quota endpoint from the API origin, e.g.
// https://api.synthetic.new/openai/v1 -> https://api.synthetic.new/v2/quotas.
// An unparseable base URL yields no endpoint, which disables probing.
func quotaURLForBaseURL(baseURL string) string {
	parsed, err := url.Parse(strings.TrimSpace(baseURL))
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return ""
	}
	return parsed.Scheme + "://" + parsed.Host + quotasPath
}

// subscription fetches the subscription quota. It is documented not to count
// against the very limits it reports.
func (p *quotaProber) subscription(ctx context.Context) (subscriptionQuota, error) {
	var quota subscriptionQuota

	if p.url == "" {
		return quota, errors.New("synthetic: no quota endpoint configured")
	}

	probeCtx, cancel := context.WithTimeout(ctx, probeTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(probeCtx, http.MethodGet, p.url, nil)
	if err != nil {
		return quota, fmt.Errorf("synthetic: build quotas request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+p.apiKey)
	req.Header.Set("Accept", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return quota, fmt.Errorf("synthetic: query quotas: %w", err)
	}
	defer resp.Body.Close() //nolint:errcheck // read-side; nothing to do with a close error

	if resp.StatusCode != http.StatusOK {
		return quota, fmt.Errorf("synthetic: quotas endpoint answered %s", resp.Status)
	}

	var payload struct {
		Subscription struct {
			Limit    int       `json:"limit"`
			Requests int       `json:"requests"`
			RenewsAt time.Time `json:"renewsAt"`
		} `json:"subscription"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return quota, fmt.Errorf("synthetic: decode quotas response: %w", err)
	}

	quota = subscriptionQuota{
		Limit:    payload.Subscription.Limit,
		Requests: payload.Subscription.Requests,
		RenewsAt: payload.Subscription.RenewsAt,
	}

	return quota, nil
}
