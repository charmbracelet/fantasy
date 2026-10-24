package synthetic

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"charm.land/fantasy"
)

var testPrompt = fantasy.Prompt{
	{
		Role: fantasy.MessageRoleUser,
		Content: []fantasy.MessagePart{
			fantasy.TextPart{Text: "Hello"},
		},
	},
}

const quotaExhaustedBody = `{"error":"You've exceeded your subscription rate limits. Upgrade, or try again later. You can view your usage at https://synthetic.new/billing"}`

// newServer stands in for the Synthetic API: model calls answer with the
// documented quota-exhausted body, and /v2/quotas reports a renewal moment.
func newServer(t *testing.T, renewsAt time.Time) *httptest.Server {
	t.Helper()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == quotasPath {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"subscription":{"limit":135,"requests":137,"renewsAt":"` + renewsAt.Format(time.RFC3339) + `"}}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(quotaExhaustedBody))
	}))
	t.Cleanup(server.Close)

	return server
}

func TestQuotaExhaustionCarriesRenewalDeadline(t *testing.T) {
	t.Parallel()

	renewsAt := time.Now().Add(3 * time.Hour).UTC().Truncate(time.Second)
	server := newServer(t, renewsAt)

	provider, err := New(WithAPIKey("test-api-key"), WithBaseURL(server.URL+"/openai/v1"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	model, err := provider.LanguageModel(context.Background(), "hf:zai-org/GLM-5.3")
	if err != nil {
		t.Fatalf("LanguageModel: %v", err)
	}

	_, err = model.Generate(context.Background(), fantasy.Call{Prompt: testPrompt})
	if err == nil {
		t.Fatal("the request was meant to be rejected")
	}

	providerErr, ok := errors.AsType[*fantasy.ProviderError](err)
	if !ok {
		t.Fatalf("error is %T, want *fantasy.ProviderError", err)
	}
	if !providerErr.QuotaExhausted {
		t.Error("QuotaExhausted = false, want true")
	}
	if providerErr.RetryAt.IsZero() {
		t.Fatal("RetryAt is zero, want the renewal moment from /v2/quotas")
	}
	if diff := providerErr.RetryAt.Sub(renewsAt); diff < -time.Second || diff > time.Second {
		t.Errorf("RetryAt = %v, want %v", providerErr.RetryAt, renewsAt)
	}
	if !providerErr.IsRetryable() {
		t.Error("quota exhaustion with a deadline must stay retryable so the middleware can time the wait")
	}
	if want := "137 of 135 requests used"; !strings.Contains(providerErr.Message, want) {
		t.Errorf("Message = %q, want it to report %q", providerErr.Message, want)
	}
	if msg := providerErr.Error(); !strings.Contains(msg, "retry at ") {
		t.Errorf("Error() = %q, want it to name the retry deadline", msg)
	}
}

func TestQuotaProbeFailureLeavesErrorUntouched(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == quotasPath {
			http.Error(w, "boom", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(quotaExhaustedBody))
	}))
	defer server.Close()

	provider, err := New(WithAPIKey("test-api-key"), WithBaseURL(server.URL+"/openai/v1"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	model, err := provider.LanguageModel(context.Background(), "hf:zai-org/GLM-5.3")
	if err != nil {
		t.Fatalf("LanguageModel: %v", err)
	}

	_, err = model.Generate(context.Background(), fantasy.Call{Prompt: testPrompt})
	if err == nil {
		t.Fatal("the request was meant to be rejected")
	}

	providerErr, ok := errors.AsType[*fantasy.ProviderError](err)
	if !ok {
		t.Fatalf("error is %T, want *fantasy.ProviderError", err)
	}
	if !providerErr.QuotaExhausted {
		t.Error("QuotaExhausted = false, want true")
	}
	if !providerErr.RetryAt.IsZero() {
		t.Errorf("RetryAt = %v, want zero: a failed probe must not invent a deadline", providerErr.RetryAt)
	}
	if providerErr.IsRetryable() {
		t.Error("quota exhaustion without a deadline must not be retried")
	}
	if providerErr.Message != quotaExhaustedBody {
		t.Errorf("Message = %q, want the original body %q", providerErr.Message, quotaExhaustedBody)
	}
}

func TestQuotaURLForBaseURL(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		baseURL string
		want    string
	}{
		{"default", DefaultURL, "https://api.synthetic.new" + quotasPath},
		{"custom origin", "http://localhost:8080/openai/v1", "http://localhost:8080" + quotasPath},
		{"unparseable", "not a url", ""},
		{"empty", "", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			if got := quotaURLForBaseURL(tc.baseURL); got != tc.want {
				t.Errorf("quotaURLForBaseURL(%q) = %q, want %q", tc.baseURL, got, tc.want)
			}
		})
	}
}

func TestRetryMiddlewareStopsWhenQuotaRenewalExceedsBudget(t *testing.T) {
	t.Parallel()

	renewsAt := time.Now().Add(23 * time.Hour).UTC().Truncate(time.Second)
	server := newServer(t, renewsAt)

	provider, err := New(WithAPIKey("test-api-key"), WithBaseURL(server.URL+"/openai/v1"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	model, err := provider.LanguageModel(context.Background(), "hf:zai-org/GLM-5.3")
	if err != nil {
		t.Fatalf("LanguageModel: %v", err)
	}

	attempts := 0
	retry := fantasy.RetryWithExponentialBackoffRespectingRetryHeaders[*fantasy.Response](fantasy.RetryOptions{
		MaxRetries:     3,
		InitialDelayIn: time.Millisecond,
		BackoffFactor:  2,
		MaxRetryDelay:  time.Minute,
	})

	_, err = retry(context.Background(), func() (*fantasy.Response, error) {
		attempts++
		return model.Generate(context.Background(), fantasy.Call{Prompt: testPrompt})
	})

	if attempts != 1 {
		t.Errorf("attempts = %d, want 1: a renewal beyond the retry budget must not burn retries", attempts)
	}
	if err == nil {
		t.Fatal("the request was meant to be rejected")
	}
	if msg := err.Error(); !strings.Contains(msg, "retry at ") || !strings.Contains(msg, renewsAt.Format(time.RFC3339)) {
		t.Errorf("Error() = %q, want it to name the renewal moment %s", msg, renewsAt.Format(time.RFC3339))
	}
}
