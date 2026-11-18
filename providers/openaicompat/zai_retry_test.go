package openaicompat

import (
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
	"testing/synctest"
	"time"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

type zaiTestTransport func(*http.Request) (*http.Response, error)

func (fn zaiTestTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return fn(req)
}

const zaiCodingURL = "https://api.z.ai/api/coding/paas/v4"

func TestZAIAccountErrorsStopAfterOneRequest(t *testing.T) {
	for _, tt := range []struct{ code, hint string }{
		{"1113", ""},
		{"1309", "2"},
		{"1308", ""},
		{"1310", "3600"},
	} {
		t.Run(tt.code, func(t *testing.T) {
			testZAI429(t, zaiCodingURL, tt.code, tt.hint, 1, 0)
		})
	}
}

func TestZAITransientThrottleStillRetries(t *testing.T) {
	testZAI429(t, zaiCodingURL, "1302", "2", 4, 2*time.Second)
}

func TestOtherEndpointsKeepRetryingTheSameCode(t *testing.T) {
	testZAI429(t, "https://api.z.ai.example/v1", "1308", "2", 4, 2*time.Second)
}

func TestZAIQuotaWithShortRetryHintStillRetries(t *testing.T) {
	testZAI429(t, zaiCodingURL, "1308", "2", 4, 2*time.Second)
}

func testZAI429(t *testing.T, baseURL, code, hint string, wantAttempts int, wantDelay time.Duration) {
	t.Helper()
	synctest.Test(t, func(t *testing.T) {
		const message = "Provider detail, including any reset time"
		var attempts []time.Time
		client := &http.Client{
			Transport: zaiTestTransport(func(req *http.Request) (*http.Response, error) {
				attempts = append(attempts, time.Now())
				header := http.Header{"Content-Type": {"application/json"}}
				if hint != "" {
					header.Set("Retry-After", hint)
				}
				body := fmt.Sprintf(`{"error":{"code":%q,"message":%q}}`, code, message)
				return &http.Response{
					StatusCode: http.StatusTooManyRequests,
					Header:     header,
					Body:       io.NopCloser(strings.NewReader(body)),
					Request:    req,
				}, nil
			}),
		}
		provider, err := New(WithBaseURL(baseURL), WithAPIKey("fixture"), WithHTTPClient(client))
		require.NoError(t, err)
		model, err := provider.LanguageModel(t.Context(), "glm-fixture")
		require.NoError(t, err)
		_, err = fantasy.NewAgent(model).Stream(t.Context(), fantasy.AgentStreamCall{Prompt: "hello"})

		var providerErr *fantasy.ProviderError
		require.ErrorAs(t, err, &providerErr)
		require.Equal(t, http.StatusTooManyRequests, providerErr.StatusCode)
		require.Equal(t, code, providerErr.ErrorType)
		require.Equal(t, message, providerErr.Message)
		require.Len(t, attempts, wantAttempts)
		if wantAttempts == 1 {
			require.Same(t, providerErr, err)
		}
		for i := 1; i < len(attempts); i++ {
			require.Equal(t, wantDelay, attempts[i].Sub(attempts[i-1]))
		}
	})
}
