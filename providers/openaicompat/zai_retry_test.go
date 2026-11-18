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
		{"1311", "2"},
		{"1313", "2"},
		{"1314", "2"},
		{"1315", "2"},
		{"1316", ""},
		{"1317", ""},
		{"1318", ""},
		{"1319", ""},
		{"1320", ""},
		{"1321", ""},
	} {
		t.Run(tt.code, func(t *testing.T) {
			testZAI429(t, zaiCodingURL, tt.code, retryAfter(tt.hint), 1, 0)
		})
	}
}

func TestZAITransientThrottleStillRetries(t *testing.T) {
	testZAI429(t, zaiCodingURL, "1302", retryAfter("2"), 4, 2*time.Second)
}

func TestOtherEndpointsKeepRetryingTheSameCode(t *testing.T) {
	testZAI429(t, "https://api.z.ai.example/v1", "1309", retryAfter("2"), 4, 2*time.Second)
}

func TestZAIQuotaWithShortRetryHintStillRetries(t *testing.T) {
	for _, tt := range []struct {
		name    string
		headers func() http.Header
		delay   time.Duration
	}{
		{"seconds", retryAfter("2"), 2 * time.Second},
		{"milliseconds take precedence", func() http.Header {
			return http.Header{"Retry-After": {"9"}, "Retry-After-Ms": {"1250"}}
		}, 1250 * time.Millisecond},
		{"date", func() http.Header {
			return http.Header{"Retry-After": {time.Now().Add(2 * time.Second).UTC().Format(http.TimeFormat)}}
		}, 2 * time.Second},
		{"below cutoff", retryAfter("59"), 59 * time.Second},
	} {
		t.Run(tt.name, func(t *testing.T) {
			testZAI429(t, zaiCodingURL, "1308", tt.headers, 4, tt.delay)
		})
	}
}

func TestZAIRecognizesMixedCaseHost(t *testing.T) {
	testZAI429(t, "https://API.Z.AI/api/coding/paas/v4", "1309", retryAfter("2"), 1, 0)
}

func TestZAIRecognizesBigModelHost(t *testing.T) {
	testZAI429(t, "https://open.bigmodel.cn/api/paas/v4", "1309", retryAfter("2"), 1, 0)
}

func TestZAIQuotaRejectsHintAtCutoff(t *testing.T) {
	testZAI429(t, zaiCodingURL, "1308", retryAfter("60"), 1, 0)
}

func retryAfter(value string) func() http.Header {
	return func() http.Header {
		headers := make(http.Header)
		if value != "" {
			headers.Set("Retry-After", value)
		}
		return headers
	}
}

func testZAI429(t *testing.T, baseURL, code string, headers func() http.Header, wantAttempts int, wantDelay time.Duration) {
	t.Helper()
	synctest.Test(t, func(t *testing.T) {
		const message = "Provider detail, including any reset time"
		var attempts []time.Time
		client := &http.Client{
			Transport: zaiTestTransport(func(req *http.Request) (*http.Response, error) {
				attempts = append(attempts, time.Now())
				header := headers()
				header.Set("Content-Type", "application/json")
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
