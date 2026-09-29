package openai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestNormalizeErrorBody(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name       string
		body       string
		statusCode int
		// wantMessage is the expected message after rewriting. When empty,
		// the body must pass through unchanged.
		wantMessage string
	}{
		{
			name:       "conformant error object passes through unchanged",
			body:       `{"error":{"message":"bad key","type":"invalid_request_error"}}`,
			statusCode: http.StatusUnauthorized,
		},
		{
			name:        "string error member",
			body:        `{"error":"rate limit reached"}`,
			statusCode:  http.StatusTooManyRequests,
			wantMessage: "rate limit reached",
		},
		{
			name:        "error object with numeric code",
			body:        `{"error":{"code":429,"message":"slow down"}}`,
			statusCode:  http.StatusTooManyRequests,
			wantMessage: "slow down",
		},
		{
			name:        "top-level message without error member",
			body:        `{"message":"bad key"}`,
			statusCode:  http.StatusUnauthorized,
			wantMessage: "bad key",
		},
		{
			name:        "whole body is a JSON string",
			body:        `"quota exceeded"`,
			statusCode:  http.StatusPaymentRequired,
			wantMessage: "quota exceeded",
		},
		{
			name:        "plain text body",
			body:        `Service unavailable`,
			statusCode:  http.StatusServiceUnavailable,
			wantMessage: "Service unavailable",
		},
		{
			name:        "html error page",
			body:        `<html><body>Bad gateway</body></html>`,
			statusCode:  http.StatusBadGateway,
			wantMessage: `<html><body>Bad gateway</body></html>`,
		},
		{
			name:        "empty body falls back to status text",
			body:        ``,
			statusCode:  http.StatusInternalServerError,
			wantMessage: "Internal Server Error",
		},
		{
			name:        "json array body",
			body:        `["oops"]`,
			statusCode:  http.StatusBadRequest,
			wantMessage: `["oops"]`,
		},
		{
			name:        "null error member",
			body:        `{"error":null}`,
			statusCode:  http.StatusBadRequest,
			wantMessage: `{"error":null}`,
		},
		{
			name:        "empty error message falls through to top-level message",
			body:        `{"error":{"code":429,"message":""},"message":"quota exhausted"}`,
			statusCode:  http.StatusTooManyRequests,
			wantMessage: "quota exhausted",
		},
		{
			name:        "empty string error member falls through to top-level message",
			body:        `{"error":"","message":"quota exhausted"}`,
			statusCode:  http.StatusTooManyRequests,
			wantMessage: "quota exhausted",
		},
		{
			name:        "empty JSON string body falls back to status text",
			body:        `""`,
			statusCode:  http.StatusInternalServerError,
			wantMessage: "Internal Server Error",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			got := normalizeErrorBody([]byte(tc.body), tc.statusCode)

			if tc.wantMessage == "" {
				require.Equal(t, tc.body, string(got), "conformant body must pass through unchanged")
				return
			}

			var rewritten struct {
				Error struct {
					Message string `json:"message"`
				} `json:"error"`
			}
			require.NoError(t, json.Unmarshal(got, &rewritten), "rewritten body must be valid JSON in the OpenAI error shape")
			require.Equal(t, tc.wantMessage, rewritten.Error.Message)
		})
	}
}

func TestNormalizeErrorBodyMiddleware(t *testing.T) {
	t.Parallel()

	respond := func(status int, body string) func(*http.Request) (*http.Response, error) {
		return func(*http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: status,
				Body:       io.NopCloser(strings.NewReader(body)),
				Header:     make(http.Header),
			}, nil
		}
	}

	t.Run("rewrites non-standard error body", func(t *testing.T) {
		t.Parallel()

		res, err := normalizeErrorBodyMiddleware(&http.Request{}, respond(http.StatusTooManyRequests, `{"error":"rate limit reached"}`))
		require.NoError(t, err)

		got, err := io.ReadAll(res.Body)
		require.NoError(t, err)

		var rewritten struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		require.NoError(t, json.Unmarshal(got, &rewritten))
		require.Equal(t, "rate limit reached", rewritten.Error.Message)
	})

	t.Run("leaves success responses untouched", func(t *testing.T) {
		t.Parallel()

		res, err := normalizeErrorBodyMiddleware(&http.Request{}, respond(http.StatusOK, `{"error":"not an error response"}`))
		require.NoError(t, err)

		got, err := io.ReadAll(res.Body)
		require.NoError(t, err)
		require.Equal(t, `{"error":"not an error response"}`, string(got))
	})

	t.Run("passes through transport errors", func(t *testing.T) {
		t.Parallel()

		wantErr := errors.New("connection reset")
		next := func(*http.Request) (*http.Response, error) { return nil, wantErr }

		res, err := normalizeErrorBodyMiddleware(&http.Request{}, next)
		require.ErrorIs(t, err, wantErr)
		require.Nil(t, res)
	})

	t.Run("updates ContentLength when rewriting the body", func(t *testing.T) {
		t.Parallel()

		original := `{"error":"rate limit reached"}`
		next := func(*http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode:    http.StatusTooManyRequests,
				Body:          io.NopCloser(strings.NewReader(original)),
				ContentLength: int64(len(original)),
				Header:        http.Header{"Content-Length": []string{strconv.Itoa(len(original))}},
			}, nil
		}

		res, err := normalizeErrorBodyMiddleware(&http.Request{}, next)
		require.NoError(t, err)

		body, err := io.ReadAll(res.Body)
		require.NoError(t, err)
		require.Equal(t, int64(len(body)), res.ContentLength, "stale ContentLength breaks httputil.DumpResponse")
		require.Equal(t, strconv.Itoa(len(body)), res.Header.Get("Content-Length"))
	})

	t.Run("initializes nil response headers", func(t *testing.T) {
		t.Parallel()

		// A custom transport supplied via WithHTTPClient may return a
		// response with a nil header map.
		body := `{"error":{"message":"bad key","type":"invalid_request_error"}}`
		next := func(*http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusUnauthorized,
				Body:       io.NopCloser(strings.NewReader(body)),
			}, nil
		}

		res, err := normalizeErrorBodyMiddleware(&http.Request{}, next)
		require.NoError(t, err)
		require.NotNil(t, res.Header)
		require.Equal(t, strconv.Itoa(len(body)), res.Header.Get("Content-Length"))

		got, err := io.ReadAll(res.Body)
		require.NoError(t, err)
		require.Equal(t, body, string(got), "conformant body must pass through unchanged")
	})

	t.Run("propagates body read errors", func(t *testing.T) {
		t.Parallel()

		next := func(*http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusTooManyRequests,
				Body:       io.NopCloser(&failingReader{err: io.ErrUnexpectedEOF}),
				Header:     make(http.Header),
			}, nil
		}

		_, err := normalizeErrorBodyMiddleware(&http.Request{}, next)
		require.ErrorIs(t, err, io.ErrUnexpectedEOF, "a truncated error body must surface as a read error so it stays retryable")
	})
}

type failingReader struct {
	err error
}

func (r *failingReader) Read([]byte) (int, error) {
	return 0, r.err
}

func TestGenerateNonStandardErrorBody(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":"rate limit reached"}`))
	}))
	defer server.Close()

	provider, err := New(
		WithAPIKey("test-api-key"),
		WithBaseURL(server.URL),
	)
	require.NoError(t, err)
	model, err := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")
	require.NoError(t, err)

	_, err = model.Generate(context.Background(), fantasy.Call{Prompt: testPrompt})

	var providerErr *fantasy.ProviderError
	require.ErrorAs(t, err, &providerErr, "a string-shaped error body must still surface as a provider error, not a JSON unmarshal error")
	require.Equal(t, http.StatusTooManyRequests, providerErr.StatusCode)
	require.Equal(t, "rate limit reached", providerErr.Message)
	require.Contains(t, string(providerErr.ResponseBody), "rate limit reached", "response dump must include the normalized body")
}

func TestGenerateTruncatedErrorBodyStaysRetryable(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Content-Length", "128")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":"rate limit reached`))
	}))
	defer server.Close()

	provider, err := New(
		WithAPIKey("test-api-key"),
		WithBaseURL(server.URL),
	)
	require.NoError(t, err)
	model, err := provider.LanguageModel(t.Context(), "gpt-3.5-turbo")
	require.NoError(t, err)

	_, err = model.Generate(context.Background(), fantasy.Call{Prompt: testPrompt})

	var providerErr *fantasy.ProviderError
	require.ErrorAs(t, err, &providerErr)
	require.True(t, providerErr.IsRetryable(), "a truncated error body must stay retryable, not degrade into a JSON parse error")
	require.ErrorIs(t, err, io.ErrUnexpectedEOF)
}
