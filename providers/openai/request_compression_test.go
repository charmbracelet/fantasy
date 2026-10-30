package openai

// Request compression is a transport concern, so these tests drive a stub
// server that plays the three roles a real one can take: it understands
// gzip, it does not, or the answer has nothing to do with the body at all.

import (
	"bytes"
	"compress/gzip"
	"crypto/rand"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

// seenRequest records what one arrival at the server looked like.
type seenRequest struct {
	contentEncoding string
	body            string
	contentLength   int64
}

// captureServer answers with the statuses in responses, one per arrival, and
// records every request it received.
type captureServer struct {
	server    *httptest.Server
	mu        sync.Mutex
	requests  []seenRequest
	responses []int
}

func newCaptureServer(t *testing.T, responses ...int) *captureServer {
	t.Helper()
	cs := &captureServer{responses: responses}
	cs.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		require.NoError(t, err)

		seen := seenRequest{
			contentEncoding: r.Header.Get("Content-Encoding"),
			body:            string(raw),
			contentLength:   r.ContentLength,
		}
		if seen.contentEncoding == "gzip" {
			zr, gzErr := gzip.NewReader(bytes.NewReader(raw))
			if gzErr != nil {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(`{"error":{"message":"not gzip"}}`))
				return
			}
			inflated, gzErr := io.ReadAll(zr)
			if gzErr != nil {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(`{"error":{"message":"bad gzip"}}`))
				return
			}
			seen.body = string(inflated)
		}

		cs.mu.Lock()
		cs.requests = append(cs.requests, seen)
		index := len(cs.requests) - 1
		cs.mu.Unlock()

		status := http.StatusOK
		if index < len(cs.responses) {
			status = cs.responses[index]
		}
		if status != http.StatusOK {
			w.WriteHeader(status)
			_, _ = w.Write([]byte(`{"error":{"message":"no","type":"invalid_request_error"}}`))
			return
		}
		var shape struct {
			Model string `json:"model"`
		}
		_ = json.Unmarshal([]byte(seen.body), &shape)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"z","created":1,"model":"` + shape.Model +
			`","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	t.Cleanup(cs.server.Close)
	return cs
}

func (cs *captureServer) received() []seenRequest {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	return append([]seenRequest(nil), cs.requests...)
}

// longPrompt is comfortably over the compression floor, which is what a
// conversation with any history in it looks like.
func longPrompt() fantasy.Prompt {
	return fantasy.Prompt{
		{
			Role:    fantasy.MessageRoleSystem,
			Content: []fantasy.MessagePart{fantasy.TextPart{Text: strings.Repeat("You are a helpful agent. ", 200)}},
		},
		{
			Role:    fantasy.MessageRoleUser,
			Content: []fantasy.MessagePart{fantasy.TextPart{Text: "hello"}},
		},
	}
}

func shortPrompt() fantasy.Prompt {
	return fantasy.Prompt{
		{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "hi"}}},
	}
}

func providerFor(t *testing.T, cs *captureServer, opts ...Option) fantasy.Provider {
	t.Helper()
	providerOpts := []Option{
		WithBaseURL(cs.server.URL),
		WithAPIKey("test"),
	}
	providerOpts = append(providerOpts, opts...)
	provider, err := New(providerOpts...)
	require.NoError(t, err)
	return provider
}

func generate(t *testing.T, provider fantasy.Provider, prompt fantasy.Prompt) {
	t.Helper()
	lm, err := provider.LanguageModel(t.Context(), "test-model")
	require.NoError(t, err)
	_, err = lm.Generate(t.Context(), fantasy.Call{Prompt: prompt})
	require.NoError(t, err)
}

func TestRequestCompression(t *testing.T) {
	t.Parallel()

	t.Run("off by default", func(t *testing.T) {
		t.Parallel()
		cs := newCaptureServer(t)

		generate(t, providerFor(t, cs), longPrompt())

		received := cs.received()
		require.Len(t, received, 1)
		require.Empty(t, received[0].contentEncoding)
	})

	t.Run("enabled compresses the body", func(t *testing.T) {
		t.Parallel()
		cs := newCaptureServer(t)

		generate(t, providerFor(t, cs, WithRequestCompression()), longPrompt())

		received := cs.received()
		require.Len(t, received, 1, "a server that reads gzip needs no retry")
		require.Equal(t, "gzip", received[0].contentEncoding)
		require.Contains(t, received[0].body, `"model":"test-model"`,
			"the server must be able to inflate what it was given")
	})

	t.Run("bodies under the floor are left alone", func(t *testing.T) {
		t.Parallel()
		cs := newCaptureServer(t)

		generate(t, providerFor(t, cs, WithRequestCompression()), shortPrompt())

		received := cs.received()
		require.Len(t, received, 1)
		require.Empty(t, received[0].contentEncoding)
	})

	t.Run("a server that cannot read gzip gets an uncompressed retry", func(t *testing.T) {
		t.Parallel()
		// First arrival answers 400, the way a Hyper without the
		// decompression middleware refuses a gzip body.
		cs := newCaptureServer(t, http.StatusBadRequest, http.StatusOK)

		generate(t, providerFor(t, cs, WithRequestCompression()), longPrompt())

		received := cs.received()
		require.Len(t, received, 2)
		require.Equal(t, "gzip", received[0].contentEncoding)
		require.Empty(t, received[1].contentEncoding)
		require.Contains(t, received[1].body, `"model":"test-model"`,
			"the retry must carry the same request, not an empty one")
	})

	t.Run("a server that answers 415 also gets the retry", func(t *testing.T) {
		t.Parallel()
		cs := newCaptureServer(t, http.StatusUnsupportedMediaType, http.StatusOK)

		generate(t, providerFor(t, cs, WithRequestCompression()), longPrompt())

		received := cs.received()
		require.Len(t, received, 2)
		require.Empty(t, received[1].contentEncoding)
	})

	// A 429 or a 5xx can follow a request that was already served and
	// charged. Replaying those would bill the customer twice.
	for _, status := range []int{
		http.StatusUnauthorized,
		http.StatusPaymentRequired,
		http.StatusRequestEntityTooLarge,
		http.StatusTooManyRequests,
		http.StatusInternalServerError,
		http.StatusBadGateway,
	} {
		t.Run("no retry on "+http.StatusText(status), func(t *testing.T) {
			t.Parallel()
			cs := newCaptureServer(t, status, http.StatusOK)

			provider := providerFor(t, cs, WithRequestCompression())
			lm, err := provider.LanguageModel(t.Context(), "test-model")
			require.NoError(t, err)
			_, err = lm.Generate(t.Context(), fantasy.Call{Prompt: longPrompt()})
			require.Error(t, err)

			received := cs.received()
			require.Len(t, received, 1, "a status that may follow real work must not be replayed")
			require.Equal(t, "gzip", received[0].contentEncoding)
		})
	}
}

// The compressor must not disturb the framing the SDK established: it declares
// the compressed length and keeps the body replayable.
func TestRequestWithBody(t *testing.T) {
	t.Parallel()

	original, err := http.NewRequestWithContext(t.Context(), http.MethodPost, "http://example.invalid", strings.NewReader(`{"a":"b"}`))
	require.NoError(t, err)
	original.Header.Set("Content-Type", "application/json")
	original.ContentLength = 9

	compressed := requestWithBody(original, []byte("gzipped"), true)

	require.Equal(t, "gzip", compressed.Header.Get("Content-Encoding"))
	require.EqualValues(t, 7, compressed.ContentLength,
		"a stale length makes net/http drop the request")
	require.NotEmpty(t, compressed.Header.Get("Content-Type"))

	replay, err := compressed.GetBody()
	require.NoError(t, err)
	replayed, err := io.ReadAll(replay)
	require.NoError(t, err)
	require.Equal(t, "gzipped", string(replayed))

	plain := requestWithBody(original, []byte("plain"), false)
	require.Empty(t, plain.Header.Get("Content-Encoding"))

	require.Equal(t, "", original.Header.Get("Content-Encoding"),
		"the caller's request is never modified")

	plainBody, err := io.ReadAll(original.Body)
	require.NoError(t, err)
	require.Equal(t, `{"a":"b"}`, string(plainBody), "the caller's body is still there")
}

func TestGzipForWire(t *testing.T) {
	t.Parallel()

	t.Run("repetitive JSON compresses", func(t *testing.T) {
		t.Parallel()
		body := []byte(`{"text":"` + strings.Repeat("hello world ", 500) + `"}`)
		out, ok := gzipForWire(body)
		require.True(t, ok)
		require.Less(t, len(out), len(body)/2)
	})

	t.Run("incompressible input is declined", func(t *testing.T) {
		t.Parallel()
		// Random bytes do not deflate. Sending them gzip-wrapped would spend
		// the header and the server's inflation for nothing, so the body goes
		// up exactly as it came.
		body := make([]byte, 1<<14)
		_, err := rand.Read(body)
		require.NoError(t, err)

		out, ok := gzipForWire(body)
		require.False(t, ok)
		require.Empty(t, out)
	})
}

func TestCompressedBodyCandidate(t *testing.T) {
	t.Parallel()

	// Past the floor, so the checks below are being tested for what they
	// actually gate on rather than for size.
	big := `{"a":"` + strings.Repeat("x", requestCompressionMinBytes) + `"}`

	tests := []struct {
		name     string
		method   string
		body     io.Reader
		ctype    string
		encoding string
		clearGet bool
		want     bool
	}{
		{"json body", http.MethodPost, strings.NewReader(big), "application/json", "", false, true},
		{"json with parameters", http.MethodPost, strings.NewReader(big), "application/json; charset=utf-8", "", false, true},
		{"empty body", http.MethodPost, nil, "application/json", "", false, false},
		{"already encoded", http.MethodPost, strings.NewReader(big), "application/json", "gzip", false, false},
		{"not json", http.MethodPost, strings.NewReader(big), "multipart/form-data", "", false, false},
		{"body with no way to replay it", http.MethodPost, strings.NewReader(big), "application/json", "", true, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			req, err := http.NewRequestWithContext(t.Context(), tt.method, "http://example.invalid", tt.body)
			require.NoError(t, err)
			if tt.ctype != "" {
				req.Header.Set("Content-Type", tt.ctype)
			}
			if tt.encoding != "" {
				req.Header.Set("Content-Encoding", tt.encoding)
			}
			if req.Body != nil {
				req.ContentLength = int64(len(big))
			}
			if tt.clearGet {
				req.GetBody = nil
			}

			_, got := compressedBodyCandidate(req)
			require.Equal(t, tt.want, got)
			if req.Body != nil {
				remaining, _ := io.ReadAll(req.Body)
				require.Equal(t, len(big), len(remaining),
					"probing a body must not consume the caller's copy")
			}
		})
	}

	t.Run("too short to be worth it", func(t *testing.T) {
		t.Parallel()
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, "http://example.invalid", strings.NewReader(`{"a":1}`))
		require.NoError(t, err)
		req.Header.Set("Content-Type", "application/json")
		req.ContentLength = 7

		_, ok := compressedBodyCandidate(req)
		require.False(t, ok)
	})
}

// Enabling compression without supplying a client must not quietly drop the
// SDK's own defaults.
func TestWithRequestCompressionKeepsClientBehavior(t *testing.T) {
	t.Parallel()

	t.Run("no client supplied", func(t *testing.T) {
		t.Parallel()
		wrapped := withRequestCompression(nil)
		client, ok := wrapped.(*http.Client)
		require.True(t, ok, "a native client keeps the SDK's redirect and origin handling")

		transport, ok := client.Transport.(*compressingTransport)
		require.True(t, ok)
		inner, ok := transport.base.(*http.Transport)
		require.True(t, ok)
		require.Equal(t, sdkDefaultResponseHeaderTimeout, inner.ResponseHeaderTimeout)
	})

	t.Run("caller client", func(t *testing.T) {
		t.Parallel()
		supplied := &http.Client{Transport: &countingRoundTripper{}, Timeout: 30}
		wrapped := withRequestCompression(supplied)
		client, ok := wrapped.(*http.Client)
		require.True(t, ok)
		require.EqualValues(t, 30, client.Timeout, "the caller's settings survive")

		transport, ok := client.Transport.(*compressingTransport)
		require.True(t, ok)
		_, ok = transport.base.(*countingRoundTripper)
		require.True(t, ok, "a caller's own transport stays in the path")

		require.NotSame(t, supplied, client)
	})

	t.Run("bespoke doer", func(t *testing.T) {
		t.Parallel()
		wrapped := withRequestCompression(&countingClient{})
		_, ok := wrapped.(*compressingClient)
		require.True(t, ok)
	})
}

type countingRoundTripper struct{}

func (countingRoundTripper) RoundTrip(*http.Request) (*http.Response, error) {
	return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader("{}"))}, nil
}

type countingClient struct{ calls int }

func (c *countingClient) Do(*http.Request) (*http.Response, error) {
	c.calls++
	return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader("{}"))}, nil
}
