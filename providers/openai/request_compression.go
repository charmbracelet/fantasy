package openai

import (
	"bytes"
	"compress/gzip"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/charmbracelet/openai-go/option"
)

// Request compression is for clients that re-upload a whole conversation on
// every turn: gzip shrinks that JSON by roughly 4x, which matters on a slow
// uplink and is a few milliseconds of CPU on a fast one. Only bodies the
// server reads as JSON are compressed, and an uncompressed request is what the
// server gets whenever anything is in doubt.

const (
	// requestCompressionMinBytes leaves short bodies alone: compressing a
	// few hundred bytes of JSON saves nothing and costs a gzip header on
	// every request.
	requestCompressionMinBytes = 1024

	// requestCompressionLevel is gzip.BestSpeed: a third of the CPU of the
	// default level for a ratio close to the best deflate reaches on real
	// conversation JSON, on the critical path of every request.
	requestCompressionLevel = gzip.BestSpeed

	// sdkDefaultResponseHeaderTimeout mirrors the SDK's own default client
	// (openai-go default_http_client.go) so that enabling compression
	// without supplying a client changes nothing about how requests fail
	// when a server never answers.
	sdkDefaultResponseHeaderTimeout = 10 * time.Minute
)

// WithRequestCompression gzip-compresses JSON request bodies, for servers that
// accept `Content-Encoding: gzip`.
//
// It is off by default. Enabling it changes nothing for requests the server
// would reject before doing work: a 400 or 415 answer to a compressed body is
// retried once, uncompressed, so a server without this support (an older
// deployment, or one reached through a base URL override) behaves exactly as
// it did before the option existed. The fallback deliberately matches only
// those statuses: a 429 or a 5xx can arrive after a request was served and
// charged, and replaying it would be a second request, not a fallback.
//
// The wrapper never touches `Accept-Encoding`. Setting it by hand disables
// net/http's transparent response gunzipping, and this package has no
// decompressor to make up for that.
func WithRequestCompression() Option {
	return func(o *options) {
		o.requestCompression = true
	}
}

// withRequestCompression wraps client so its requests are compressed. The
// result keeps whatever behavior the caller supplied: a native *http.Client
// comes back as a clone with the compressor under its redirect policy, so the
// SDK's origin checks keep working; a caller's bespoke HTTPClient keeps its
// own Do.
func withRequestCompression(client option.HTTPClient) option.HTTPClient {
	switch base := client.(type) {
	case nil:
		return &http.Client{Transport: &compressingTransport{base: sdkDefaultTransport()}}
	case *http.Client:
		transport := base.Transport
		if transport == nil {
			transport = http.DefaultTransport
		}
		clone := *base
		clone.Transport = &compressingTransport{base: transport}
		return &clone
	default:
		return &compressingClient{base: client}
	}
}

// sdkDefaultTransport reproduces the transport the SDK builds for itself when
// no client is supplied: http.DefaultTransport with a header timeout so a
// server that accepts the connection and never answers fails instead of
// hanging, and the wrap preserved when DefaultTransport has been wrapped.
func sdkDefaultTransport() http.RoundTripper {
	if t, ok := http.DefaultTransport.(*http.Transport); ok {
		t = t.Clone()
		t.ResponseHeaderTimeout = sdkDefaultResponseHeaderTimeout
		return t
	}
	return http.DefaultTransport
}

// compressingTransport gzip-compresses request bodies and falls back to the
// plain body when the answer says the server could not read it.
type compressingTransport struct {
	base http.RoundTripper
}

// RoundTrip sends req, compressing its body when that is possible and worth
// it. The caller's request is never modified.
func (t *compressingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return sendCompressed(req, t.base.RoundTrip)
}

// compressingClient is the same compressor for a caller's bespoke HTTPClient,
// which the SDK keeps at Do level.
type compressingClient struct {
	base option.HTTPClient
}

// Do sends req, compressing its body when that is possible and worth it.
func (c *compressingClient) Do(req *http.Request) (*http.Response, error) {
	return sendCompressed(req, c.base.Do)
}

// CloseIdleConnections forwards to the base client when it can.
func (c *compressingClient) CloseIdleConnections() {
	if closer, ok := c.base.(interface{ CloseIdleConnections() }); ok {
		closer.CloseIdleConnections()
	}
}

// sendCompressed performs one send of req, compressed when its body qualifies,
// retrying uncompressed when the server's answer says it could not parse the
// body it was given.
func sendCompressed(req *http.Request, send func(*http.Request) (*http.Response, error)) (*http.Response, error) {
	body, ok := compressedBodyCandidate(req)
	if !ok {
		return send(req)
	}

	compressed, ok := gzipForWire(body)
	if !ok {
		return send(req)
	}

	res, err := send(requestWithBody(req, compressed, true))
	if err != nil {
		return nil, err //nolint:wrapcheck // transport errors belong to the caller
	}
	if !bodyRejected(res) {
		return res, nil
	}

	// The server could not read the compressed body. Drain and drop the
	// rejection, then send the same request uncompressed.
	drainAndClose(res)
	return send(requestWithBody(req, body, false))
}

// compressedBodyCandidate reports whether req's body is worth compressing, and
// hands it over without consuming the request's own copy.
//
// The bytes come from GetBody, not Body, on purpose: reading Body would empty
// the request the caller owns, and a body judged too short to compress still
// has to be sent. The SDK serializes every request body into memory and always
// records GetBody, so anything it built qualifies; a hand-made streaming body
// does not, and is left alone.
func compressedBodyCandidate(req *http.Request) ([]byte, bool) {
	if req.Body == nil || req.Body == http.NoBody || req.GetBody == nil {
		return nil, false
	}
	if !strings.HasPrefix(req.Header.Get("Content-Type"), "application/json") {
		// Multipart bodies carry file attachments, which are already dense,
		// and anything else is not the JSON this endpoint was built for.
		return nil, false
	}
	if req.ContentLength == 0 {
		return nil, false
	}
	if req.Header.Get("Content-Encoding") != "" {
		// Somebody already encoded this body. Stacking gzip on top would
		// hand the server an encoding it has to undo in the right order.
		return nil, false
	}

	body, err := req.GetBody()
	if err != nil {
		return nil, false
	}
	defer body.Close() //nolint:errcheck

	content, err := io.ReadAll(body)
	if err != nil || len(content) < requestCompressionMinBytes {
		return nil, false
	}
	return content, true
}

// gzipForWire compresses body, reporting whether the result is worth sending:
// incompressible JSON, which is unusual, would spend the header and the
// server's inflation for nothing.
func gzipForWire(body []byte) ([]byte, bool) {
	buf := bytes.NewBuffer(nil)
	zw, err := gzip.NewWriterLevel(buf, requestCompressionLevel)
	if err != nil {
		return nil, false
	}
	if _, err := zw.Write(body); err != nil {
		return nil, false
	}
	if err := zw.Close(); err != nil {
		return nil, false
	}
	if buf.Len() >= len(body) {
		return nil, false
	}
	return buf.Bytes(), true
}

// requestWithBody rebuilds req around a body the caller already holds. The
// original request is not modified, in keeping with the RoundTripper contract,
// and both ContentLength and GetBody describe the new body: a stale length
// makes net/http report "ContentLength=N with Body length M" and drop the
// request, and a GetBody that replays different bytes turns any retry into a
// request the server has never seen.
func requestWithBody(req *http.Request, body []byte, gzipped bool) *http.Request {
	clone := req.Clone(req.Context())
	clone.Body = io.NopCloser(bytes.NewReader(body))
	clone.ContentLength = int64(len(body))
	clone.GetBody = func() (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(body)), nil
	}
	if gzipped {
		clone.Header.Set("Content-Encoding", "gzip")
	} else {
		clone.Header.Del("Content-Encoding")
	}
	return clone
}

// bodyRejected reports whether res is an answer a server gives to a request
// whose body it could not parse. Those are the only statuses where replaying
// is known to be safe: a 400 or 415 is decided before the request does any
// work, and both arrive before a streaming response begins.
func bodyRejected(res *http.Response) bool {
	return res.StatusCode == http.StatusBadRequest || res.StatusCode == http.StatusUnsupportedMediaType
}

// drainAndClose empties res so the connection underneath can be reused, which
// is the polite thing to do before the retry takes a fresh one.
func drainAndClose(res *http.Response) {
	_, _ = io.Copy(io.Discard, io.LimitReader(res.Body, 64<<10))
	_ = res.Body.Close()
}
