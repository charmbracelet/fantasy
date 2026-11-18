package openai

import (
	"bytes"
	"compress/gzip"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/charmbracelet/openai-go/option"
	brrr "github.com/molecule-man/go-brrr"
)

// Request compression is for clients that re-upload a whole conversation on
// every turn: gzip shrinks that JSON by roughly 4x, and brotli does better
// for less CPU, which matters on a slow uplink. Only bodies the server reads
// as JSON are compressed, and an uncompressed request is what the server gets
// whenever anything is in doubt.

const (
	// requestCompressionMinBytes leaves short bodies alone: compressing a
	// few hundred bytes of JSON saves nothing and costs a header on every
	// request.
	requestCompressionMinBytes = 1024

	// requestCompressionGzipLevel is gzip.BestSpeed: a third of the CPU of
	// the default level for a ratio close to the best deflate reaches on
	// real conversation JSON, on the critical path of every request.
	requestCompressionGzipLevel = gzip.BestSpeed

	// requestCompressionBrotliLevel is quality 1, the two-pass fast
	// encoder: the speed-matched answer to gzip's BestSpeed while still
	// beating its ratio on conversation JSON. Higher levels buy ratio with
	// latency, which is the wrong trade on a request's critical path.
	requestCompressionBrotliLevel = 1

	// sdkDefaultResponseHeaderTimeout mirrors the SDK's own default client
	// (openai-go default_http_client.go) so that enabling compression
	// without supplying a client changes nothing about how requests fail
	// when a server never answers.
	sdkDefaultResponseHeaderTimeout = 10 * time.Minute
)

// RequestEncoding identifies a Content-Encoding the compressor can put on a
// JSON request body.
type RequestEncoding int

const (
	// EncodingGzip is Content-Encoding: gzip.
	EncodingGzip RequestEncoding = iota
	// EncodingBrotli is Content-Encoding: br.
	EncodingBrotli
)

// requestCodec pairs a Content-Encoding name with its encoder. encode
// reports whether the result is worth sending: incompressible JSON, which is
// unusual, would spend the header and the server's inflation for nothing.
type requestCodec struct {
	name   string
	encode func([]byte) ([]byte, bool)
}

var requestCodecs = map[RequestEncoding]requestCodec{
	EncodingGzip:   {name: "gzip", encode: gzipForWire},
	EncodingBrotli: {name: "br", encode: brotliForWire},
}

// WithRequestCompression compresses JSON request bodies, for servers that
// accept a Content-Encoding, using the encodings given, most preferred first.
// No encodings is equivalent to gzip alone, which is what it did before the
// argument existed.
//
// It is off by default. Enabling it changes nothing for requests the server
// would reject before doing work: a 400 or 415 answer to a compressed body
// falls back to the next encoding, then to an uncompressed request, so a
// server without support for an encoding (an older deployment, or one reached
// through a base URL override) behaves exactly as it did before the option
// existed. The fallback deliberately matches only those statuses: a 429 or a
// 5xx can arrive after a request was served and charged, and replaying it
// would be a second request, not a fallback. The first encoding the server
// accepts is remembered, so the cascade is paid once per client, not once
// per request.
//
// The wrapper never touches `Accept-Encoding`. Setting it by hand disables
// net/http's transparent response gunzipping, and this package has no
// decompressor to make up for that.
func WithRequestCompression(encodings ...RequestEncoding) Option {
	return func(o *options) {
		if len(encodings) == 0 {
			encodings = []RequestEncoding{EncodingGzip}
		}
		o.requestEncodings = encodings
	}
}

// withRequestCompression wraps client so its requests are compressed. The
// result keeps whatever behavior the caller supplied: a native *http.Client
// comes back as a clone with the compressor under its redirect policy, so the
// SDK's origin checks keep working; a caller's bespoke HTTPClient keeps its
// own Do. The compressor is supplied rather than built here so every model
// made from one provider shares what the server accepted.
func withRequestCompression(client option.HTTPClient, compressor *requestCompressor) option.HTTPClient {
	switch base := client.(type) {
	case nil:
		return &http.Client{Transport: &compressingTransport{base: sdkDefaultTransport(), requestCompressor: compressor}}
	case *http.Client:
		transport := base.Transport
		if transport == nil {
			transport = http.DefaultTransport
		}
		clone := *base
		clone.Transport = &compressingTransport{base: transport, requestCompressor: compressor}
		return &clone
	default:
		return &compressingClient{base: client, requestCompressor: compressor}
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

// newRequestCompressor builds a compressor over an encoding preference list.
func newRequestCompressor(preferences []RequestEncoding) *requestCompressor {
	return &requestCompressor{preferences: preferences}
}

// requestCompressor carries the encoding preference list and what the server
// has accepted so far.
type requestCompressor struct {
	preferences []RequestEncoding

	// settled is the index into preferences of the encoding the server
	// accepted, or len(preferences) when only plain bodies get through. It
	// starts at 0 and is stored on the first non-refusal answer, so a server
	// that speaks gzip but not brotli costs one refused request per client
	// rather than one per request.
	settled atomic.Int64
}

// send performs one send of req, compressed when its body qualifies, falling
// back down the preference list and finally to plain when the server's answer
// says it could not read what it was given.
func (c *requestCompressor) send(req *http.Request, send func(*http.Request) (*http.Response, error)) (*http.Response, error) {
	body, ok := compressedBodyCandidate(req)
	if !ok {
		return send(req)
	}

	start := int(c.settled.Load())
	if start >= len(c.preferences) {
		// Only plain bodies ever got through. Skipping the cascade also
		// skips the encoding cost for a server that would refuse it.
		return send(req)
	}

	attempted := false
	for i := start; i < len(c.preferences); i++ {
		codec := requestCodecs[c.preferences[i]]
		compressed, ok := codec.encode(body)
		if !ok {
			continue
		}
		attempted = true

		res, err := send(requestWithBody(req, compressed, codec.name))
		if err != nil {
			return nil, err //nolint:wrapcheck // transport errors belong to the caller
		}
		if bodyRejected(res) {
			// The server could not read this encoding. Drain and drop the
			// rejection, then try the next one.
			drainAndClose(res)
			continue
		}
		c.settled.Store(int64(i))
		return res, nil
	}

	// Every encoding was refused or declined. Send the same request
	// uncompressed; whether the server can serve it at all is for the
	// caller to find out.
	res, err := send(requestWithBody(req, body, ""))
	if err != nil {
		return nil, err //nolint:wrapcheck // transport errors belong to the caller
	}
	// Settle on plain only when an encoding was actually refused: a body
	// every encoder declined says nothing about the server, and a plain
	// answer that is itself a refusal says nothing either.
	if attempted && !bodyRejected(res) {
		c.settled.Store(int64(len(c.preferences)))
	}
	return res, nil
}

// compressingTransport compresses request bodies and falls back down the
// preference list when the answer says the server could not read them.
type compressingTransport struct {
	base http.RoundTripper
	*requestCompressor
}

// RoundTrip sends req, compressing its body when that is possible and worth
// it. The caller's request is never modified.
func (t *compressingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return t.send(req, t.base.RoundTrip)
}

// compressingClient is the same compressor for a caller's bespoke HTTPClient,
// which the SDK keeps at Do level.
type compressingClient struct {
	base option.HTTPClient
	*requestCompressor
}

// Do sends req, compressing its body when that is possible and worth it.
func (c *compressingClient) Do(req *http.Request) (*http.Response, error) {
	return c.send(req, c.base.Do)
}

// CloseIdleConnections forwards to the base client when it can.
func (c *compressingClient) CloseIdleConnections() {
	if closer, ok := c.base.(interface{ CloseIdleConnections() }); ok {
		closer.CloseIdleConnections()
	}
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
		// Somebody already encoded this body. Stacking another encoding on
		// top would hand the server a body it has to undo in the right order.
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
	zw, err := gzip.NewWriterLevel(buf, requestCompressionGzipLevel)
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

// brotliForWire is the same judgment for brotli. The one-shot API feeds the
// encoder the exact input size as its SizeHint, so no writer or pooled
// buffer is needed for a body that is already in memory.
func brotliForWire(body []byte) ([]byte, bool) {
	out, err := brrr.Compress(body, requestCompressionBrotliLevel)
	if err != nil || len(out) >= len(body) {
		return nil, false
	}
	return out, true
}

// requestWithBody rebuilds req around a body the caller already holds. The
// original request is not modified, in keeping with the RoundTripper contract,
// and both ContentLength and GetBody describe the new body: a stale length
// makes net/http report "ContentLength=N with Body length M" and drop the
// request, and a GetBody that replays different bytes turns any retry into a
// request the server has never seen.
func requestWithBody(req *http.Request, body []byte, encoding string) *http.Request {
	clone := req.Clone(req.Context())
	clone.Body = io.NopCloser(bytes.NewReader(body))
	clone.ContentLength = int64(len(body))
	clone.GetBody = func() (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(body)), nil
	}
	if encoding != "" {
		clone.Header.Set("Content-Encoding", encoding)
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
