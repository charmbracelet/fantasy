package google

import (
	"context"
	"net/http"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/internal/httpheaders"
)

type callUAKey struct{}

func withCallUA(ctx context.Context, call fantasy.Call) context.Context {
	if ua, ok := httpheaders.CallUserAgent(fantasy.Version, call.UserAgent, call.ModelSegment); ok {
		return context.WithValue(ctx, callUAKey{}, ua)
	}
	return ctx
}

func withObjectCallUA(ctx context.Context, call fantasy.ObjectCall) context.Context {
	if ua, ok := httpheaders.CallUserAgent(fantasy.Version, call.UserAgent, call.ModelSegment); ok {
		return context.WithValue(ctx, callUAKey{}, ua)
	}
	return ctx
}

func wrapHTTPClient(c *http.Client) *http.Client {
	if c == nil {
		c = http.DefaultClient
	}
	transport := c.Transport
	if transport == nil {
		transport = http.DefaultTransport
	}
	return &http.Client{
		Transport:     &uaTransport{base: transport},
		CheckRedirect: c.CheckRedirect,
		Jar:           c.Jar,
		Timeout:       c.Timeout,
	}
}

type uaTransport struct {
	base http.RoundTripper
}

func (t *uaTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if ua, ok := req.Context().Value(callUAKey{}).(string); ok && ua != "" {
		req = req.Clone(req.Context())
		req.Header.Set("User-Agent", ua)
	}
	return t.base.RoundTrip(req)
}
