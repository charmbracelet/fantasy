package fantasy

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestLostPingIsClassifiedLive pins http2ConnectionLostMessages to what the
// standard library actually produces.
//
// The entry it guards is a message, not a type, so nothing but this test
// notices if a future Go release rewords it. The failure that would follow
// is silent and bad: a connection killed while the machine slept stops
// being retryable and surfaces as a hard error instead, which is the exact
// bug the entry was added to fix.
//
// The setup reproduces a slept laptop rather than a closed socket. A proxy
// stops forwarding in both directions without sending FIN or RST, so the
// peer is gone but the kernel has no reason to say so, and only the
// transport's health-check ping can tell the difference.
func TestLostPingIsClassifiedLive(t *testing.T) {
	t.Parallel()

	// Streams a first chunk so the response is live, then blocks. The
	// freeze lands mid-stream, where a real one would.
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		_, _ = w.Write([]byte("data: hello\n\n"))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		<-r.Context().Done()
	}))
	srv.EnableHTTP2 = true
	srv.StartTLS()
	defer srv.Close()

	proxy := newFreezableProxy(t, srv.Listener.Addr().String())
	defer proxy.Close()

	srvTLS := srv.Client().Transport.(*http.Transport).TLSClientConfig.Clone()
	tr := &http.Transport{
		// Required: a Transport carrying a TLSClientConfig does not
		// negotiate h2 on its own, and the ping config is h2-only, so
		// without this the test would quietly prove nothing over HTTP/1.1.
		ForceAttemptHTTP2: true,
		TLSClientConfig:   srvTLS,
		HTTP2: &http.HTTP2Config{
			SendPingTimeout: 200 * time.Millisecond,
			PingTimeout:     200 * time.Millisecond,
		},
		DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, network, proxy.Addr())
		},
	}
	defer tr.CloseIdleConnections()

	resp, err := (&http.Client{Transport: tr}).Get(srv.URL)
	if err != nil {
		t.Fatalf("initial request: %v", err)
	}
	defer resp.Body.Close()
	if resp.ProtoMajor != 2 {
		t.Fatalf("got %s, want HTTP/2: the ping config only applies to h2", resp.Proto)
	}

	buf := make([]byte, 64)
	if _, err := resp.Body.Read(buf); err != nil {
		t.Fatalf("reading first chunk: %v", err)
	}

	proxy.Freeze()

	type readResult struct{ err error }
	done := make(chan readResult, 1)
	go func() {
		for {
			if _, err := resp.Body.Read(buf); err != nil {
				done <- readResult{err}
				return
			}
		}
	}()

	select {
	case got := <-done:
		if got.err == nil || errors.Is(got.err, io.EOF) {
			t.Fatalf("expected a transport failure, got %v", got.err)
		}
		if !IsTransportError(got.err) {
			t.Fatalf("IsTransportError(%q) = false.\n"+
				"The standard library's lost-connection message no longer matches "+
				"http2ConnectionLostMessages, so a dead connection is not retryable. "+
				"Update the list to match.", got.err)
		}
		if !isRetryableError(got.err) {
			t.Errorf("a lost connection is not retryable: %v", got.err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("the read never returned: no health check noticed the dead peer")
	}
}

// freezableProxy forwards TCP to a backend until Freeze, after which it
// drops every byte in both directions while holding both sockets open.
type freezableProxy struct {
	ln     net.Listener
	frozen atomic.Bool

	mu    sync.Mutex
	conns []net.Conn
	done  bool
}

func newFreezableProxy(t *testing.T, backend string) *freezableProxy {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("proxy listen: %v", err)
	}
	p := &freezableProxy{ln: ln}
	go func() {
		for {
			client, err := ln.Accept()
			if err != nil {
				return
			}
			upstream, err := net.Dial("tcp", backend)
			if err != nil {
				_ = client.Close()
				return
			}
			if !p.track(client, upstream) {
				return
			}
			go p.pipe(client, upstream)
			go p.pipe(upstream, client)
		}
	}()
	return p
}

// track registers a pair for teardown, or closes it and reports false when
// the proxy is already shutting down.
func (p *freezableProxy) track(conns ...net.Conn) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.done {
		for _, c := range conns {
			_ = c.Close()
		}
		return false
	}
	p.conns = append(p.conns, conns...)
	return true
}

func (p *freezableProxy) pipe(dst, src net.Conn) {
	buf := make([]byte, 32*1024)
	for {
		n, err := src.Read(buf)
		// Discarded rather than forwarded once frozen, so neither end
		// learns the connection is gone.
		if n > 0 && !p.frozen.Load() {
			if _, werr := dst.Write(buf[:n]); werr != nil {
				return
			}
		}
		if err != nil {
			return
		}
	}
}

func (p *freezableProxy) Addr() string { return p.ln.Addr().String() }

func (p *freezableProxy) Freeze() { p.frozen.Store(true) }

// Close tears down every connection as well as the listener. Freezing hides
// the client's disappearance from the server, so without this the server's
// handler waits on a request context that will never be cancelled and
// httptest.Server.Close blocks on it.
func (p *freezableProxy) Close() {
	_ = p.ln.Close()
	p.mu.Lock()
	defer p.mu.Unlock()
	p.done = true
	for _, c := range p.conns {
		_ = c.Close()
	}
	p.conns = nil
}
