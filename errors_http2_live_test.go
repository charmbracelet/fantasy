package fantasy

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

// TestLostPingIsClassifiedLive pins http2ConnectionLostMessages to the
// message Go actually produces. The entry is matched as a string, so a
// reword in a future release would silently stop dead connections being
// retryable, and only this test would notice.
func TestLostPingIsClassifiedLive(t *testing.T) {
	t.Parallel()

	// Streams a chunk so the response is live, then holds the stream open
	// so the freeze lands mid-stream, where a real one would.
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("data: hello\n\n"))
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	srv.EnableHTTP2 = true
	srv.StartTLS()
	defer srv.Close()

	conn := &freezableConn{closed: make(chan struct{}), frozen: make(chan struct{})}
	defer conn.Close()

	tr := &http.Transport{
		// A Transport carrying a TLSClientConfig does not negotiate h2 on
		// its own, and the ping is h2-only, so without this the test would
		// pass over HTTP/1.1 having proved nothing.
		ForceAttemptHTTP2: true,
		TLSClientConfig:   srv.Client().Transport.(*http.Transport).TLSClientConfig.Clone(),
		HTTP2: &http.HTTP2Config{
			SendPingTimeout: 200 * time.Millisecond,
			PingTimeout:     200 * time.Millisecond,
		},
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			return conn.dial(ctx, network, addr)
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

	conn.freeze()

	read := make(chan error, 1)
	go func() {
		for {
			if _, err := resp.Body.Read(buf); err != nil {
				read <- err
				return
			}
		}
	}()

	select {
	case err := <-read:
		if err == nil || errors.Is(err, io.EOF) {
			t.Fatalf("expected a transport failure, got %v", err)
		}
		if !IsTransportError(err) {
			t.Fatalf("IsTransportError(%q) = false, so a dead connection is no "+
				"longer retryable. Go reworded it; update http2ConnectionLostMessages.", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("the read never returned: no health check noticed the dead peer")
	}
}

// freezableConn is a client connection that goes deaf and mute on freeze
// while staying open, which is what a socket killed by a sleeping laptop
// looks like: the peer is gone, but no FIN or RST ever says so, and only
// the transport's health-check ping can tell that from a slow reply.
type freezableConn struct {
	net.Conn
	frozen chan struct{}
	closed chan struct{}
	once   sync.Once
}

func (c *freezableConn) dial(ctx context.Context, network, addr string) (net.Conn, error) {
	conn, err := (&net.Dialer{}).DialContext(ctx, network, addr)
	c.Conn = conn
	return c, err
}

func (c *freezableConn) freeze() { close(c.frozen) }

func (c *freezableConn) isFrozen() bool {
	select {
	case <-c.frozen:
		return true
	default:
		return false
	}
}

// Read parks until teardown once frozen, so no frame arrives to reset the
// transport's read-idle timer and the health check fires.
func (c *freezableConn) Read(p []byte) (int, error) {
	if c.isFrozen() {
		<-c.closed
		return 0, io.EOF
	}
	return c.Conn.Read(p)
}

// Write reports success without sending once frozen, so the health-check
// ping never reaches the server and goes unanswered.
func (c *freezableConn) Write(p []byte) (int, error) {
	if c.isFrozen() {
		return len(p), nil
	}
	return c.Conn.Write(p)
}

// Close releases any parked Read and lets the server see EOF, which is
// what ends its handler and unblocks httptest.Server.Close.
func (c *freezableConn) Close() error {
	c.once.Do(func() { close(c.closed) })
	return c.Conn.Close()
}
