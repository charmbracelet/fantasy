package replaytest

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// connectionClosedMarker, when a fixture event consists of exactly this
// line, makes the replay server flush the events written so far and then
// close the connection without a terminator. It simulates an upstream that
// dies mid-stream.
const connectionClosedMarker = "<connection closed>"

// Server replays a fixture's response over HTTP and records every request
// body it receives.
type Server struct {
	server *httptest.Server

	mu       sync.Mutex
	requests [][]byte
	calls    int

	fixture        *Fixture
	subsequentSSE  []string
	subsequentJSON []byte
	sseSet         bool
	jsonSet        bool
}

// ServeOption customizes the replay server.
type ServeOption func(*Server)

// WithSubsequentSSE overrides the canned response for streaming requests
// after the first. Events are raw SSE event blocks, exactly as produced by
// SplitSSE.
func WithSubsequentSSE(events ...string) ServeOption {
	return func(s *Server) {
		s.subsequentSSE = events
		s.sseSet = true
	}
}

// WithSubsequentJSON overrides the canned response for non-streaming
// requests after the first.
func WithSubsequentJSON(body []byte) ServeOption {
	return func(s *Server) {
		s.subsequentJSON = body
		s.jsonSet = true
	}
}

// Serve starts an httptest server that answers the first request with the
// fixture's response: the raw SSE events for a streaming fixture (written
// one event at a time, flushed after each) or the raw JSON body for a
// non-streaming fixture. Events are never merged or split.
//
// Second and later requests get a canned minimal finish response shaped by
// the request endpoint, unless WithSubsequentSSE or WithSubsequentJSON
// supplies one. The canned responses are keyed by what the request looks
// like (streaming or not, messages endpoint or chat-completions endpoint),
// never by provider name.
func Serve(t testing.TB, f *Fixture, opts ...ServeOption) *Server {
	t.Helper()
	s := &Server{fixture: f}
	for _, opt := range opts {
		opt(s)
	}
	s.server = httptest.NewServer(http.HandlerFunc(s.serveHTTP))
	f.server = s
	t.Cleanup(s.server.Close)
	return s
}

// URL returns the server's base URL.
func (s *Server) URL() string {
	return s.server.URL
}

// Requests returns a copy of every request body the server has received, in
// arrival order.
func (s *Server) Requests() [][]byte {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([][]byte, len(s.requests))
	copy(out, s.requests)
	return out
}

func (s *Server) serveHTTP(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "replaytest: failed to read request body", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	s.requests = append(s.requests, body)
	call := s.calls
	s.calls++
	s.mu.Unlock()

	streaming := isStreamingRequest(body)
	if call == 0 {
		if len(s.fixture.Events) > 0 {
			writeSSE(w, s.fixture.Events)
			return
		}
		writeJSON(w, s.fixture.Body)
		return
	}

	if s.sseSet && streaming {
		writeSSE(w, s.subsequentSSE)
		return
	}
	if s.jsonSet && !streaming {
		writeJSON(w, s.subsequentJSON)
		return
	}
	if streaming {
		writeSSE(w, SplitSSE(cannedSSE(messagesShaped(r.URL.Path))))
		return
	}
	writeJSON(w, []byte(cannedJSON(messagesShaped(r.URL.Path))))
}

func writeSSE(w http.ResponseWriter, events []string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	rc := http.NewResponseController(w)
	for _, event := range events {
		if strings.TrimSpace(event) == connectionClosedMarker {
			_ = rc.Flush()
			hijacker, ok := w.(http.Hijacker)
			if !ok {
				return
			}
			conn, _, err := hijacker.Hijack()
			if err != nil {
				return
			}
			_ = conn.Close()
			return
		}
		if _, err := fmt.Fprint(w, event+"\n\n"); err != nil {
			return
		}
		_ = rc.Flush()
	}
}

func writeJSON(w http.ResponseWriter, body []byte) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(body)
}

// isStreamingRequest reports whether the request body asks for a streaming
// response.
func isStreamingRequest(body []byte) bool {
	var req struct {
		Stream bool `json:"stream"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return false
	}
	return req.Stream
}

// messagesShaped reports whether the request path targets a messages-style
// endpoint (as opposed to a chat-completions-style one).
func messagesShaped(path string) bool {
	return strings.Contains(path, "/messages")
}

func cannedSSE(messages bool) string {
	if messages {
		return `event: message_start
data: {"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","content":[],"model":"test-model","stop_reason":null,"usage":{"input_tokens":1,"output_tokens":1}}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}

event: message_stop
data: {"type":"message_stop"}`
	}
	return `data: {"id":"test-response","created":0,"model":"test-model","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]`
}

func cannedJSON(messages bool) string {
	if messages {
		return `{"id":"msg_test","type":"message","role":"assistant","content":[{"type":"text","text":""}],"model":"test-model","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
	}
	return `{"id":"test-response","created":0,"model":"test-model","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`
}
