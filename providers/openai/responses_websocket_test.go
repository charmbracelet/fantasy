package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"charm.land/fantasy"
	"github.com/gorilla/websocket"
)

// mockWSServer creates a test WebSocket server that sends predefined events.
func mockWSServer(t *testing.T, handler func(conn *websocket.Conn)) *httptest.Server {
	t.Helper()
	upgrader := websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool { return true },
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Fatalf("upgrade: %v", err)
			return
		}
		defer conn.Close()
		handler(conn)
	}))
	return server
}

func wsURLFromHTTP(httpURL string) string {
	return strings.Replace(httpURL, "http://", "ws://", 1)
}

func TestWSTransport_Connect(t *testing.T) {
	server := mockWSServer(t, func(conn *websocket.Conn) {
		// Just accept and hold connection
		for {
			_, _, err := conn.ReadMessage()
			if err != nil {
				return
			}
		}
	})
	defer server.Close()

	ws := newWSTransport(wsURLFromHTTP(server.URL), "test-key", nil)
	// Override wsURL to point at test server
	ws.baseURL = wsURLFromHTTP(server.URL)

	err := ws.connect(context.Background())
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer ws.Close()

	if ws.conn == nil {
		t.Fatal("expected conn to be set")
	}
	if ws.connectedAt.IsZero() {
		t.Fatal("expected connectedAt to be set")
	}
}

func TestWSTransport_EnsureConnected_Reconnect(t *testing.T) {
	var connectCount int
	var mu sync.Mutex

	server := mockWSServer(t, func(conn *websocket.Conn) {
		mu.Lock()
		connectCount++
		mu.Unlock()
		for {
			_, _, err := conn.ReadMessage()
			if err != nil {
				return
			}
		}
	})
	defer server.Close()

	ws := newWSTransport(wsURLFromHTTP(server.URL), "test-key", nil)
	ws.baseURL = wsURLFromHTTP(server.URL)

	// First connect
	err := ws.ensureConnected(context.Background())
	if err != nil {
		t.Fatalf("first connect: %v", err)
	}

	// Should not reconnect (within threshold)
	err = ws.ensureConnected(context.Background())
	if err != nil {
		t.Fatalf("second connect: %v", err)
	}

	mu.Lock()
	if connectCount != 1 {
		t.Fatalf("expected 1 connection, got %d", connectCount)
	}
	mu.Unlock()

	// Simulate expired connection by backdating connectedAt
	ws.connectedAt = time.Now().Add(-56 * time.Minute)

	err = ws.ensureConnected(context.Background())
	if err != nil {
		t.Fatalf("reconnect: %v", err)
	}
	defer ws.Close()

	// Wait a moment for the server to register the new connection
	time.Sleep(50 * time.Millisecond)

	mu.Lock()
	if connectCount != 2 {
		t.Fatalf("expected 2 connections after reconnect, got %d", connectCount)
	}
	mu.Unlock()
}

func TestWSTransport_SendResponseCreate(t *testing.T) {
	server := mockWSServer(t, func(conn *websocket.Conn) {
		// Read the response.create message
		_, message, err := conn.ReadMessage()
		if err != nil {
			t.Errorf("read: %v", err)
			return
		}

		var evt map[string]json.RawMessage
		if err := json.Unmarshal(message, &evt); err != nil {
			t.Errorf("unmarshal: %v", err)
			return
		}

		var eventType string
		json.Unmarshal(evt["type"], &eventType)
		if eventType != "response.create" {
			t.Errorf("expected type response.create, got %s", eventType)
			return
		}

		// Send response events
		events := []string{
			`{"type":"response.created","response":{"id":"resp_123","status":"in_progress"}}`,
			`{"type":"response.output_item.added","output_index":0,"item":{"id":"item_1","type":"message","role":"assistant","content":[]}}`,
			`{"type":"response.output_text.delta","output_index":0,"content_index":0,"item_id":"item_1","delta":"Hello"}`,
			`{"type":"response.output_text.delta","output_index":0,"content_index":0,"item_id":"item_1","delta":" world"}`,
			`{"type":"response.completed","response":{"id":"resp_123","status":"completed","output":[{"id":"item_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello world"}]}],"usage":{"input_tokens":10,"output_tokens":5}}}`,
		}
		for _, event := range events {
			if err := conn.WriteMessage(websocket.TextMessage, []byte(event)); err != nil {
				return
			}
		}
	})
	defer server.Close()

	ws := newWSTransport(wsURLFromHTTP(server.URL), "test-key", nil)
	ws.baseURL = wsURLFromHTTP(server.URL)

	ws.mu.Lock()
	defer ws.mu.Unlock()

	body := json.RawMessage(`{"model":"gpt-4o","input":[]}`)
	events, err := ws.sendResponseCreate(context.Background(), body)
	if err != nil {
		t.Fatalf("sendResponseCreate: %v", err)
	}

	var eventTypes []string
	for evt := range events {
		eventTypes = append(eventTypes, evt.Type)
	}

	expected := []string{
		"response.created",
		"response.output_item.added",
		"response.output_text.delta",
		"response.output_text.delta",
		"response.completed",
	}

	if len(eventTypes) != len(expected) {
		t.Fatalf("expected %d events, got %d: %v", len(expected), len(eventTypes), eventTypes)
	}

	for i, eventType := range eventTypes {
		if eventType != expected[i] {
			t.Errorf("event %d: expected %s, got %s", i, expected[i], eventType)
		}
	}
}

func TestWSTransport_PreviousResponseIDChaining(t *testing.T) {
	ws := newWSTransport("wss://api.openai.com/v1", "test-key", nil)
	ws.lastResponseID = "resp_prev_123"

	call := fantasy.Call{
		ProviderOptions: fantasy.ProviderOptions{},
	}

	body := json.RawMessage(`{"model":"gpt-4o","input":[]}`)
	result := ws.applyWSOptions(body, call)

	var resultMap map[string]json.RawMessage
	if err := json.Unmarshal(result, &resultMap); err != nil {
		t.Fatalf("unmarshal result: %v", err)
	}

	prevIDRaw, ok := resultMap["previous_response_id"]
	if !ok {
		t.Fatal("expected previous_response_id in result")
	}

	var prevID string
	json.Unmarshal(prevIDRaw, &prevID)
	if prevID != "resp_prev_123" {
		t.Errorf("expected resp_prev_123, got %s", prevID)
	}
}

func TestWSTransport_GenerateWarmup(t *testing.T) {
	ws := newWSTransport("wss://api.openai.com/v1", "test-key", nil)

	warmup := true
	call := fantasy.Call{
		ProviderOptions: fantasy.ProviderOptions{
			Name: &ResponsesProviderOptions{
				GenerateWarmup: &warmup,
			},
		},
	}

	body := json.RawMessage(`{"model":"gpt-4o","input":[]}`)
	result := ws.applyWSOptions(body, call)

	var resultMap map[string]json.RawMessage
	if err := json.Unmarshal(result, &resultMap); err != nil {
		t.Fatalf("unmarshal result: %v", err)
	}

	genRaw, ok := resultMap["generate"]
	if !ok {
		t.Fatal("expected generate in result")
	}

	if string(genRaw) != "false" {
		t.Errorf("expected generate=false, got %s", string(genRaw))
	}
}

func TestWSTransport_ExplicitPreviousResponseIDOverridesAuto(t *testing.T) {
	ws := newWSTransport("wss://api.openai.com/v1", "test-key", nil)
	ws.lastResponseID = "resp_auto_123"

	// Set explicit previous_response_id in the body
	body := json.RawMessage(`{"model":"gpt-4o","input":[],"previous_response_id":"resp_explicit_456"}`)
	call := fantasy.Call{}
	result := ws.applyWSOptions(body, call)

	var resultMap map[string]json.RawMessage
	if err := json.Unmarshal(result, &resultMap); err != nil {
		t.Fatalf("unmarshal result: %v", err)
	}

	var prevID string
	json.Unmarshal(resultMap["previous_response_id"], &prevID)
	if prevID != "resp_explicit_456" {
		t.Errorf("expected explicit ID resp_explicit_456, got %s", prevID)
	}
}

func TestWSTransport_FallbackToHTTP(t *testing.T) {
	// Create a provider with WebSocket enabled but no server running
	provider, err := New(
		WithAPIKey("test-key"),
		WithBaseURL("https://localhost:1"),
		WithUseResponsesAPI(),
		WithWebSocket(),
	)
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}

	model, err := provider.LanguageModel(context.Background(), "gpt-4o")
	if err != nil {
		t.Fatalf("language model: %v", err)
	}

	// The model should be a responsesLanguageModel with wsTransport set
	rlm, ok := model.(responsesLanguageModel)
	if !ok {
		t.Fatal("expected responsesLanguageModel")
	}
	if rlm.wsTransport == nil {
		t.Fatal("expected wsTransport to be set")
	}
}

func TestWSURL(t *testing.T) {
	tests := []struct {
		baseURL  string
		expected string
	}{
		{"https://api.openai.com/v1", "wss://api.openai.com/v1/responses"},
		{"https://custom.api.com/v1", "wss://custom.api.com/v1/responses"},
		{"http://localhost:8080/v1", "ws://localhost:8080/v1/responses"},
		{"https://api.openai.com/v1/", "wss://api.openai.com/v1/responses"},
	}

	for _, tt := range tests {
		ws := newWSTransport(tt.baseURL, "key", nil)
		ws.baseURL = tt.baseURL
		got := ws.wsURL()
		if got != tt.expected {
			t.Errorf("wsURL(%s) = %s, want %s", tt.baseURL, got, tt.expected)
		}
	}
}
