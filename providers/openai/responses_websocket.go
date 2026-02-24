package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"charm.land/fantasy"
	"github.com/gorilla/websocket"
)

// wsReconnectThreshold is how close to the 60-minute connection timeout we allow before reconnecting.
const wsReconnectThreshold = 55 * time.Minute

// wsTransport manages a persistent WebSocket connection to the OpenAI Responses API.
type wsTransport struct {
	mu             sync.Mutex
	conn           *websocket.Conn
	connectedAt    time.Time
	baseURL        string
	apiKey         string
	headers        map[string]string
	lastResponseID string
	lastInputLen   int // number of input items sent in the last successful request
}

// newWSTransport creates a new WebSocket transport for the OpenAI Responses API.
func newWSTransport(baseURL, apiKey string, headers map[string]string) *wsTransport {
	return &wsTransport{
		baseURL: baseURL,
		apiKey:  apiKey,
		headers: headers,
	}
}

// wsURL converts the base URL to a WebSocket URL.
func (ws *wsTransport) wsURL() string {
	url := ws.baseURL
	url = strings.Replace(url, "https://", "wss://", 1)
	url = strings.Replace(url, "http://", "ws://", 1)
	url = strings.TrimSuffix(url, "/")
	// Remove trailing /v1 if present since we add /v1/responses
	url = strings.TrimSuffix(url, "/v1")
	return url + "/v1/responses"
}

// connect establishes a WebSocket connection.
func (ws *wsTransport) connect(ctx context.Context) error {
	header := http.Header{}
	header.Set("Authorization", "Bearer "+ws.apiKey)
	for key, value := range ws.headers {
		header.Set(key, value)
	}

	dialer := websocket.Dialer{
		HandshakeTimeout: 30 * time.Second,
	}

	conn, _, err := dialer.DialContext(ctx, ws.wsURL(), header)
	if err != nil {
		return fmt.Errorf("websocket connect: %w", err)
	}

	ws.conn = conn
	ws.connectedAt = time.Now()
	return nil
}

// ensureConnected connects if not connected or reconnects if approaching the 60-minute limit.
func (ws *wsTransport) ensureConnected(ctx context.Context) error {
	if ws.conn != nil && time.Since(ws.connectedAt) < wsReconnectThreshold {
		return nil
	}

	if ws.conn != nil {
		ws.conn.Close()
		ws.conn = nil
	}

	return ws.connect(ctx)
}

// Close closes the WebSocket connection.
func (ws *wsTransport) Close() error {
	ws.mu.Lock()
	defer ws.mu.Unlock()

	if ws.conn != nil {
		err := ws.conn.Close()
		ws.conn = nil
		return err
	}
	return nil
}

// responseCreateEvent wraps the response params in a WebSocket event envelope.
type responseCreateEvent struct {
	Type string          `json:"type"`
	Body json.RawMessage `json:"-"`
}

// MarshalJSON implements custom marshaling to flatten the body into the event.
func (e responseCreateEvent) MarshalJSON() ([]byte, error) {
	// Start with the body fields and add the type field
	var bodyMap map[string]json.RawMessage
	if err := json.Unmarshal(e.Body, &bodyMap); err != nil {
		return nil, fmt.Errorf("unmarshal body: %w", err)
	}
	typeBytes, err := json.Marshal(e.Type)
	if err != nil {
		return nil, fmt.Errorf("marshal type: %w", err)
	}
	bodyMap["type"] = typeBytes
	return json.Marshal(bodyMap)
}

// wsServerEvent represents a server-sent event from the WebSocket connection.
type wsServerEvent struct {
	Type string          `json:"type"`
	Raw  json.RawMessage `json:"-"`
}

// sendResponseCreate sends a response.create event and returns a channel of raw server events.
// The caller must hold ws.mu.
func (ws *wsTransport) sendResponseCreate(ctx context.Context, body json.RawMessage) (chan wsServerEvent, error) {
	if err := ws.ensureConnected(ctx); err != nil {
		return nil, err
	}

	event := responseCreateEvent{
		Type: "response.create",
		Body: body,
	}

	data, err := event.MarshalJSON()
	if err != nil {
		return nil, fmt.Errorf("marshal response.create: %w", err)
	}

	if err := ws.conn.WriteMessage(websocket.TextMessage, data); err != nil {
		return nil, fmt.Errorf("websocket write: %w", err)
	}

	events := make(chan wsServerEvent, 64)

	conn := ws.conn
	go func() {
		defer close(events)

		// Set a read deadline when the context is cancelled to unblock ReadMessage.
		done := make(chan struct{})
		defer close(done)
		go func() {
			select {
			case <-ctx.Done():
				conn.SetReadDeadline(time.Now())
			case <-done:
			}
		}()

		for {
			_, message, err := conn.ReadMessage()
			if err != nil {
				// Don't emit an error event if the context was cancelled.
				if ctx.Err() != nil {
					return
				}
				events <- wsServerEvent{
					Type: "error",
					Raw:  mustMarshal(map[string]string{"type": "error", "code": "websocket_read_error", "message": err.Error()}),
				}
				return
			}

			var evt wsServerEvent
			if err := json.Unmarshal(message, &evt); err != nil {
				continue
			}
			evt.Raw = message

			events <- evt

			// Terminal events
			if evt.Type == "response.completed" || evt.Type == "response.incomplete" || evt.Type == "response.failed" {
				return
			}
			if evt.Type == "error" {
				return
			}
		}
	}()

	return events, nil
}

func mustMarshal(v any) json.RawMessage {
	data, _ := json.Marshal(v)
	return data
}

// extractIncrementalInput returns only the new input items that the server hasn't
// seen yet. When chaining with previous_response_id, the server already has the
// prior context, so we only send function_call_output items and new user messages.
// Items of type "function_call" are filtered out because the server generated those
// as part of its own response output.
func (ws *wsTransport) extractIncrementalInput(fullInput json.RawMessage) (json.RawMessage, int) {
	var items []json.RawMessage
	if err := json.Unmarshal(fullInput, &items); err != nil {
		return fullInput, 0
	}

	fullLen := len(items)

	if ws.lastResponseID == "" || ws.lastInputLen == 0 {
		return fullInput, fullLen
	}

	if len(items) <= ws.lastInputLen {
		return fullInput, fullLen
	}

	// Take only items appended since the last request.
	newItems := items[ws.lastInputLen:]

	// Filter out function_call items — the server already has these from its
	// own response output; sending them again would be redundant.
	var incremental []json.RawMessage
	for _, item := range newItems {
		var parsed map[string]json.RawMessage
		if err := json.Unmarshal(item, &parsed); err != nil {
			incremental = append(incremental, item)
			continue
		}
		if typeField, ok := parsed["type"]; ok {
			var itemType string
			if err := json.Unmarshal(typeField, &itemType); err == nil && itemType == "function_call" {
				continue
			}
		}
		incremental = append(incremental, item)
	}

	result, err := json.Marshal(incremental)
	if err != nil {
		return fullInput, fullLen
	}
	return result, fullLen
}

// applyWSOptions modifies the marshaled params JSON to add WebSocket-specific fields
// like previous_response_id (from transport state) and generate (for warmup).
// It returns the modified body and the full input item count (before any trimming)
// so callers can update lastInputLen after a successful response.
func (ws *wsTransport) applyWSOptions(body json.RawMessage, call fantasy.Call) (json.RawMessage, int) {
	var bodyMap map[string]json.RawMessage
	if err := json.Unmarshal(body, &bodyMap); err != nil {
		return body, 0
	}

	var fullInputLen int

	// Auto-chain with previous_response_id from transport state if not explicitly set
	usingPrevID := false
	if _, hasPrevID := bodyMap["previous_response_id"]; !hasPrevID && ws.lastResponseID != "" {
		prevIDBytes, _ := json.Marshal(ws.lastResponseID)
		bodyMap["previous_response_id"] = prevIDBytes
		usingPrevID = true
	} else if _, hasPrevID := bodyMap["previous_response_id"]; hasPrevID {
		usingPrevID = true
	}

	// When chaining, send only incremental input items.
	if inputField, hasInput := bodyMap["input"]; hasInput {
		if usingPrevID && ws.lastInputLen > 0 {
			bodyMap["input"], fullInputLen = ws.extractIncrementalInput(inputField)
		} else {
			// Count full input items for tracking.
			var items []json.RawMessage
			if err := json.Unmarshal(inputField, &items); err == nil {
				fullInputLen = len(items)
			}
		}
	}

	// Handle GenerateWarmup from provider options
	var openaiOptions *ResponsesProviderOptions
	if opts, ok := call.ProviderOptions[Name]; ok {
		if typedOpts, ok := opts.(*ResponsesProviderOptions); ok {
			openaiOptions = typedOpts
		}
	}
	if openaiOptions != nil && openaiOptions.GenerateWarmup != nil && *openaiOptions.GenerateWarmup {
		bodyMap["generate"] = json.RawMessage("false")
	}

	result, err := json.Marshal(bodyMap)
	if err != nil {
		return body, fullInputLen
	}
	return result, fullInputLen
}
