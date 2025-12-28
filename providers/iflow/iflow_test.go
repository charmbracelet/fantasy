package iflow

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNew(t *testing.T) {
	tests := []struct {
		name    string
		opts    []Option
		wantErr bool
	}{
		{
			name:    "default options",
			opts:    []Option{},
			wantErr: false,
		},
		{
			name: "with custom base URL",
			opts: []Option{
				WithBaseURL("https://custom.iflow.com/v1"),
			},
			wantErr: false,
		},
		{
			name: "with API key",
			opts: []Option{
				WithAPIKey("test-api-key"),
			},
			wantErr: false,
		},
		{
			name: "with headers",
			opts: []Option{
				WithHeaders(map[string]string{
					"X-Custom-Header": "value",
				}),
			},
			wantErr: false,
		},
		{
			name: "with HTTP client",
			opts: []Option{
				WithHTTPClient(&http.Client{}),
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider, err := New(tt.opts...)
			if (err != nil) != tt.wantErr {
				t.Errorf("New() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if provider == nil {
					t.Error("New() returned nil provider")
				}
			}
		})
	}
}

func TestName(t *testing.T) {
	if Name != "iflow" {
		t.Errorf("Expected Name to be 'iflow', got '%s'", Name)
	}
}

func TestIFlowTransport(t *testing.T) {
	// Create a mock server to receive the request
	var capturedBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(body, &capturedBody); err != nil {
			t.Fatal(err)
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"hello"}}]}`))
	}))
	defer server.Close()

	// Create the transport
	transport := &iflowTransport{
		base: http.DefaultTransport,
	}

	// Create a request with max_tokens and max_token
	payload := map[string]any{
		"model":      "test-model",
		"messages":   []any{map[string]any{"role": "user", "content": "hi"}},
		"max_tokens": 100,
		"max_token":  100,
	}
	body, err := json.Marshal(payload)
	require.NoError(t, err)

	req, err := http.NewRequest(http.MethodPost, server.URL, bytes.NewReader(body))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")

	// Send the request through the transport
	client := &http.Client{Transport: transport}
	resp, err := client.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()

	// Verify that max_tokens and max_token were removed
	assert.NotContains(t, capturedBody, "max_tokens")
	assert.NotContains(t, capturedBody, "max_token")
	assert.Equal(t, "test-model", capturedBody["model"])
}

func TestProviderImplementsInterface(t *testing.T) {
	provider, err := New()
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}

	// Verify it implements the Provider interface
	var _ fantasy.Provider = provider
}