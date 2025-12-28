package xiaomi

import (
	"testing"

	"charm.land/fantasy"
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
				WithBaseURL("https://custom.xiaomi.com/v1"),
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
			name: "with extra body",
			opts: []Option{
				WithExtraBody(map[string]any{
					"thinking": map[string]any{
						"type": "enabled",
					},
				}),
			},
			wantErr: false,
		},
		{
			name: "with all options",
			opts: []Option{
				WithBaseURL("https://custom.xiaomi.com/v1"),
				WithAPIKey("test-api-key"),
				WithHeaders(map[string]string{
					"X-Custom-Header": "value",
				}),
				WithExtraBody(map[string]any{
					"thinking": map[string]any{
						"type": "enabled",
					},
				}),
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
	if Name != "xiaomi" {
		t.Errorf("Expected Name to be 'xiaomi', got '%s'", Name)
	}
}

func TestProviderImplementsInterface(t *testing.T) {
	provider, err := New()
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}
	
	// Verify it implements the Provider interface
	var _ fantasy.Provider = provider
}