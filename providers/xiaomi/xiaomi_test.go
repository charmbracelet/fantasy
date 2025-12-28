package xiaomi

import (
	"encoding/json"
	"testing"

	"charm.land/fantasy"
)

func TestExtractXiaomiToolCalls(t *testing.T) {
	tests := []struct {
		name           string
		content        string
		expectedCalls  []xiaomiToolCall
		expectedRemain string
		wantErr        bool
	}{
		{
			name:    "single tool call",
			content: `<function=editor><parameter=command>view</parameter><parameter=file_path>/path/to/file</parameter></function>`,
			expectedCalls: []xiaomiToolCall{
				{
					name:      "editor",
					arguments: `{"command":"view","file_path":"/path/to/file"}`,
				},
			},
			expectedRemain: "",
			wantErr:        false,
		},
		{
			name:    "multiple tool calls",
			content: `<function=editor><parameter=command>view</parameter></function><function=write><parameter=content>hello</parameter></function>`,
			expectedCalls: []xiaomiToolCall{
				{
					name:      "editor",
					arguments: `{"command":"view"}`,
				},
				{
					name:      "write",
					arguments: `{"content":"hello"}`,
				},
			},
			expectedRemain: "",
			wantErr:        false,
		},
		{
			name:           "incomplete tool call",
			content:        `<function=editor><parameter=command>view</parameter>`,
			expectedCalls:  []xiaomiToolCall{},
			expectedRemain: `<function=editor><parameter=command>view</parameter>`,
			wantErr:        false,
		},
		{
			name:    "tool call with special characters",
			content: `<function=test><parameter=path>/path/with spaces</parameter></function>`,
			expectedCalls: []xiaomiToolCall{
				{
					name:      "test",
					arguments: `{"path":"/path/with spaces"}`,
				},
			},
			expectedRemain: "",
			wantErr:        false,
		},
		{
			name:    "tool call with newlines (actual Xiaomi format)",
			content: "\n<function=editor>\n<parameter=command>view</parameter>\n<parameter=file_path>/Users/aero/Documents/charm/catwalk/internal/providers/providers.go</parameter>\n</function>\n",
			expectedCalls: []xiaomiToolCall{
				{
					name:      "editor",
					arguments: `{"command":"view","file_path":"/Users/aero/Documents/charm/catwalk/internal/providers/providers.go"}`,
				},
			},
			expectedRemain: "\n",
			wantErr:        false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls, remain, err := extractXiaomiToolCalls(tt.content)
			if (err != nil) != tt.wantErr {
				t.Errorf("extractXiaomiToolCalls() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if len(calls) != len(tt.expectedCalls) {
				t.Errorf("extractXiaomiToolCalls() got %d calls, want %d", len(calls), len(tt.expectedCalls))
				return
			}
			for i, call := range calls {
				if call.name != tt.expectedCalls[i].name {
					t.Errorf("extractXiaomiToolCalls() call[%d].name = %v, want %v", i, call.name, tt.expectedCalls[i].name)
				}
				if call.arguments != tt.expectedCalls[i].arguments {
					t.Errorf("extractXiaomiToolCalls() call[%d].arguments = %v, want %v", i, call.arguments, tt.expectedCalls[i].arguments)
				}
			}
			if remain != tt.expectedRemain {
				t.Errorf("extractXiaomiToolCalls() remain = %v, want %v", remain, tt.expectedRemain)
			}
		})
	}
}

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
			name: "with thinking enabled",
			opts: []Option{
				WithThinking(true),
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

func TestToolNameExtraction(t *testing.T) {
	// Test that wrapper functions have their command parameter extracted
	testCases := []struct {
		name         string
		xml          string
		expectedTool string
	}{
		{
			name:         "editor wrapper with view command",
			xml:          `<function=editor><parameter=command>view</parameter><parameter=file_path>/path/to/file</parameter></function>`,
			expectedTool: "view",
		},
		{
			name:         "editor wrapper with write command",
			xml:          `<function=editor><parameter=command>write</parameter><parameter=file_path>/path/to/file</parameter><parameter=content>hello</parameter></function>`,
			expectedTool: "write",
		},
		{
			name:         "bash wrapper with ls command",
			xml:          `<function=bash><parameter=command>ls</parameter><parameter=path>/tmp</parameter></function>`,
			expectedTool: "ls",
		},
		{
			name:         "non-wrapper function",
			xml:          `<function=write><parameter=file_path>/path/to/file</parameter><parameter=content>hello</parameter></function>`,
			expectedTool: "write",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Extract tool calls
			toolCalls, _, err := extractXiaomiToolCalls(tc.xml)
			if err != nil {
				t.Fatalf("extractXiaomiToolCalls() error = %v", err)
			}
			if len(toolCalls) != 1 {
				t.Fatalf("extractXiaomiToolCalls() got %d calls, want 1", len(toolCalls))
			}

			// Simulate the tool name extraction logic from parseXiaomiToolCalls
			parsedTC := toolCalls[0]
			toolName := parsedTC.name

			// Parse arguments to extract command if present
			var argsMap map[string]string
			if err := json.Unmarshal([]byte(parsedTC.arguments), &argsMap); err == nil {
				if command, ok := argsMap["command"]; ok && (parsedTC.name == "editor" || parsedTC.name == "bash" || parsedTC.name == "agent") {
					// Use command parameter as tool name for wrapper functions
					toolName = command
					// Remove command from arguments
					delete(argsMap, "command")
				}
			}

			// Verify the tool name was extracted correctly
			if toolName != tc.expectedTool {
				t.Errorf("Expected tool name '%s', got '%s'", tc.expectedTool, toolName)
			}
		})
	}
}

func TestWrapYieldToExtractToolName(t *testing.T) {
	// Test that the yield wrapper correctly extracts tool names from wrapper functions
	testCases := []struct {
		name           string
		inputToolCall  fantasy.StreamPart
		expectedName   string
		expectedInput  string
	}{
		{
			name: "editor wrapper with view command",
			inputToolCall: fantasy.StreamPart{
				Type:          fantasy.StreamPartTypeToolCall,
				ID:            "test_id",
				ToolCallName:  "editor",
				ToolCallInput: `{"command":"view","file_path":"/path/to/file"}`,
			},
			expectedName:  "view",
			expectedInput: `{"file_path":"/path/to/file"}`,
		},
		{
			name: "bash wrapper with ls command",
			inputToolCall: fantasy.StreamPart{
				Type:          fantasy.StreamPartTypeToolCall,
				ID:            "test_id",
				ToolCallName:  "bash",
				ToolCallInput: `{"command":"ls","path":"/tmp"}`,
			},
			expectedName:  "ls",
			expectedInput: `{"path":"/tmp"}`,
		},
		{
			name: "non-wrapper function",
			inputToolCall: fantasy.StreamPart{
				Type:          fantasy.StreamPartTypeToolCall,
				ID:            "test_id",
				ToolCallName:  "write",
				ToolCallInput: `{"file_path":"/path/to/file","content":"hello"}`,
			},
			expectedName:  "write",
			expectedInput: `{"file_path":"/path/to/file","content":"hello"}`,
		},
		{
			name: "non-tool-call event",
			inputToolCall: fantasy.StreamPart{
				Type:  fantasy.StreamPartTypeTextDelta,
				Delta: "hello",
			},
			expectedName:  "",
			expectedInput: "",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create a yield function that captures the modified stream part
			var captured fantasy.StreamPart
			yield := func(sp fantasy.StreamPart) bool {
				captured = sp
				return true
			}

			// Apply the wrapper
			wrapped := wrapYieldToExtractToolName(yield)
			wrapped(tc.inputToolCall)

			// Verify the tool name was extracted correctly
			if tc.expectedName != "" {
				if captured.ToolCallName != tc.expectedName {
					t.Errorf("Expected tool name '%s', got '%s'", tc.expectedName, captured.ToolCallName)
				}
				if captured.ToolCallInput != tc.expectedInput {
					t.Errorf("Expected tool input '%s', got '%s'", tc.expectedInput, captured.ToolCallInput)
				}
			}
		})
	}
}