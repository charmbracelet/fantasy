package schema

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestParsePartialJSON(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		input     string
		wantState ParseState
		wantVal   any
	}{
		{
			name:      "empty",
			input:     "",
			wantState: ParseStateUndefined,
			wantVal:   nil,
		},
		{
			name:      "complete object",
			input:     `{"name":"Alice","age":30}`,
			wantState: ParseStateSuccessful,
			wantVal:   map[string]any{"name": "Alice", "age": float64(30)},
		},
		{
			name:      "truncated object",
			input:     `{"name":"John","age":25`,
			wantState: ParseStateRepaired,
			wantVal:   map[string]any{"name": "John", "age": float64(25)},
		},
		{
			name:      "truncated string inside array",
			input:     `{"items":["one","tw`,
			wantState: ParseStateRepaired,
			wantVal:   map[string]any{"items": []any{"one", "tw"}},
		},
		{
			name:      "partial literal",
			input:     `{"ok":tr`,
			wantState: ParseStateRepaired,
			wantVal:   map[string]any{"ok": true},
		},
		{
			name:      "markdown fence",
			input:     "```json\n{\"name\":\"Bob\"}\n```",
			wantState: ParseStateRepaired,
			wantVal:   map[string]any{"name": "Bob"},
		},
		{
			name:      "malformed single quotes fallback to jsonrepair",
			input:     `{'name': 'Alice', 'age': 30}`,
			wantState: ParseStateRepaired,
			wantVal:   map[string]any{"name": "Alice", "age": float64(30)},
		},
		{
			name:      "malformed unquoted keys fallback to jsonrepair",
			input:     `{name: "Alice", active: true}`,
			wantState: ParseStateRepaired,
			wantVal:   map[string]any{"name": "Alice", "active": true},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, state, err := ParsePartialJSON(tt.input)
			require.NoError(t, err)
			require.Equal(t, tt.wantState, state)
			require.Equal(t, tt.wantVal, got)
		})
	}
}

func TestStreamParserIncrementalAndFallback(t *testing.T) {
	t.Parallel()

	t.Run("chunked valid json", func(t *testing.T) {
		t.Parallel()
		full := `{"name":"Alice","tags":["go","simd"],"age":30}`
		for _, chunkSize := range []int{1, 3, 8} {
			p := NewStreamParser()
			var last any
			var lastState ParseState
			for i := 0; i < len(full); i += chunkSize {
				end := min(i+chunkSize, len(full))
				obj, state, err := p.Feed(full[i:end])
				if err == nil && (state == ParseStateSuccessful || state == ParseStateRepaired) {
					last = obj
					lastState = state
				}
			}
			require.Equal(t, ParseStateSuccessful, lastState)
			require.Equal(t, map[string]any{
				"name": "Alice",
				"tags": []any{"go", "simd"},
				"age":  float64(30),
			}, last)
			require.Equal(t, full, p.Raw())
		}
	})

	t.Run("chunked malformed json falls back cleanly", func(t *testing.T) {
		t.Parallel()
		chunks := []string{"{name: ", "'Alice'", ", age: 30}"}
		p := NewStreamParser()
		var last any
		for _, ch := range chunks {
			obj, state, err := p.Feed(ch)
			if err == nil && (state == ParseStateSuccessful || state == ParseStateRepaired) {
				last = obj
			}
		}
		require.Equal(t, map[string]any{"name": "Alice", "age": float64(30)}, last)
	})
}
