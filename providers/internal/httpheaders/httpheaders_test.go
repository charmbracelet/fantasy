package httpheaders

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDefaultUserAgent(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		version string
		agent   string
		want    string
	}{
		{name: "no agent", version: "0.11.0", agent: "", want: "Charm Fantasy/0.11.0"},
		{name: "with agent", version: "0.11.0", agent: "Claude 4.6 Opus", want: "Charm Fantasy/0.11.0 (Claude 4.6 Opus)"},
		{name: "agent trimmed", version: "1.0.0", agent: "  spaces  ", want: "Charm Fantasy/1.0.0 (spaces)"},
		{name: "agent strips parens", version: "1.0.0", agent: "foo(bar)", want: "Charm Fantasy/1.0.0 (foobar)"},
		{name: "agent strips control chars", version: "1.0.0", agent: "foo\x01bar", want: "Charm Fantasy/1.0.0 (foobar)"},
		{name: "agent capped at 64 chars", version: "1.0.0", agent: strings.Repeat("a", 100), want: "Charm Fantasy/1.0.0 (" + strings.Repeat("a", 64) + ")"},
		{name: "whitespace-only agent treated as empty", version: "1.0.0", agent: "   ", want: "Charm Fantasy/1.0.0"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := DefaultUserAgent(tt.version, tt.agent)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestResolveHeaders_Precedence(t *testing.T) {
	t.Parallel()

	t.Run("explicit UA wins over headers and default", func(t *testing.T) {
		t.Parallel()
		headers := map[string]string{"User-Agent": "from-headers"}
		got := ResolveHeaders(headers, "explicit-ua", "default-ua")
		assert.Equal(t, "explicit-ua", got["User-Agent"])
	})

	t.Run("header UA wins over default", func(t *testing.T) {
		t.Parallel()
		headers := map[string]string{"User-Agent": "from-headers"}
		got := ResolveHeaders(headers, "", "default-ua")
		assert.Equal(t, "from-headers", got["User-Agent"])
	})

	t.Run("default UA used when nothing else set", func(t *testing.T) {
		t.Parallel()
		got := ResolveHeaders(nil, "", "default-ua")
		assert.Equal(t, "default-ua", got["User-Agent"])
	})

	t.Run("explicit UA wins over case-insensitive header key", func(t *testing.T) {
		t.Parallel()
		headers := map[string]string{"user-agent": "from-headers"}
		got := ResolveHeaders(headers, "explicit-ua", "default-ua")
		assert.Equal(t, "explicit-ua", got["User-Agent"])
		_, hasLower := got["user-agent"]
		assert.False(t, hasLower, "old case-insensitive key should be removed")
	})

	t.Run("case-insensitive header key preserved when no explicit UA", func(t *testing.T) {
		t.Parallel()
		headers := map[string]string{"user-agent": "from-headers"}
		got := ResolveHeaders(headers, "", "default-ua")
		assert.Equal(t, "from-headers", got["user-agent"])
	})
}

func TestResolveHeaders_NoMutation(t *testing.T) {
	t.Parallel()

	original := map[string]string{"X-Custom": "value"}
	_ = ResolveHeaders(original, "explicit", "default")

	_, hasUA := original["User-Agent"]
	require.False(t, hasUA, "input map must not be mutated")
	assert.Equal(t, "value", original["X-Custom"])
}

func TestResolveHeaders_PreservesOtherHeaders(t *testing.T) {
	t.Parallel()

	headers := map[string]string{
		"X-Custom":      "custom-value",
		"Authorization": "Bearer token",
	}
	got := ResolveHeaders(headers, "", "Charm Fantasy/1.0.0")
	assert.Equal(t, "custom-value", got["X-Custom"])
	assert.Equal(t, "Bearer token", got["Authorization"])
	assert.Equal(t, "Charm Fantasy/1.0.0", got["User-Agent"])
}

func TestSanitizeAgent(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input string
		want  string
	}{
		{name: "normal text", input: "Claude 4.6 Opus", want: "Claude 4.6 Opus"},
		{name: "leading trailing spaces", input: "  spaced  ", want: "spaced"},
		{name: "parentheses removed", input: "agent(v2)", want: "agentv2"},
		{name: "control chars removed", input: "a\x00b\x1fc", want: "abc"},
		{name: "capped at 64", input: strings.Repeat("x", 100), want: strings.Repeat("x", 64)},
		{name: "multibyte runes capped at 64 chars", input: strings.Repeat("é", 100), want: strings.Repeat("é", 64)},
		{name: "empty stays empty", input: "", want: ""},
		{name: "only spaces", input: "   ", want: ""},
		{name: "trailing space after cap", input: strings.Repeat("a", 63) + " b", want: strings.Repeat("a", 63)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := sanitizeAgent(tt.input)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestResolveHeaders_DuplicateCaseInsensitiveKeys(t *testing.T) {
	t.Parallel()

	headers := map[string]string{
		"User-Agent": "canonical",
		"user-agent": "lowercase",
	}
	got := ResolveHeaders(headers, "explicit", "default")
	assert.Equal(t, "explicit", got["User-Agent"])
	_, hasLower := got["user-agent"]
	assert.False(t, hasLower, "all case-insensitive UA keys must be removed")
}
