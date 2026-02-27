package httpheaders

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDefaultUserAgent(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		version string
		want    string
	}{
		{name: "basic version", version: "0.11.0", want: "Charm Fantasy/0.11.0"},
		{name: "another version", version: "1.0.0", want: "Charm Fantasy/1.0.0"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := DefaultUserAgent(tt.version)
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

func TestCallUserAgent(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		callUA string
		wantUA string
		wantOK bool
	}{
		{name: "no override", callUA: "", wantUA: "", wantOK: false},
		{name: "explicit UA", callUA: "MyAgent/1.0", wantUA: "MyAgent/1.0", wantOK: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			ua, ok := CallUserAgent(tt.callUA)
			assert.Equal(t, tt.wantOK, ok)
			assert.Equal(t, tt.wantUA, ua)
		})
	}
}
