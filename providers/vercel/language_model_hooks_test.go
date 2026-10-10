package vercel

import "testing"

func TestIsVertexUpstream(t *testing.T) {
	tests := []struct {
		provider string
		want     bool
	}{
		{`"vertex"`, true},
		{`"google"`, false},
		{`"openai"`, false},
		{"", false},
		{"vertex", true}, // defensive: already-unquoted value
	}
	for _, tc := range tests {
		if got := isVertexUpstream(tc.provider); got != tc.want {
			t.Errorf("isVertexUpstream(%q) = %v, want %v", tc.provider, got, tc.want)
		}
	}
}
