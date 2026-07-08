package anthropic

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestSanitizeAnthropicDocumentTitle(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "empty falls back to Document",
			input: "",
			want:  "Document",
		},
		{
			name:  "all disallowed falls back to Document",
			input: "...",
			want:  "Document",
		},
		{
			name:  "whitespace only falls back to Document",
			input: "   \t\n",
			want:  "Document",
		},
		{
			name:  "alphanumeric is preserved",
			input: "report 2026",
			want:  "report 2026",
		},
		{
			name:  "dots and underscores become spaces",
			input: "quarterly_report.v1.pdf",
			want:  "quarterly report v1 pdf",
		},
		{
			name:  "preserves hyphens, parentheses, square brackets",
			input: "draft-1 (final) [v2].pdf",
			want:  "draft-1 (final) [v2] pdf",
		},
		{
			name:  "collapses runs of whitespace",
			input: "name  with    spaces",
			want:  "name with spaces",
		},
		{
			name:  "trims leading and trailing whitespace",
			input: "  leading and trailing  ",
			want:  "leading and trailing",
		},
		{
			name:  "leading dots collapse to single space then trim",
			input: "..hidden.txt",
			want:  "hidden txt",
		},
		{
			name:  "non-ascii letters are not allowlisted",
			input: "résumé.pdf",
			want:  "r sum pdf",
		},
		{
			name:  "production failure example is sanitized",
			input: "D19910350Lj.pdf",
			want:  "D19910350Lj pdf",
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			require.Equal(t, tc.want, sanitizeAnthropicDocumentTitle(tc.input))
		})
	}
}
