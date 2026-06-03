package anthropic

import (
	"regexp"
	"strings"
)

// anthropicDocumentTitleDisallowed matches every rune that Anthropic's
// document title field rejects. The allowlist is alphanumerics, whitespace,
// hyphens, parentheses, and square brackets. Anything else is replaced
// with a space; consecutive whitespace is then collapsed.
//
// Anthropic returns "The document file name can only contain alphanumeric
// characters, whitespace characters, hyphens, parentheses, and square
// brackets." when the title falls outside this set.
var anthropicDocumentTitleDisallowed = regexp.MustCompile(`[^a-zA-Z0-9\s\-()\[\]]`)

// anthropicDocumentTitleWhitespace collapses runs of whitespace.
var anthropicDocumentTitleWhitespace = regexp.MustCompile(`\s+`)

// sanitizeAnthropicDocumentTitle adapts a filename for use as the title
// field on an Anthropic DocumentBlock. Disallowed characters are replaced
// with spaces, runs of whitespace are collapsed, and the result is trimmed.
// Empty input (or input that sanitizes to empty) returns "Document" so the
// model always has a stable handle for the attachment.
func sanitizeAnthropicDocumentTitle(filename string) string {
	replaced := anthropicDocumentTitleDisallowed.ReplaceAllString(filename, " ")
	collapsed := strings.TrimSpace(anthropicDocumentTitleWhitespace.ReplaceAllString(replaced, " "))
	if collapsed == "" {
		return "Document"
	}
	return collapsed
}
