package prompttest_test

import (
	"testing"

	"charm.land/fantasy/providers/internal/prompttest"
	"charm.land/fantasy/providers/openai"
	"charm.land/fantasy/providers/openaicompat"
	"charm.land/fantasy/providers/openrouter"
	"charm.land/fantasy/providers/vercel"
)

// TestPrompt records every chat-completions converter's output for the shared
// corpus. One file per case holds all four providers, so a divergence between
// them is a difference between adjacent keys in the same file.
//
// Regenerate with: go test ./providers/internal/prompttest -update
func TestPrompt(t *testing.T) {
	t.Parallel()

	prompttest.Golden(t, []prompttest.Converter{
		{Name: "openai", Convert: openai.DefaultToPrompt},
		{Name: "openaicompat", Convert: openaicompat.ToPromptFunc},
		{Name: "openrouter", Convert: openrouter.ToPrompt},
		{Name: "vercel", Convert: vercel.ToPrompt},
	})
}
