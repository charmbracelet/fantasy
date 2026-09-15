package openrouter

import (
	"testing"

	"charm.land/fantasy/providers/internal/prompttest"
)

// TestToPromptGolden records languageModelToPrompt's output for the shared corpus.
// See providers/internal/prompttest for why the corpus is shared.
func TestToPromptGolden(t *testing.T) {
	t.Parallel()

	for _, tc := range prompttest.Cases() {
		t.Run(tc.Name, func(t *testing.T) {
			t.Parallel()
			messages, warnings := languageModelToPrompt(tc.Prompt, "openrouter", tc.Model)
			prompttest.Golden(t, "testdata/prompt", tc.Name, messages, warnings)
		})
	}
}
