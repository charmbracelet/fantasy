package openai

import (
	"testing"

	"charm.land/fantasy/providers/internal/prompttest"
)

// TestToPromptGolden records DefaultToPrompt's output for the shared corpus.
// See providers/internal/prompttest for why the corpus is shared.
func TestToPromptGolden(t *testing.T) {
	t.Parallel()

	for _, tc := range prompttest.Cases() {
		t.Run(tc.Name, func(t *testing.T) {
			t.Parallel()
			messages, warnings := DefaultToPrompt(tc.Prompt, "openai", tc.Model)
			prompttest.Golden(t, "testdata/prompt", tc.Name, messages, warnings)
		})
	}
}
