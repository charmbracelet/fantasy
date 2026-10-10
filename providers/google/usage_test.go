package google

import (
	"testing"

	"google.golang.org/genai"
)

func TestMapUsage(t *testing.T) {
	t.Run("vertex shape: thoughts disjoint from candidates", func(t *testing.T) {
		lm := languageModel{providerOptions: options{backend: genai.BackendVertexAI}}
		u := lm.mapUsage(&genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     100,
			CandidatesTokenCount: 50,
			ThoughtsTokenCount:   30,
			TotalTokenCount:      180,
		})
		if u.OutputTokens != 80 {
			t.Fatalf("expected output 50+30=80, got %d", u.OutputTokens)
		}
		if u.ReasoningTokens != 30 {
			t.Fatalf("expected reasoning 30, got %d", u.ReasoningTokens)
		}
	})

	t.Run("ai studio shape: candidates include thoughts", func(t *testing.T) {
		lm := languageModel{providerOptions: options{backend: genai.BackendGeminiAPI}}
		u := lm.mapUsage(&genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     100,
			CandidatesTokenCount: 80, // 50 visible + 30 thoughts
			ThoughtsTokenCount:   30,
			TotalTokenCount:      180,
		})
		if u.OutputTokens != 80 {
			t.Fatalf("expected output unchanged at 80, got %d", u.OutputTokens)
		}
		if u.ReasoningTokens != 30 {
			t.Fatalf("expected reasoning 30, got %d", u.ReasoningTokens)
		}
	})

	t.Run("tool use prompt tokens count as prompt in the discriminant", func(t *testing.T) {
		lm := languageModel{providerOptions: options{backend: genai.BackendGeminiAPI}}
		u := lm.mapUsage(&genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:        100,
			ToolUsePromptTokenCount: 20,
			CandidatesTokenCount:    50,
			ThoughtsTokenCount:      30,
			TotalTokenCount:         200, // 120 + 50 + 30: disjoint
		})
		if u.OutputTokens != 80 {
			t.Fatalf("expected output 50+30=80, got %d", u.OutputTokens)
		}
	})

	t.Run("no thoughts leaves output untouched", func(t *testing.T) {
		lm := languageModel{providerOptions: options{backend: genai.BackendVertexAI}}
		u := lm.mapUsage(&genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     100,
			CandidatesTokenCount: 50,
			TotalTokenCount:      150,
		})
		if u.OutputTokens != 50 {
			t.Fatalf("expected output 50, got %d", u.OutputTokens)
		}
		if u.ReasoningTokens != 0 {
			t.Fatalf("expected reasoning 0, got %d", u.ReasoningTokens)
		}
	})

	t.Run("ambiguous totals fall back to the backend default", func(t *testing.T) {
		// Neither equality holds (partial or inconsistent metadata).
		metadata := &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     100,
			CandidatesTokenCount: 50,
			ThoughtsTokenCount:   30,
			TotalTokenCount:      177,
		}

		vertex := languageModel{providerOptions: options{backend: genai.BackendVertexAI}}
		if u := vertex.mapUsage(metadata); u.OutputTokens != 80 {
			t.Fatalf("vertex fallback: expected output 80, got %d", u.OutputTokens)
		}

		studio := languageModel{providerOptions: options{backend: genai.BackendGeminiAPI}}
		if u := studio.mapUsage(metadata); u.OutputTokens != 50 {
			t.Fatalf("ai studio fallback: expected output 50, got %d", u.OutputTokens)
		}
	})
}
