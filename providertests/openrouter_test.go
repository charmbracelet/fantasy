package providertests

import (
	"net/http"
	"os"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openrouter"
	"github.com/stretchr/testify/require"
)

var openrouterTestModels = []testModel{
	{"anthropic", "anthropic/claude-haiku-4.5", false},
	{"openai", "openai/gpt-5.4-nano", false},
	{"gemini", "google/gemini-3.1-flash-lite-preview", false},
	{"grok", "x-ai/grok-4.1-fast", false},
	{"glm", "z-ai/glm-5-turbo", false},
	{"kimi", "moonshotai/kimi-k2.5", false},
}

func TestOpenRouterCommon(t *testing.T) {
	var pairs []builderPair
	for _, m := range openrouterTestModels {
		pairs = append(pairs, builderPair{m.name, openrouterBuilder(m.model), nil, nil})
	}
	testCommon(t, pairs)
}

func TestOpenRouterCommonWithAnthropicCache(t *testing.T) {
	testCommon(t, []builderPair{
		{"anthropic", openrouterBuilder("anthropic/claude-haiku-4.5"), nil, addAnthropicCaching},
	})
}

func TestOpenRouterThinking(t *testing.T) {
	opts := fantasy.ProviderOptions{
		openrouter.Name: &openrouter.ProviderOptions{
			Reasoning: &openrouter.ReasoningOptions{
				Effort: openrouter.ReasoningEffortOption(openrouter.ReasoningEffortMedium),
			},
		},
	}

	var pairs []builderPair
	for _, m := range openrouterTestModels {
		if !m.reasoning {
			continue
		}
		pairs = append(pairs, builderPair{m.name, openrouterBuilder(m.model), opts, nil})
	}
	testThinking(t, pairs, testOpenrouterThinking)

	// test anthropic signature
	testThinking(t, []builderPair{
		{"anthropic", openrouterBuilder("anthropic/claude-haiku-4.5"), opts, nil},
	}, requireThinkingWithSignatures)
}

func testOpenrouterThinking(t *testing.T, result *fantasy.AgentResult) {
	reasoningContentCount := 0
	for _, step := range result.Steps {
		for _, msg := range step.Messages {
			for _, content := range msg.Content {
				if content.GetType() == fantasy.ContentTypeReasoning {
					reasoningContentCount += 1
				}
			}
		}
	}
	require.Greater(t, reasoningContentCount, 0)
}

func openrouterBuilder(model string) builderFunc {
	return vcrProviderBuilder(model,
		openrouter.New,
		openrouter.WithAPIKey,
		func(client *http.Client) openrouter.Option { return openrouter.WithHTTPClient(client) },
		os.Getenv("FANTASY_OPENROUTER_API_KEY"),
	)
}
