package main

// This example demonstrates provider-defined web search tools.
// It auto-selects the provider based on which API key is set:
//
//	ANTHROPIC_API_KEY → Anthropic (Claude)
//	OPENAI_API_KEY    → OpenAI (GPT, Responses API)
//
// Provider tools are executed server-side by the model provider,
// so there is no local tool implementation needed.

import (
	"context"
	"fmt"
	"os"
	"strings"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/anthropic"
	"charm.land/fantasy/providers/openai"
)

func main() {
	ctx := context.Background()

	var (
		model     fantasy.LanguageModel
		webSearch fantasy.ProviderDefinedTool
		err       error
	)

	switch {
	case os.Getenv("ANTHROPIC_API_KEY") != "":
		p, _ := anthropic.New(anthropic.WithAPIKey(os.Getenv("ANTHROPIC_API_KEY")))
		model, err = p.LanguageModel(ctx, "claude-sonnet-4-20250514")
		webSearch = anthropic.WebSearchTool(nil)

	case os.Getenv("OPENAI_API_KEY") != "":
		p, _ := openai.New(
			openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
			openai.WithUseResponsesAPI(),
		)
		model, err = p.LanguageModel(ctx, "gpt-5.4")
		webSearch = openai.WebSearchTool(nil)

	default:
		fmt.Fprintln(os.Stderr, "Set ANTHROPIC_API_KEY or OPENAI_API_KEY")
		os.Exit(1)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	agent := fantasy.NewAgent(model,
		fantasy.WithProviderDefinedTools(webSearch),
	)

	result, err := agent.Generate(ctx, fantasy.AgentCall{
		Prompt: "What is the population of Tokyo? Cite your source.",
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	var text strings.Builder
	for _, c := range result.Response.Content {
		if tc, ok := c.(fantasy.TextContent); ok {
			text.WriteString(tc.Text)
		}
	}
	fmt.Println(text.String())

	for _, source := range result.Response.Content.Sources() {
		fmt.Printf("Source: %s — %s\n", source.Title, source.URL)
	}
}
