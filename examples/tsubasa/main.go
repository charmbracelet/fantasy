package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openaicompat"
)

// API access and live model qualification are pending. This example requires an
// enabled Tsubasa account and uses the text-only Chat Completions endpoint.
func main() {
	key := os.Getenv("TSUBASA_API_KEY")
	if key == "" {
		log.Fatal("Set TSUBASA_API_KEY before running this example")
	}
	provider, err := openaicompat.New(
		openaicompat.WithName("tsubasa"),
		openaicompat.WithBaseURL("https://api.tsubasa.sh/v1"),
		openaicompat.WithAPIKey(key),
	)
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	model, err := provider.LanguageModel(ctx, "tsubasa-pro")
	if err != nil {
		log.Fatal(err)
	}
	result, err := fantasy.NewAgent(model).Generate(ctx, fantasy.AgentCall{
		Prompt: "Explain binary search in three sentences.",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Response.Content.Text())
}
