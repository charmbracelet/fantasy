package main

// This example demonstrates how to get structured, type-safe outputs from an
// LLM. Here we're getting a recipe with validated fields that we can use
// directly in our code.

import (
	"context"
	"fmt"
	"os"

	"charm.land/fantasy"
	"charm.land/fantasy/object"
	"charm.land/fantasy/providers/openai"
)

// Here's what we want the LLM to fill out. The struct tags tell the model
// what each field is for.
type Recipe struct {
	Name        string   `json:"name" description:"The name of the recipe"`
	Ingredients []string `json:"ingredients" description:"List of ingredients needed"`
	Steps       []string `json:"steps" description:"Step-by-step cooking instructions"`
	PrepTime    int      `json:"prep_time" description:"Preparation time in minutes"`
}

func main() {
	// We'll use OpenAI for this one.
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("Please set OPENAI_API_KEY environment variable")
		os.Exit(1)
	}

	ctx := context.Background()

	// Set up the provider.
	provider, err := openai.New(openai.WithAPIKey(apiKey))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Whoops: %v\n", err)
		os.Exit(1)
	}

	// Pick the model.
	model, err := provider.LanguageModel(ctx, "gpt-4o-mini")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Dang: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("\n🍪 Generating a recipe...")

	// Ask for a structured recipe. The model will return a Recipe struct
	// that's been validated against our schema.
	result, err := object.Generate[Recipe](ctx, model, fantasy.ObjectCall{
		Prompt: fantasy.Prompt{
			fantasy.NewUserMessage("Give me a recipe for chocolate chip cookies"),
		},
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Oof: %v\n", err)
		os.Exit(1)
	}

	// Now we have a type-safe Recipe we can use directly.
	fmt.Printf("Recipe: %s\n", result.Object.Name)
	fmt.Printf("Prep time: %d minutes\n", result.Object.PrepTime)
	fmt.Printf("Ingredients: %d\n", len(result.Object.Ingredients))
	for i, ing := range result.Object.Ingredients {
		fmt.Printf("  %d. %s\n", i+1, ing)
	}
	fmt.Printf("Steps: %d\n", len(result.Object.Steps))
	for i, step := range result.Object.Steps {
		fmt.Printf("  %d. %s\n", i+1, step)
	}
	fmt.Printf("\nTokens used: %d\n\n", result.Usage.TotalTokens)

	// Want to see progressive updates as the object builds? Use streaming!
	fmt.Println("🌊 Now let's try streaming...")

	stream, err := object.Stream[Recipe](ctx, model, fantasy.ObjectCall{
		Prompt: fantasy.Prompt{
			fantasy.NewUserMessage("Give me a recipe for banana bread"),
		},
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Oof: %v\n", err)
		os.Exit(1)
	}

	// Watch the recipe build in real-time!
	updateCount := 0
	for partial := range stream.PartialObjectStream() {
		updateCount++
		fmt.Printf("  Update %d: %s (%d ingredients, %d steps)\n",
			updateCount, partial.Name, len(partial.Ingredients), len(partial.Steps))
	}

	fmt.Println()
}
