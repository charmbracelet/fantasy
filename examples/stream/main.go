package main

// This example demonstrates how to hook into the various parts of a streaming
// tool call.

import (
	"context"
	"fmt"
	"math/rand/v2"
	"os"
	"strings"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openai"
)

const systemPrompt = `
You are moderately helpful assistant with a new puppy named Chuck. Chuck is
moody and ranges from very happy to very annoyed. He's pretty happy-go-lucky,
but new encounters make him pretty uncomfortable.

You despise emojis and never use them. Same with Markdown. Same with em-dashes.
You prefer "welp" to "well" when starting a sentence (that's just how you were
raised). You also don't use run-on sentences, including entering a comma where
there should be a period. You had a decent education and did well in elementary
school grammar. You grew up in the United States, specifically Kansas City,
Missouri.
`

// Input for a tool call. The LLM will look at the struct tags and fill out the
// values as necessary.
type dogInteraction struct {
	OtherDogName string `json:"dogName" description:"Name of the other dog. Just make something up. All the dogs are named after Japanese cars from the 80s."`
}

// Here's a tool call. In this case it's a set of random barks.
func letsBark(ctx context.Context, i dogInteraction, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
	var r fantasy.ToolResponse
	if rand.Float64() >= 0.5 {
		r.Content = randomBarks(1, 3)
	} else {
		r.Content = randomBarks(5, 10)
	}
	return r, nil
}

func main() {
	// We're going to use OpenAI.
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("Please set OPENAI_API_KEY environment variable")
		os.Exit(1)
	}

	// Make sure we have the API key we need.
	provider, err := openai.New(openai.WithAPIKey(apiKey))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating OpenAI provider: %v\n", err)
		os.Exit(1)
	}

	ctx := context.Background()

	// Choose the model.
	model, err := provider.LanguageModel(ctx, "gpt-5")
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	// Let's add a tool to our belt. A tool for dogs.
	barkTool := fantasy.NewAgentTool(
		"bark",
		"Have Chuck express his feelings by barking. A few barks means he's happy and many barks means he's not.",
		letsBark,
	)

	// Time to make the agent.
	agent := fantasy.NewAgent(
		model,
		fantasy.WithSystemPrompt(systemPrompt),
		fantasy.WithTools(barkTool),
	)

	// Alright, let's setup a streaming request!
	streamCall := fantasy.AgentStreamCall{
		// The prompt.
		Prompt: "Chuck just met a new dog at the park. Find out what he thinks of the dog. Make sure to thank Chuck afterwards.",

		// When we receive a chunk of streaming data.
		OnTextDelta: func(id, text string) error {
			_, fmtErr := fmt.Print(text)
			return fmtErr
		},

		// When tool calls are invoked.
		OnToolCall: func(toolCall fantasy.ToolCallContent) error {
			fmt.Printf("-> Invoking the %s tool with input %s", toolCall.ToolName, toolCall.Input)
			return nil
		},

		// When a tool call completes.
		OnToolResult: func(res fantasy.ToolResultContent) error {
			text, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentText](res.Result)
			if !ok {
				return fmt.Errorf("failed to cast result to text")
			}
			_, fmtErr := fmt.Printf("\n-> Using the %s tool: %s", res.ToolName, text.Text)
			return fmtErr
		},

		// When a step finishes, such as a tool call or a response from the
		// LLM.
		OnStepFinish: func(_ fantasy.StepResult) error {
			fmt.Print("\n-> Step completed\n")
			return nil
		},
	}

	fmt.Println("Generating...")

	// Finally, let's stream everything!
	_, err = agent.Stream(ctx, streamCall)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error generating response: %v\n", err)
		os.Exit(1)
	}
}

// Return a random number of barks between low and high.
func randomBarks(low, high int) string {
	const bark = "ruff"
	numBarks := low + rand.IntN(high-low+1)
	var barks strings.Builder
	for i := range numBarks {
		if i > 0 {
			barks.WriteString(" ")
		}
		barks.WriteString(bark)
	}
	return barks.String()
}
