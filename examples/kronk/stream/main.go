package main

// This example demonstrates how to hook into the various parts of a streaming
// tool call.

import (
	"context"
	"fmt"
	"math/rand/v2"
	"os"
	"strings"
	"time"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/kronk"
)

const modelURL = "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q8_0.gguf"

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
	if err := run(); err != nil {
		fmt.Printf("\nERROR: %s\n", err)
		os.Exit(1)
	}
}

func run() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	// Create the provider with optional logging.
	provider, err := kronk.New(
		kronk.WithName("kronk"),
		kronk.WithLogger(kronk.FmtLogger),
	)
	if err != nil {
		return fmt.Errorf("unable to create provider: %w", err)
	}

	// Clean up when done.
	defer func() {
		fmt.Println("\nUnloading Kronk")
		if closer, ok := provider.(interface{ Close(context.Context) error }); ok {
			if err := closer.Close(context.Background()); err != nil {
				fmt.Printf("failed to close provider: %v\n", err)
			}
		}
	}()

	// Get a language model by providing the model URL.
	// The provider will download and initialize the model automatically.
	model, err := provider.LanguageModel(ctx, modelURL)
	if err != nil {
		return fmt.Errorf("unable to get language model: %w", err)
	}

	// -------------------------------------------------------------------------

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
		Prompt: "what does Chuck say when he is happy",

		// When reasoning starts (Qwen3 models use "thinking" mode).
		OnReasoningStart: func(id string, content fantasy.ReasoningContent) error {
			fmt.Print("\n[Thinking: ")
			return nil
		},

		// When we receive reasoning content.
		OnReasoningDelta: func(id, text string) error {
			// Print reasoning in a subdued way
			fmt.Print(text)
			return nil
		},

		// When reasoning ends.
		OnReasoningEnd: func(id string, reasoning fantasy.ReasoningContent) error {
			fmt.Print("]\n\n")
			return nil
		},

		// When we receive a chunk of streaming data.
		OnTextDelta: func(id, text string) error {
			_, fmtErr := fmt.Print(text)
			return fmtErr
		},

		// When tool calls are invoked.
		OnToolCall: func(toolCall fantasy.ToolCallContent) error {
			fmt.Printf("\n-> Invoking the %s tool with input %s\n", toolCall.ToolName, toolCall.Input)
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

	return nil
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
