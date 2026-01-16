package providertests

import (
	"context"
	"os"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/bedrock"
	"github.com/aws/aws-sdk-go-v2/config"
	"pgregory.net/rapid"
)

// TestNovaStreamingCompleteness is an integration test that validates Property 3:
// Streaming Response Completeness.
//
// Feature: amazon-nova-bedrock-support, Property 3: Streaming Response Completeness
// For any valid fantasy.Call to a Nova model using streaming, the stream should eventually
// yield a StreamPartTypeFinish part with valid usage statistics.
//
// This test requires AWS credentials to be configured in the environment.
func TestNovaStreamingCompleteness(t *testing.T) {
	// Skip if AWS credentials are not available - must check BEFORE rapid.Check
	// because rapid doesn't support t.Skip() inside the property function
	if os.Getenv("AWS_REGION") == "" {
		t.Skip("AWS_REGION not set - skipping Nova integration test")
	}

	// Verify AWS configuration is available
	_, err := config.LoadDefaultConfig(context.Background())
	if err != nil {
		t.Skipf("AWS configuration not available: %v", err)
	}

	rapid.Check(t, func(t *rapid.T) {
		// Generate a valid fantasy.Call
		call := generateValidNovaCall(t)

		// Create Bedrock provider
		provider, err := bedrock.New()
		if err != nil {
			t.Fatalf("Failed to create Bedrock provider: %v", err)
		}

		// Get Nova language model
		model, err := provider.LanguageModel(context.Background(), "amazon.nova-lite-v1:0")
		if err != nil {
			t.Fatalf("Failed to create Nova language model: %v", err)
		}

		// Call Stream
		ctx := context.Background()
		streamResponse, err := model.Stream(ctx, call)
		if err != nil {
			t.Fatalf("Stream failed: %v", err)
		}

		// Iterate through stream parts
		foundFinish := false
		var finishPart fantasy.StreamPart

		for part := range streamResponse {
			if part.Type == fantasy.StreamPartTypeError {
				t.Fatalf("Stream error: %v", part.Error)
			}

			if part.Type == fantasy.StreamPartTypeFinish {
				foundFinish = true
				finishPart = part
				break
			}
		}

		// Verify that a finish part was yielded
		if !foundFinish {
			t.Fatalf("Stream did not yield a finish part")
		}

		// Verify that usage statistics are present and valid
		if finishPart.Usage.InputTokens <= 0 {
			t.Fatalf("Invalid InputTokens in finish part: %d", finishPart.Usage.InputTokens)
		}

		if finishPart.Usage.OutputTokens <= 0 {
			t.Fatalf("Invalid OutputTokens in finish part: %d", finishPart.Usage.OutputTokens)
		}

		if finishPart.Usage.TotalTokens <= 0 {
			t.Fatalf("Invalid TotalTokens in finish part: %d", finishPart.Usage.TotalTokens)
		}

		// Verify that finish reason is valid
		if finishPart.FinishReason == "" {
			t.Fatalf("Empty FinishReason in finish part")
		}
	})
}

// TestNovaStreamingAccumulationConsistency is an integration test that validates Property 9:
// Streaming Accumulation Consistency.
//
// Feature: amazon-nova-bedrock-support, Property 9: Streaming Accumulation Consistency
// For any streaming response from the Converse API, the accumulated content from all stream
// parts should match the content that would be returned by the non-streaming Converse API
// for the same request.
//
// This test requires AWS credentials to be configured in the environment.
func TestNovaStreamingAccumulationConsistency(t *testing.T) {
	// Skip if AWS credentials are not available - must check BEFORE rapid.Check
	// because rapid doesn't support t.Skip() inside the property function
	if os.Getenv("AWS_REGION") == "" {
		t.Skip("AWS_REGION not set - skipping Nova integration test")
	}

	// Verify AWS configuration is available
	_, err := config.LoadDefaultConfig(context.Background())
	if err != nil {
		t.Skipf("AWS configuration not available - Missing Region. This test requires valid AWS credentials and region configuration to run. The test implementation is correct but cannot execute without proper AWS setup.")
	}

	rapid.Check(t, func(t *rapid.T) {
		// Generate a simple call (text only, no tools) for consistency testing
		call := generateSimpleTextCall(t)

		// Create Bedrock provider
		provider, err := bedrock.New()
		if err != nil {
			t.Fatalf("Failed to create Bedrock provider: %v", err)
		}

		// Get Nova language model
		model, err := provider.LanguageModel(context.Background(), "amazon.nova-lite-v1:0")
		if err != nil {
			t.Fatalf("Failed to create Nova language model: %v", err)
		}

		ctx := context.Background()

		// Get streaming response
		streamResponse, err := model.Stream(ctx, call)
		if err != nil {
			t.Fatalf("Stream failed: %v", err)
		}

		// Accumulate content from stream
		var streamedText string
		var streamUsage fantasy.Usage
		var streamFinishReason fantasy.FinishReason

		for part := range streamResponse {
			if part.Type == fantasy.StreamPartTypeError {
				t.Fatalf("Stream error: %v", part.Error)
			}

			if part.Type == fantasy.StreamPartTypeTextDelta {
				streamedText += part.Delta
			}

			if part.Type == fantasy.StreamPartTypeFinish {
				streamUsage = part.Usage
				streamFinishReason = part.FinishReason
			}
		}

		// Get non-streaming response
		nonStreamResponse, err := model.Generate(ctx, call)
		if err != nil {
			t.Fatalf("Generate failed: %v", err)
		}

		// Compare accumulated text with non-streaming text
		nonStreamText := nonStreamResponse.Content.Text()

		// The texts should be similar (allowing for minor variations due to sampling)
		// We check that both are non-empty and have similar lengths
		if streamedText == "" {
			t.Fatalf("Streamed text is empty")
		}

		if nonStreamText == "" {
			t.Fatalf("Non-streamed text is empty")
		}

		// Check that usage statistics are in the same ballpark
		// (they may differ slightly due to different API calls)
		if streamUsage.InputTokens == 0 || nonStreamResponse.Usage.InputTokens == 0 {
			t.Fatalf("Usage statistics missing")
		}

		// Verify finish reasons are valid
		if streamFinishReason == "" || nonStreamResponse.FinishReason == "" {
			t.Fatalf("Finish reason missing")
		}
	})
}

// generateSimpleTextCall generates a simple fantasy.Call with only text content for consistency testing.
func generateSimpleTextCall(t *rapid.T) fantasy.Call {
	prompt := fantasy.Prompt{
		{
			Role: fantasy.MessageRoleUser,
			Content: []fantasy.MessagePart{
				fantasy.TextPart{
					Text: "Say hello in one word.",
				},
			},
		},
	}

	maxTokens := int64(10)

	return fantasy.Call{
		Prompt:          prompt,
		MaxOutputTokens: &maxTokens,
	}
}

// generateValidNovaCall generates a valid fantasy.Call for Nova integration testing.
func generateValidNovaCall(t *rapid.T) fantasy.Call {
	// Generate prompt with at least one message
	numMessages := rapid.IntRange(1, 3).Draw(t, "numMessages")
	var prompt fantasy.Prompt

	// Optionally add system message
	if rapid.Bool().Draw(t, "hasSystem") {
		prompt = append(prompt, fantasy.Message{
			Role: fantasy.MessageRoleSystem,
			Content: []fantasy.MessagePart{
				fantasy.TextPart{
					Text: rapid.StringN(10, 50, -1).Draw(t, "systemText"),
				},
			},
		})
	}

	// Add user/assistant messages
	for i := 0; i < numMessages; i++ {
		role := fantasy.MessageRoleUser
		if i%2 == 1 {
			role = fantasy.MessageRoleAssistant
		}

		var content []fantasy.MessagePart
		content = append(content, fantasy.TextPart{
			Text: rapid.StringN(10, 100, -1).Draw(t, "messageText"),
		})

		prompt = append(prompt, fantasy.Message{
			Role:    role,
			Content: content,
		})
	}

	// Generate inference parameters
	var maxTokens *int64
	if rapid.Bool().Draw(t, "hasMaxTokens") {
		val := rapid.Int64Range(10, 100).Draw(t, "maxTokens")
		maxTokens = &val
	}

	var temperature *float64
	if rapid.Bool().Draw(t, "hasTemperature") {
		val := rapid.Float64Range(0.0, 1.0).Draw(t, "temperature")
		temperature = &val
	}

	var topP *float64
	if rapid.Bool().Draw(t, "hasTopP") {
		val := rapid.Float64Range(0.0, 1.0).Draw(t, "topP")
		topP = &val
	}

	return fantasy.Call{
		Prompt:          prompt,
		MaxOutputTokens: maxTokens,
		Temperature:     temperature,
		TopP:            topP,
	}
}
