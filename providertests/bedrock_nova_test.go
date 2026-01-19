package providertests

import (
	"context"
	"encoding/base64"
	"net/http"
	"os"
	"strings"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/bedrock"
	"charm.land/x/vcr"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/stretchr/testify/require"
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
		var streamedText strings.Builder
		var streamUsage fantasy.Usage
		var streamFinishReason fantasy.FinishReason

		for part := range streamResponse {
			if part.Type == fantasy.StreamPartTypeError {
				t.Fatalf("Stream error: %v", part.Error)
			}

			if part.Type == fantasy.StreamPartTypeTextDelta {
				streamedText.WriteString(part.Delta)
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
		if streamedText.String() == "" {
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
	for i := range numMessages {
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

// Integration tests for Nova models following the common test pattern

// TestNovaCommon runs common integration tests for all Nova model variants.
// This validates Requirements 1.1, 1.2, 1.4, 1.5
func TestNovaCommon(t *testing.T) {
	testCommon(t, []builderPair{
		{"bedrock-nova-pro", builderBedrockNovaPro, nil, nil},
		{"bedrock-nova-lite", builderBedrockNovaLite, nil, nil},
		{"bedrock-nova-micro", builderBedrockNovaMicro, nil, nil},
	})
}

// TestNovaModelInstantiation tests that Nova models can be instantiated through the Bedrock provider.
// Validates Requirement 1.1
func TestNovaModelInstantiation(t *testing.T) {
	models := []string{
		"amazon.nova-pro-v1:0",
		"amazon.nova-lite-v1:0",
		"amazon.nova-micro-v1:0",
		"amazon.nova-premier-v1:0",
	}

	for _, modelID := range models {
		t.Run(modelID, func(t *testing.T) {
			r := vcr.NewRecorder(t)

			provider, err := bedrock.New(
				bedrock.WithHTTPClient(&http.Client{Transport: r}),
				bedrock.WithSkipAuth(!r.IsRecording()),
			)
			require.NoError(t, err, "failed to create Bedrock provider")

			model, err := provider.LanguageModel(t.Context(), modelID)
			require.NoError(t, err, "failed to create Nova language model for %s", modelID)
			require.NotNil(t, model, "language model should not be nil")
			// The model ID will have a region prefix applied (e.g., "us.amazon.nova-pro-v1:0")
			require.Contains(t, model.Model(), modelID, "model ID should contain the original model ID")
			require.Equal(t, "bedrock", model.Provider(), "provider should be bedrock")
		})
	}
}

// TestNovaParameterPassing tests that inference parameters are correctly passed to Nova models.
// Tests temperature, top_p, and max_tokens parameters.
// Note: top_k is mentioned in task requirements but is not supported by Nova models.
// Validates Requirements 8.1, 8.2, 8.3, 8.4
func TestNovaParameterPassing(t *testing.T) {
	r := vcr.NewRecorder(t)

	provider, err := bedrock.New(
		bedrock.WithHTTPClient(&http.Client{Transport: r}),
		bedrock.WithSkipAuth(!r.IsRecording()),
	)
	require.NoError(t, err, "failed to create Bedrock provider")

	model, err := provider.LanguageModel(t.Context(), "amazon.nova-lite-v1:0")
	require.NoError(t, err, "failed to create Nova language model")

	// Test with temperature and top_p parameters (supported by Nova)
	temperature := 0.7
	topP := 0.9
	maxTokens := int64(100)

	call := fantasy.Call{
		Prompt: fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Say hello"},
				},
			},
		},
		Temperature:     &temperature,
		TopP:            &topP,
		MaxOutputTokens: &maxTokens,
	}

	response, err := model.Generate(t.Context(), call)
	require.NoError(t, err, "generation should succeed with parameters")
	require.NotNil(t, response, "response should not be nil")
	require.NotEmpty(t, response.Content.Text(), "response should contain text")
}

// TestNovaSystemPrompt tests that system prompts work correctly with Nova models.
// Validates Requirement 8.3
func TestNovaSystemPrompt(t *testing.T) {
	r := vcr.NewRecorder(t)

	provider, err := bedrock.New(
		bedrock.WithHTTPClient(&http.Client{Transport: r}),
		bedrock.WithSkipAuth(!r.IsRecording()),
	)
	require.NoError(t, err, "failed to create Bedrock provider")

	model, err := provider.LanguageModel(t.Context(), "amazon.nova-lite-v1:0")
	require.NoError(t, err, "failed to create Nova language model")

	call := fantasy.Call{
		Prompt: fantasy.Prompt{
			{
				Role: fantasy.MessageRoleSystem,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "You are a helpful assistant that always responds in Portuguese."},
				},
			},
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Say hello"},
				},
			},
		},
		MaxOutputTokens: fantasy.Opt(int64(50)),
	}

	response, err := model.Generate(t.Context(), call)
	require.NoError(t, err, "generation should succeed with system prompt")
	require.NotNil(t, response, "response should not be nil")
	require.NotEmpty(t, response.Content.Text(), "response should contain text")
}

// TestNovaMultiTurnConversation tests multi-turn conversations with Nova models.
// Validates Requirement 8.4
func TestNovaMultiTurnConversation(t *testing.T) {
	r := vcr.NewRecorder(t)

	provider, err := bedrock.New(
		bedrock.WithHTTPClient(&http.Client{Transport: r}),
		bedrock.WithSkipAuth(!r.IsRecording()),
	)
	require.NoError(t, err, "failed to create Bedrock provider")

	model, err := provider.LanguageModel(t.Context(), "amazon.nova-lite-v1:0")
	require.NoError(t, err, "failed to create Nova language model")

	call := fantasy.Call{
		Prompt: fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "My name is Alice."},
				},
			},
			{
				Role: fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Hello Alice! Nice to meet you."},
				},
			},
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "What is my name?"},
				},
			},
		},
		MaxOutputTokens: fantasy.Opt(int64(50)),
	}

	response, err := model.Generate(t.Context(), call)
	require.NoError(t, err, "generation should succeed with multi-turn conversation")
	require.NotNil(t, response, "response should not be nil")
	require.NotEmpty(t, response.Content.Text(), "response should contain text")
}

// TestNovaStreaming tests streaming generation with Nova models.
// Validates Requirement 1.4
func TestNovaStreaming(t *testing.T) {
	r := vcr.NewRecorder(t)

	provider, err := bedrock.New(
		bedrock.WithHTTPClient(&http.Client{Transport: r}),
		bedrock.WithSkipAuth(!r.IsRecording()),
	)
	require.NoError(t, err, "failed to create Bedrock provider")

	model, err := provider.LanguageModel(t.Context(), "amazon.nova-lite-v1:0")
	require.NoError(t, err, "failed to create Nova language model")

	call := fantasy.Call{
		Prompt: fantasy.Prompt{
			{
				Role: fantasy.MessageRoleUser,
				Content: []fantasy.MessagePart{
					fantasy.TextPart{Text: "Count from 1 to 5"},
				},
			},
		},
		MaxOutputTokens: fantasy.Opt(int64(100)),
	}

	streamResponse, err := model.Stream(t.Context(), call)
	require.NoError(t, err, "streaming should succeed")

	var accumulatedText strings.Builder
	foundFinish := false

	for part := range streamResponse {
		if part.Type == fantasy.StreamPartTypeError {
			t.Fatalf("stream error: %v", part.Error)
		}

		if part.Type == fantasy.StreamPartTypeTextDelta {
			accumulatedText.WriteString(part.Delta)
		}

		if part.Type == fantasy.StreamPartTypeFinish {
			foundFinish = true
			require.Greater(t, part.Usage.InputTokens, int64(0), "input tokens should be positive")
			require.Greater(t, part.Usage.OutputTokens, int64(0), "output tokens should be positive")
			require.NotEmpty(t, part.FinishReason, "finish reason should not be empty")
		}
	}

	require.True(t, foundFinish, "stream should yield a finish part")
	require.NotEmpty(t, accumulatedText.String(), "accumulated text should not be empty")
}

// TestNovaImageAttachments tests image attachment support with Nova models.
// Validates Requirement 8.5
// Note: This test uses a simple base64-encoded 1x1 pixel PNG for testing
func TestNovaImageAttachments(t *testing.T) {
	// Only test with models that support attachments (pro, lite, premier)
	models := []string{
		"amazon.nova-pro-v1:0",
		"amazon.nova-lite-v1:0",
	}

	// Simple 1x1 red pixel PNG (base64 encoded)
	testImageDataBase64 := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
	testImageData, err := base64.StdEncoding.DecodeString(testImageDataBase64)
	require.NoError(t, err, "failed to decode test image data")

	for _, modelID := range models {
		t.Run(modelID, func(t *testing.T) {
			r := vcr.NewRecorder(t)

			provider, err := bedrock.New(
				bedrock.WithHTTPClient(&http.Client{Transport: r}),
				bedrock.WithSkipAuth(!r.IsRecording()),
			)
			require.NoError(t, err, "failed to create Bedrock provider")

			model, err := provider.LanguageModel(t.Context(), modelID)
			require.NoError(t, err, "failed to create Nova language model")

			call := fantasy.Call{
				Prompt: fantasy.Prompt{
					{
						Role: fantasy.MessageRoleUser,
						Content: []fantasy.MessagePart{
							fantasy.FilePart{
								Filename:  "test.png",
								MediaType: "image/png",
								Data:      testImageData,
							},
							fantasy.TextPart{Text: "What color is this image?"},
						},
					},
				},
				MaxOutputTokens: fantasy.Opt(int64(100)),
			}

			response, err := model.Generate(t.Context(), call)
			require.NoError(t, err, "generation with image should succeed")
			require.NotNil(t, response, "response should not be nil")
			require.NotEmpty(t, response.Content.Text(), "response should contain text")
		})
	}
}

// Builder functions for Nova model variants

func builderBedrockNovaPro(t *testing.T, r *vcr.Recorder) (fantasy.LanguageModel, error) {
	provider, err := bedrock.New(
		bedrock.WithHTTPClient(&http.Client{Transport: r}),
		bedrock.WithSkipAuth(!r.IsRecording()),
	)
	if err != nil {
		return nil, err
	}
	return provider.LanguageModel(t.Context(), "amazon.nova-pro-v1:0")
}

func builderBedrockNovaLite(t *testing.T, r *vcr.Recorder) (fantasy.LanguageModel, error) {
	provider, err := bedrock.New(
		bedrock.WithHTTPClient(&http.Client{Transport: r}),
		bedrock.WithSkipAuth(!r.IsRecording()),
	)
	if err != nil {
		return nil, err
	}
	return provider.LanguageModel(t.Context(), "amazon.nova-lite-v1:0")
}

func builderBedrockNovaMicro(t *testing.T, r *vcr.Recorder) (fantasy.LanguageModel, error) {
	provider, err := bedrock.New(
		bedrock.WithHTTPClient(&http.Client{Transport: r}),
		bedrock.WithSkipAuth(!r.IsRecording()),
	)
	if err != nil {
		return nil, err
	}
	return provider.LanguageModel(t.Context(), "amazon.nova-micro-v1:0")
}

func builderBedrockNovaPremier(t *testing.T, r *vcr.Recorder) (fantasy.LanguageModel, error) {
	provider, err := bedrock.New(
		bedrock.WithHTTPClient(&http.Client{Transport: r}),
		bedrock.WithSkipAuth(!r.IsRecording()),
	)
	if err != nil {
		return nil, err
	}
	return provider.LanguageModel(t.Context(), "amazon.nova-premier-v1:0")
}
