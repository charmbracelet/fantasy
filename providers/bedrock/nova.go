package bedrock

import (
	"context"
	"fmt"
	"strings"
	"time"

	"charm.land/fantasy"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

// novaLanguageModel implements the fantasy.LanguageModel interface for Amazon Nova models
// using the AWS SDK Bedrock Runtime Converse API.
type novaLanguageModel struct {
	modelID  string
	provider string
	client   *bedrockruntime.Client
	options  options
}

// Model returns the model ID.
func (n *novaLanguageModel) Model() string {
	return n.modelID
}

// Provider returns the provider name.
func (n *novaLanguageModel) Provider() string {
	return n.provider
}

// Generate implements non-streaming generation.
// It converts the fantasy.Call to a Converse API request, invokes the API,
// and converts the response back to fantasy.Response format.
func (n *novaLanguageModel) Generate(ctx context.Context, call fantasy.Call) (*fantasy.Response, error) {
	// Prepare the Converse API request
	request, warnings, err := n.prepareConverseRequest(call)
	if err != nil {
		return nil, fmt.Errorf("failed to prepare converse request: %w", err)
	}

	// Apply extended thinking timeout if enabled
	if thinkingEnabled(extractProviderOptions(call)) {
		opts := extractProviderOptions(call)
		effort := resolveReasoningEffort(opts)

		timeout := getReasoningTimeout(effort, opts.Thinking.TimeoutMinutes)
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}

	// Invoke the Converse API
	output, err := n.client.Converse(ctx, request)
	if err != nil {
		return nil, convertAWSError(err)
	}

	// Convert the response to fantasy.Response
	response, err := n.convertConverseResponse(output, warnings)
	if err != nil {
		return nil, fmt.Errorf("failed to convert converse response: %w", err)
	}

	return response, nil
}

// Stream implements streaming generation.
// It converts the fantasy.Call to a ConverseStream API request, invokes the API,
// and returns a streaming response handler.
func (n *novaLanguageModel) Stream(ctx context.Context, call fantasy.Call) (fantasy.StreamResponse, error) {
	// Prepare the ConverseStream API request
	request, warnings, err := n.prepareConverseStreamRequest(call)
	if err != nil {
		return nil, fmt.Errorf("failed to prepare converse stream request: %w", err)
	}

	// Apply extended thinking timeout if enabled
	if thinkingEnabled(extractProviderOptions(call)) {
		opts := extractProviderOptions(call)
		effort := resolveReasoningEffort(opts)

		timeout := getReasoningTimeout(effort, opts.Thinking.TimeoutMinutes)
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}

	// Invoke the ConverseStream API
	output, err := n.client.ConverseStream(ctx, request)
	if err != nil {
		return nil, convertAWSError(err)
	}

	// Return streaming response handler
	return n.handleConverseStream(output, warnings), nil
}

// GenerateObject implements object generation.
// This is a stub that will be implemented later if needed.
func (n *novaLanguageModel) GenerateObject(ctx context.Context, call fantasy.ObjectCall) (*fantasy.ObjectResponse, error) {
	return nil, fmt.Errorf("GenerateObject not yet implemented for Nova models")
}

// StreamObject implements streaming object generation.
// This is a stub that will be implemented later if needed.
func (n *novaLanguageModel) StreamObject(ctx context.Context, call fantasy.ObjectCall) (fantasy.ObjectStreamResponse, error) {
	return nil, fmt.Errorf("StreamObject not yet implemented for Nova models")
}

// createNovaModel creates a language model instance for Nova models.
// It loads AWS configuration, applies region prefix to the model ID,
// and creates a Bedrock Runtime client.
func (p *provider) createNovaModel(ctx context.Context, modelID string) (fantasy.LanguageModel, error) {
	cfgOpts := []func(*config.LoadOptions) error{}
	if p.options.region != "" {
		cfgOpts = append(cfgOpts, config.WithRegion(p.options.region))
	}

	cfg, err := config.LoadDefaultConfig(ctx, cfgOpts...)
	if err != nil {
		return nil, fmt.Errorf("failed to load AWS configuration: %w", err)
	}

	prefixedModelID := applyRegionPrefix(modelID, cfg.Region)

	// Create Bedrock Runtime client
	client := bedrockruntime.NewFromConfig(cfg)

	return &novaLanguageModel{
		modelID:  prefixedModelID,
		provider: Name,
		client:   client,
		options:  p.options,
	}, nil
}

// supportsExtendedThinking checks if a Nova model supports extended thinking.
// Only Nova 2 Lite supports extended thinking (nova-2-lite-v1:0).
// Nova Generation 1 models (Micro, Lite, Pro, Premier) do NOT support it.
func supportsExtendedThinking(modelID string) bool {
	// Remove region prefix using existing stripRegionPrefix function
	modelID = stripRegionPrefix(modelID)

	// Check for Nova 2 Lite
	return strings.Contains(modelID, "nova-2-lite")
}

// stripRegionPrefix removes the region prefix from a model ID.
// Reuses logic from applyRegionPrefix in region.go.
// E.g., "us.amazon.nova-pro-v1:0" -> "amazon.nova-pro-v1:0"
func stripRegionPrefix(modelID string) string {
	// Region prefixes are always 2 letters followed by a dot (e.g., "us.", "eu.")
	if len(modelID) >= 3 && modelID[2] == '.' {
		firstTwo := modelID[:2]
		// Check if it's a lowercase letter pattern (region code)
		// Reuse isLowercaseLetters from region.go
		if isLowercaseLetters(firstTwo) {
			return modelID[3:]
		}
	}
	return modelID
}

// requiresHighEffortRestrictions checks if high effort mode parameter restrictions apply.
// In high effort mode, temperature, topP, and topK must not be set.
func requiresHighEffortRestrictions(effort ReasoningEffort) bool {
	return effort == ReasoningEffortHigh
}

// getReasoningTimeout returns the appropriate timeout duration based on reasoning effort.
// AWS recommends 60+ minute timeouts for extended thinking operations.
func getReasoningTimeout(effort ReasoningEffort, customTimeout int) time.Duration {
	if customTimeout > 0 {
		return time.Duration(customTimeout) * time.Minute
	}

	// Default timeouts based on AWS recommendations
	switch effort {
	case ReasoningEffortLow:
		return 10 * time.Minute
	case ReasoningEffortMedium:
		return 30 * time.Minute
	case ReasoningEffortHigh:
		return 90 * time.Minute
	default:
		return 30 * time.Minute
	}
}
