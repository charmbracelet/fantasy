package bedrock

import (
	"context"
	"fmt"

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
// This is a stub that will be implemented in task 10.
func (n *novaLanguageModel) Stream(ctx context.Context, call fantasy.Call) (fantasy.StreamResponse, error) {
	return nil, fmt.Errorf("Stream not yet implemented for Nova models")
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
	// Load AWS configuration using default credential chain
	cfg, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to load AWS configuration: %w", err)
	}

	// Apply region prefix to model ID
	// The region is obtained from the AWS config
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
