//go:build !noaws

package anthropic

import (
	"context"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/charmbracelet/anthropic-sdk-go/bedrock"
	"github.com/charmbracelet/anthropic-sdk-go/option"
)

// configureBedrockOptions applies AWS Bedrock-specific options when useBedrock is set.
// Returns the potentially modified modelID (with region prefix) and updated client options.
func (a *provider) configureBedrockOptions(ctx context.Context, modelID string, clientOptions []option.RequestOption) (string, []option.RequestOption, error) {
	modelID = bedrockPrefixModelWithRegion(modelID)

	if a.options.skipAuth || a.options.apiKey != "" {
		clientOptions = append(clientOptions, bedrock.WithConfig(bedrockBasicAuthConfig(a.options.apiKey)))
	} else {
		cfg, err := config.LoadDefaultConfig(ctx)
		if err != nil {
			return "", nil, err
		}
		clientOptions = append(clientOptions, bedrock.WithConfig(cfg))
	}

	return modelID, clientOptions, nil
}
