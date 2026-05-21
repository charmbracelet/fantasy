//go:build noaws

package anthropic

import (
	"context"
	"fmt"

	"github.com/charmbracelet/anthropic-sdk-go/option"
)

func (a *provider) configureBedrockOptions(ctx context.Context, modelID string, clientOptions []option.RequestOption) (string, []option.RequestOption, error) {
	return "", nil, fmt.Errorf("AWS Bedrock support not compiled in; remove -tags noaws")
}
