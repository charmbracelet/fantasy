//go:build nogoogle

package anthropic

import (
	"context"
	"fmt"

	"github.com/charmbracelet/anthropic-sdk-go/option"
)

func (a *provider) configureVertexOptions(ctx context.Context, clientOptions []option.RequestOption) ([]option.RequestOption, error) {
	return nil, fmt.Errorf("Google Vertex AI support not compiled in; remove -tags nogoogle")
}
