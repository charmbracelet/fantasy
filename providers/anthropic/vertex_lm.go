//go:build !nogoogle

package anthropic

import (
	"context"

	"github.com/charmbracelet/anthropic-sdk-go/option"
	"github.com/charmbracelet/anthropic-sdk-go/vertex"
	"golang.org/x/oauth2/google"
)

// configureVertexOptions applies Google Vertex AI-specific options when both
// vertexProject and vertexLocation are set.
func (a *provider) configureVertexOptions(ctx context.Context, clientOptions []option.RequestOption) ([]option.RequestOption, error) {
	var credentials *google.Credentials
	if a.options.skipAuth {
		credentials = &google.Credentials{TokenSource: &googleDummyTokenSource{}}
	} else {
		var err error
		credentials, err = google.FindDefaultCredentials(ctx, VertexAuthScope)
		if err != nil {
			return nil, err
		}
	}

	clientOptions = append(clientOptions, vertex.WithCredentials(
		ctx,
		a.options.vertexLocation,
		a.options.vertexProject,
		credentials,
	))

	return clientOptions, nil
}
