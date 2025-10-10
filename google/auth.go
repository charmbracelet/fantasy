package google

import (
	"context"

	"cloud.google.com/go/auth"
)

type dummyTokenProvider struct{}

func (dummyTokenProvider) Token(ctx context.Context) (*auth.Token, error) {
	return &auth.Token{Value: "dummy-token"}, nil
}
