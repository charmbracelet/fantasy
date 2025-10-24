package anthropic

import "golang.org/x/oauth2"

type dummyTokenProvider struct{}

// Token implements the auth.TokenProvider interface.
func (dummyTokenProvider) Token() (*oauth2.Token, error) {
	return &oauth2.Token{AccessToken: "dummy-token"}, nil
}
