package anthropic

import (
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/smithy-go/auth/bearer"
)

func bedrockBasicAuthConfig(apiKey string) aws.Config {
	return aws.Config{
		Region:                  "us-east-1",
		BearerAuthTokenProvider: bearer.StaticTokenProvider{Token: bearer.Token{Value: apiKey}},
	}
}
