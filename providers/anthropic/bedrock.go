package anthropic

import (
	"cmp"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/smithy-go/auth/bearer"
)

func bedrockBasicAuthConfig(apiKey string, region string) aws.Config {
	return aws.Config{
		Region:                  cmp.Or(region, "us-east-1"),
		BearerAuthTokenProvider: bearer.StaticTokenProvider{Token: bearer.Token{Value: apiKey}},
	}
}
