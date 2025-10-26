package anthropic

import (
	"cmp"
	"os"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/smithy-go/auth/bearer"
)

func bedrockBasicAuthConfig(apiKey string) aws.Config {
	return aws.Config{
		Region:                  cmp.Or(os.Getenv("AWS_REGION"), "us-east-1"),
		BearerAuthTokenProvider: bearer.StaticTokenProvider{Token: bearer.Token{Value: apiKey}},
	}
}
