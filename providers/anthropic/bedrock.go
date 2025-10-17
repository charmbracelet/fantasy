package anthropic

import (
	"context"

	"github.com/aws/aws-sdk-go-v2/aws"
)

var dummyBedrockConfig = aws.Config{
	Region: "us-east-1",
	Credentials: aws.CredentialsProviderFunc(func(context.Context) (aws.Credentials, error) {
		return aws.Credentials{}, nil
	}),
}
