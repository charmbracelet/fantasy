package anthropic

import (
	"cmp"
	"os"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/smithy-go/auth/bearer"
)

func bedrockBasicAuthConfig(apiKey string) aws.Config {
	return aws.Config{
		Region:                  cmp.Or(awsRegion(), "us-east-1"),
		BearerAuthTokenProvider: bearer.StaticTokenProvider{Token: bearer.Token{Value: apiKey}},
	}
}

func bedrockPrefixModelWithRegion(modelID string) string {
	if hasBedrockInferenceProfilePrefix(modelID) {
		return modelID
	}
	region := cmp.Or(awsRegion(), "us-east-1")
	return bedrockRegionPrefix(region) + "." + modelID
}

func awsRegion() string {
	return cmp.Or(os.Getenv("AWS_REGION"), os.Getenv("AWS_DEFAULT_REGION"))
}

func bedrockRegionPrefix(region string) string {
	switch {
	case strings.HasPrefix(region, "us-") || region == "ca-central-1":
		return "us"
	case strings.HasPrefix(region, "eu-"):
		return "eu"
	case region == "ap-northeast-1":
		return "jp"
	case region == "ap-southeast-2":
		return "au"
	default:
		return "global"
	}
}

func hasBedrockInferenceProfilePrefix(modelID string) bool {
	for _, prefix := range []string{"us.", "eu.", "jp.", "au.", "global."} {
		if strings.HasPrefix(modelID, prefix) {
			return true
		}
	}
	return false
}
