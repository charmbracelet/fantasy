package providertests

import (
	"net/http"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/bedrock"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func TestBedrockCommon(t *testing.T) {
	testCommon(t, []builderPair{
		{"bedrock-anthropic-claude-3-sonnet", builderBedrockClaude3Sonnet, nil, nil},
		{"bedrock-anthropic-claude-3-haiku", builderBedrockClaude3Haiku, nil, nil},
	})
}



func builderBedrockClaude3Sonnet(t *testing.T, r *recorder.Recorder) (fantasy.LanguageModel, error) {
	t.Setenv("AWS_REGION", "us-east-1")
	provider, err := bedrock.New(
		bedrock.WithHTTPClient(&http.Client{Transport: r}),
		bedrock.WithAPIKey("dummy"),
	)
	if err != nil {
		return nil, err
	}
	return provider.LanguageModel(t.Context(), "anthropic.claude-3-sonnet-20240229-v1:0")
}



func builderBedrockClaude3Haiku(t *testing.T, r *recorder.Recorder) (fantasy.LanguageModel, error) {
	t.Setenv("AWS_REGION", "us-east-1")
	provider, err := bedrock.New(
		bedrock.WithHTTPClient(&http.Client{Transport: r}),
		bedrock.WithAPIKey("dummy"),
	)
	if err != nil {
		return nil, err
	}
	return provider.LanguageModel(t.Context(), "anthropic.claude-3-haiku-20240307-v1:0")
}


