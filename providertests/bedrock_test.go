package providertests

import (
	"net/http"
	"os"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/bedrock"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func TestBedrockCommon(t *testing.T) {
	testCommon(t, []builderPair{
		{"bedrock-anthropic-claude-3-sonnet", builderBedrockClaude3Sonnet, nil, nil},
		{"bedrock-anthropic-claude-3-opus", builderBedrockClaude3Opus, nil, nil},
		{"bedrock-anthropic-claude-3-haiku", builderBedrockClaude3Haiku, nil, nil},
	})
}

func TestBedrockBasicAuth(t *testing.T) {
	testSimple(t, builderPair{"bedrock-anthropic-claude-3-sonnet", buildersBedrockBasicAuth, nil, nil})
}

func builderBedrockClaude3Sonnet(t *testing.T, r *recorder.Recorder) (fantasy.LanguageModel, error) {
	provider, err := bedrock.New(
		bedrock.WithHTTPClient(&http.Client{Transport: r}),
		bedrock.WithSkipAuth(!r.IsRecording()),
	)
	if err != nil {
		return nil, err
	}
	return provider.LanguageModel(t.Context(), "us.anthropic.claude-3-sonnet-20240229-v1:0")
}

func builderBedrockClaude3Opus(t *testing.T, r *recorder.Recorder) (fantasy.LanguageModel, error) {
	provider, err := bedrock.New(
		bedrock.WithHTTPClient(&http.Client{Transport: r}),
		bedrock.WithSkipAuth(!r.IsRecording()),
	)
	if err != nil {
		return nil, err
	}
	return provider.LanguageModel(t.Context(), "us.anthropic.claude-3-opus-20240229-v1:0")
}

func builderBedrockClaude3Haiku(t *testing.T, r *recorder.Recorder) (fantasy.LanguageModel, error) {
	provider, err := bedrock.New(
		bedrock.WithHTTPClient(&http.Client{Transport: r}),
		bedrock.WithSkipAuth(!r.IsRecording()),
	)
	if err != nil {
		return nil, err
	}
	return provider.LanguageModel(t.Context(), "us.anthropic.claude-3-haiku-20240307-v1:0")
}

func buildersBedrockBasicAuth(t *testing.T, r *recorder.Recorder) (fantasy.LanguageModel, error) {
	provider, err := bedrock.New(
		bedrock.WithHTTPClient(&http.Client{Transport: r}),
		bedrock.WithAPIKey(os.Getenv("FANTASY_BEDROCK_API_KEY")),
		bedrock.WithSkipAuth(true),
	)
	if err != nil {
		return nil, err
	}
	return provider.LanguageModel(t.Context(), "us.anthropic.claude-3-sonnet-20240229-v1:0")
}
