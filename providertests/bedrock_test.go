package providertests

import (
	"net/http"
	"testing"

	"github.com/charmbracelet/fantasy/ai"
	"github.com/charmbracelet/fantasy/bedrock"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func TestBedrockCommon(t *testing.T) {
	testCommon(t, []builderPair{
		{"bedrock-anthropic-claude-3-sonnet", builderBedrockClaude3Sonnet, nil},
		{"bedrock-anthropic-claude-3-opus", builderBedrockClaude3Opus, nil},
		{"bedrock-anthropic-claude-3-haiku", builderBedrockClaude3Haiku, nil},
	})
}

func builderBedrockClaude3Sonnet(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := bedrock.New(
		bedrock.WithHTTPClient(&http.Client{Transport: r}),
		bedrock.WithSkipAuth(!r.IsRecording()),
	)
	return provider.LanguageModel("us.anthropic.claude-3-sonnet-20240229-v1:0")
}

func builderBedrockClaude3Opus(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := bedrock.New(
		bedrock.WithHTTPClient(&http.Client{Transport: r}),
		bedrock.WithSkipAuth(!r.IsRecording()),
	)
	return provider.LanguageModel("us.anthropic.claude-3-opus-20240229-v1:0")
}

func builderBedrockClaude3Haiku(r *recorder.Recorder) (ai.LanguageModel, error) {
	provider := bedrock.New(
		bedrock.WithHTTPClient(&http.Client{Transport: r}),
		bedrock.WithSkipAuth(!r.IsRecording()),
	)
	return provider.LanguageModel("us.anthropic.claude-3-haiku-20240307-v1:0")
}
