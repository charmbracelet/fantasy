package providertests

import (
	"net/http"
	"os"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/bedrock"
	"charm.land/x/vcr"
)

func TestBedrockCommon(t *testing.T) {
	testCommon(t, []builderPair{
		{"bedrock-anthropic-claude-sonnet-4-5", builderBedrockClaudeSonnet, nil, nil},
		{"bedrock-anthropic-claude-opus-4-6", builderBedrockClaudeOpus, nil, nil},
		{"bedrock-anthropic-claude-haiku-4-5", builderBedrockClaudeHaiku, nil, nil},
	})
}

func TestBedrockBasicAuth(t *testing.T) {
	testSimple(t, builderPair{"bedrock-anthropic-claude-3-sonnet", buildersBedrockBasicAuth, nil, nil})
}

func builderBedrockClaudeSonnet(t *testing.T, r *vcr.Recorder) (fantasy.LanguageModel, error) {
	provider, err := bedrock.New(
		bedrock.WithHTTPClient(&http.Client{Transport: r}),
		bedrock.WithSkipAuth(!r.IsRecording()),
	)
	if err != nil {
		return nil, err
	}
	return provider.LanguageModel(t.Context(), "anthropic.claude-sonnet-4-5-20250929-v1:0")
}

func builderBedrockClaudeOpus(t *testing.T, r *vcr.Recorder) (fantasy.LanguageModel, error) {
	provider, err := bedrock.New(
		bedrock.WithHTTPClient(&http.Client{Transport: r}),
		bedrock.WithSkipAuth(!r.IsRecording()),
	)
	if err != nil {
		return nil, err
	}
	return provider.LanguageModel(t.Context(), "anthropic.claude-opus-4-6-v1")
}

func builderBedrockClaudeHaiku(t *testing.T, r *vcr.Recorder) (fantasy.LanguageModel, error) {
	provider, err := bedrock.New(
		bedrock.WithHTTPClient(&http.Client{Transport: r}),
		bedrock.WithSkipAuth(!r.IsRecording()),
	)
	if err != nil {
		return nil, err
	}
	return provider.LanguageModel(t.Context(), "anthropic.claude-haiku-4-5-20251001-v1:0")
}

func buildersBedrockBasicAuth(t *testing.T, r *vcr.Recorder) (fantasy.LanguageModel, error) {
	provider, err := bedrock.New(
		bedrock.WithHTTPClient(&http.Client{Transport: r}),
		bedrock.WithAPIKey(os.Getenv("FANTASY_BEDROCK_API_KEY")),
		bedrock.WithSkipAuth(true),
	)
	if err != nil {
		return nil, err
	}
	return provider.LanguageModel(t.Context(), "anthropic.claude-haiku-4-5-20251001-v1:0")
}
