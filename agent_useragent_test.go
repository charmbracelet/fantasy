package fantasy

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAgent_WithUserAgent_PropagatesOnGenerate(t *testing.T) {
	t.Parallel()

	var capturedCall Call
	model := &mockLanguageModel{
		generateFunc: func(_ context.Context, call Call) (*Response, error) {
			capturedCall = call
			return &Response{
				Content:      []Content{TextContent{Text: "ok"}},
				FinishReason: FinishReasonStop,
			}, nil
		},
	}

	agent := NewAgent(model, WithUserAgent("MyApp/2.0"))
	_, err := agent.Generate(context.Background(), AgentCall{Prompt: "hi"})
	require.NoError(t, err)
	assert.Equal(t, "MyApp/2.0", capturedCall.UserAgent)
}

func TestAgent_WithUserAgent_PropagatesOnStream(t *testing.T) {
	t.Parallel()

	var capturedCall Call
	model := &mockLanguageModel{
		streamFunc: func(_ context.Context, call Call) (StreamResponse, error) {
			capturedCall = call
			return func(yield func(StreamPart) bool) {
				yield(StreamPart{
					Type:         StreamPartTypeFinish,
					FinishReason: FinishReasonStop,
				})
			}, nil
		},
	}

	agent := NewAgent(model, WithUserAgent("StreamApp/1.0"))
	_, err := agent.Stream(context.Background(), AgentStreamCall{Prompt: "hi"})
	require.NoError(t, err)
	assert.Equal(t, "StreamApp/1.0", capturedCall.UserAgent)
}

func TestAgent_NoUA_OmitsCallLevelFields(t *testing.T) {
	t.Parallel()

	var capturedCall Call
	model := &mockLanguageModel{
		generateFunc: func(_ context.Context, call Call) (*Response, error) {
			capturedCall = call
			return &Response{
				Content:      []Content{TextContent{Text: "ok"}},
				FinishReason: FinishReasonStop,
			}, nil
		},
	}

	agent := NewAgent(model)
	_, err := agent.Generate(context.Background(), AgentCall{Prompt: "hi"})
	require.NoError(t, err)
	assert.Empty(t, capturedCall.UserAgent)
}
