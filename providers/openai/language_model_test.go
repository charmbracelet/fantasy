package openai

import (
	"testing"

	"charm.land/fantasy"
	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/require"
)

func TestPrepareParams_TopK(t *testing.T) {
	t.Parallel()

	prompt := fantasy.Prompt{testTextMessage(fantasy.MessageRoleUser, "hello")}

	t.Run("warns when no hook sends top_k", func(t *testing.T) {
		t.Parallel()

		lm := languageModel{
			provider:        Name,
			modelID:         "gpt-4o",
			prepareCallFunc: DefaultPrepareCallFunc,
			toPromptFunc:    DefaultToPrompt,
		}

		params, warnings, err := lm.prepareParams(fantasy.Call{Prompt: prompt, TopK: new(int64(20))})

		require.NoError(t, err)
		require.NotContains(t, params.ExtraFields(), "top_k")
		require.Contains(t, warnings, fantasy.CallWarning{
			Type:    fantasy.CallWarningTypeUnsupportedSetting,
			Setting: "top_k",
		})
	})

	t.Run("stays silent when a hook sends top_k", func(t *testing.T) {
		t.Parallel()

		lm := languageModel{
			provider: Name,
			modelID:  "gpt-4o",
			prepareCallFunc: func(_ fantasy.LanguageModel, params *openai.ChatCompletionNewParams, call fantasy.Call) ([]fantasy.CallWarning, error) {
				params.SetExtraFields(map[string]any{"top_k": *call.TopK})
				return nil, nil
			},
			toPromptFunc: DefaultToPrompt,
		}

		params, warnings, err := lm.prepareParams(fantasy.Call{Prompt: prompt, TopK: new(int64(20))})

		require.NoError(t, err)
		require.Equal(t, int64(20), params.ExtraFields()["top_k"])
		require.Empty(t, warnings)
	})
}
