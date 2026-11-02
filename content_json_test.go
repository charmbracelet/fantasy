package fantasy

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

type codecTestProviderData struct {
	Value string `json:"value"`
}

func (*codecTestProviderData) Options() {}

func (d *codecTestProviderData) MarshalJSON() ([]byte, error) {
	type plain codecTestProviderData
	return MarshalProviderType("fantasy.codec_test", plain(*d))
}

func (d *codecTestProviderData) UnmarshalJSON(data []byte) error {
	type plain codecTestProviderData
	var value plain
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	*d = codecTestProviderData(value)
	return nil
}

const codecTestProviderDataJSON = `{"fantasy":{"type":"fantasy.codec_test","data":{"value":"kept"}}}`

func registerCodecTestProvider(t *testing.T) {
	t.Helper()
	RegisterProviderType("fantasy.codec_test", func(data []byte) (ProviderOptionsData, error) {
		var value codecTestProviderData
		if err := json.Unmarshal(data, &value); err != nil {
			return nil, err
		}
		return &value, nil
	})
}

func TestContentCodecsDecodeProviderMetadata(t *testing.T) {
	registerCodecTestProvider(t)

	t.Run("text content", func(t *testing.T) {
		content, err := UnmarshalContent([]byte(`{"type":"text","data":{"text":"hello","provider_metadata":` + codecTestProviderDataJSON + `}}`))
		require.NoError(t, err)

		text, ok := content.(TextContent)
		require.True(t, ok)
		require.Equal(t, "hello", text.Text)
		require.Equal(t, "kept", text.ProviderMetadata["fantasy"].(*codecTestProviderData).Value)
	})

	t.Run("text content without provider metadata", func(t *testing.T) {
		content, err := UnmarshalContent([]byte(`{"type":"text","data":{"text":"hello"}}`))
		require.NoError(t, err)

		text, ok := content.(TextContent)
		require.True(t, ok)
		require.Equal(t, "hello", text.Text)
		require.Nil(t, text.ProviderMetadata)
	})

	t.Run("reasoning content with an empty provider map", func(t *testing.T) {
		content, err := UnmarshalContent([]byte(`{"type":"reasoning","data":{"text":"think","provider_metadata":{}}}`))
		require.NoError(t, err)

		reasoning, ok := content.(ReasoningContent)
		require.True(t, ok)
		require.Equal(t, "think", reasoning.Text)
		require.Nil(t, reasoning.ProviderMetadata)
	})

	t.Run("text that is not a string is rejected", func(t *testing.T) {
		_, err := UnmarshalContent([]byte(`{"type":"text","data":{"text":5}}`))
		require.Error(t, err)
	})
}

func TestMessagePartCodecsDecodeProviderOptions(t *testing.T) {
	registerCodecTestProvider(t)

	t.Run("text part", func(t *testing.T) {
		part, err := UnmarshalMessagePart([]byte(`{"type":"text","data":{"text":"hello","provider_options":` + codecTestProviderDataJSON + `}}`))
		require.NoError(t, err)

		text, ok := part.(TextPart)
		require.True(t, ok)
		require.Equal(t, "hello", text.Text)
		require.Equal(t, "kept", text.ProviderOptions["fantasy"].(*codecTestProviderData).Value)
	})

	t.Run("reasoning part without provider options", func(t *testing.T) {
		part, err := UnmarshalMessagePart([]byte(`{"type":"reasoning","data":{"text":"think"}}`))
		require.NoError(t, err)

		reasoning, ok := part.(ReasoningPart)
		require.True(t, ok)
		require.Equal(t, "think", reasoning.Text)
		require.Nil(t, reasoning.ProviderOptions)
	})
}
