package fantasy

import "github.com/go-viper/mapstructure/v2"

func Opt[T any](v T) *T {
	return &v
}

func ParseOptions[T any](options map[string]any, m *T) error {
	decoder, err := mapstructure.NewDecoder(&mapstructure.DecoderConfig{
		TagName: "json",
		Result:  m,
	})
	if err != nil {
		return err
	}
	return decoder.Decode(options)
}
