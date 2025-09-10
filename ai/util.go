package ai

import (
	"github.com/go-viper/mapstructure/v2"
)

func ParseOptions[T any](options map[string]any, m *T) error {
	return mapstructure.Decode(options, m)
}

func FloatOption(f float64) *float64 {
	return &f
}

func BoolOption(b bool) *bool {
	return &b
}

func StringOption(s string) *string {
	return &s
}

func IntOption(i int64) *int64 {
	return &i
}
