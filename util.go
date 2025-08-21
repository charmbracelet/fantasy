package ai

import (
	"encoding/json"

	"github.com/go-viper/mapstructure/v2"
)

func ParseOptions[T any](options map[string]any, m *T) error {
	return mapstructure.Decode(options, m)
}

func FloatOption(f float64) *float64 {
	return &f
}

func IsParsableJSON(data string) bool {
	var m map[string]any
	err := json.Unmarshal([]byte(data), &m)
	return err == nil
}
