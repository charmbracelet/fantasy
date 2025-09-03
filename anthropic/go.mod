module github.com/charmbracelet/ai/anthropic

go 1.24.5

require (
	github.com/anthropics/anthropic-sdk-go v1.10.0
	github.com/charmbracelet/ai/ai v0.0.0-00010101000000-000000000000
)

require (
	github.com/go-viper/mapstructure/v2 v2.4.0 // indirect
	github.com/tidwall/gjson v1.18.0 // indirect
	github.com/tidwall/match v1.1.1 // indirect
	github.com/tidwall/pretty v1.2.1 // indirect
	github.com/tidwall/sjson v1.2.5 // indirect
)

replace github.com/charmbracelet/ai/ai => ../ai
