module github.com/charmbracelet/ai/openai

go 1.24.5

require (
	github.com/charmbracelet/ai/ai v0.0.0-00010101000000-000000000000
	github.com/google/uuid v1.6.0
	github.com/openai/openai-go/v2 v2.2.1
	github.com/stretchr/testify v1.11.1
)

require (
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/go-viper/mapstructure/v2 v2.4.0 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	github.com/tidwall/gjson v1.18.0 // indirect
	github.com/tidwall/match v1.1.1 // indirect
	github.com/tidwall/pretty v1.2.1 // indirect
	github.com/tidwall/sjson v1.2.5 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)

replace github.com/charmbracelet/ai/ai => ../ai
