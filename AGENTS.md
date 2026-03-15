# Fantasy Development Guide

## Commands

- **Build**: `go build ./...`
- **Test all**: `task test` or `go test ./... -count=1 -timeout=30m`
- **Test single**: `go test -run TestName ./package -v`
- **Lint**: `task lint` or `golangci-lint run`
- **Lint fix**: `task lint:fix`
- **Format**: `task fmt` (uses `gofumpt`, not `gofmt`)

## Style Notes

- Prefer `cmp.Or` for defaults
- Struct tags `json`, `description`, `enum` drive schema generation
- `charm.land/x/vcr` for HTTP test recording (not go-vcr)

## Project Layout

- `/` — Core package `fantasy`: Provider, LanguageModel, Agent, Content, Tool, errors, retry
- `/providers/{openai,anthropic,google,bedrock,azure,openrouter,openaicompat,vercel,kronk}`
- `/object` — Typed structured outputs: `object.Generate[T]`, `object.Stream[T]`
- `/schema`, `/jsonrepair` — JSON schema generation and repair utilities
- `/providertests` — Integration tests with VCR cassettes in `testdata/`

## Testing

- Env vars: `FANTASY_<PROVIDER>_API_KEY`, loaded from `.env` via godotenv
- VCR recorder injected as `http.Client` transport
