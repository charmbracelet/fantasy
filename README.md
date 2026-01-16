# Fantasy

<p>
  <img width="475" alt="The Charm Fantasy logo" src="https://github.com/user-attachments/assets/b22c5862-792a-44c1-bc98-55a2e46c8fb9" /><br>
  <a href="https://github.com/charmbracelet/fantasy/releases"><img src="https://img.shields.io/github/release/charmbracelet/fantasy.svg" alt="Latest Release"></a>
  <a href="https://pkg.go.dev/charm.land/fantasy?tab=doc"><img src="https://godoc.org/charm.land/fantasy?status.svg" alt="GoDoc"></a>
  <a href="https://github.com/charmbracelet/fantasy/actions"><img src="https://github.com/charmbracelet/fantasy/actions/workflows/build.yml/badge.svg?branch=main" alt="Build Status"></a>
</p>

Build AI agents with Go. Multi-provider, multi-model, one API.

1. Choose a model and provider
2. Add some tools
3. Compile to native machine code and let it rip

> [!NOTE]
> Fantasy is currently a preview. Expect API changes.

```go
import "charm.land/fantasy"
import "charm.land/fantasy/providers/openrouter"

// Choose your fave provider.
provider, err := openrouter.New(openrouter.WithAPIKey(myHotKey))
if err != nil {
	fmt.Fprintln(os.Stderr, "Whoops:", err)
	os.Exit(1)
}

ctx := context.Background()

// Pick your fave model.
model, err := provider.LanguageModel(ctx, "moonshotai/kimi-k2")
if err != nil {
	fmt.Fprintln(os.Stderr, "Dang:", err)
	os.Exit(1)
}

// Make your own tools.
cuteDogTool := fantasy.NewAgentTool(
  "cute_dog_tool",
  "Provide up-to-date info on cute dogs.",
  fetchCuteDogInfoFunc,
)

// Equip your agent.
agent := fantasy.NewAgent(
  model,
  fantasy.WithSystemPrompt("You are a moderately helpful, dog-centric assistant."),
  fantasy.WithTools(cuteDogTool),
)

// Put that agent to work!
const prompt = "Find all the cute dogs in Silver Lake, Los Angeles."
result, err := agent.Generate(ctx, fantasy.AgentCall{Prompt: prompt})
if err != nil {
    fmt.Fprintln(os.Stderr, "Oof:", err)
    os.Exit(1)
}
fmt.Println(result.Response.Content.Text())
```

🍔 For the full implementation and more [see the examples directory](https://github.com/charmbracelet/fantasy/tree/main/examples).

## Multi-model? Multi-provider?

Yeah! Fantasy is designed to support a wide variety of providers and models under a single API. While many providers such as Microsoft Azure, Amazon Bedrock, and OpenRouter have dedicated packages in Fantasy, many others work just fine with `openaicompat`, the generic OpenAI-compatible layer. That said, if you find a provider that’s not compatible and needs special treatment, please let us know in an issue (or open a PR).


## Amazon Nova Models via Bedrock

Fantasy supports Amazon's Nova family of foundation models through AWS Bedrock. Nova models offer high-quality text generation with competitive pricing and performance.

### Supported Nova Models

- **amazon.nova-pro-v1:0** - High-performance model for complex tasks
- **amazon.nova-lite-v1:0** - Fast, cost-effective model for simpler tasks
- **amazon.nova-micro-v1:0** - Ultra-fast model for basic text generation
- **amazon.nova-premier-v1:0** - Most capable model with advanced reasoning

### Model ID Format

Nova models use the Bedrock model identifier format: `amazon.nova-{variant}-v{version}:{revision}`

When you create a language model instance, Fantasy automatically applies the appropriate region prefix (e.g., `us.amazon.nova-pro-v1:0` for us-east-1).

### AWS Credential Requirements

To use Nova models, you need AWS credentials configured. Fantasy uses the standard AWS SDK credential chain, which checks:

1. Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
2. AWS credentials file (`~/.aws/credentials`)
3. IAM role (when running on AWS infrastructure)
4. Bearer token (`AWS_BEARER_TOKEN_BEDROCK` for testing/development)

You also need to specify the AWS region:

- Set `AWS_REGION` environment variable (e.g., `us-east-1`)
- If not set, Fantasy defaults to `us-east-1`

### Quick Example

```go
import "charm.land/fantasy/providers/bedrock"

// Create Bedrock provider
provider, err := bedrock.New()
if err != nil {
    fmt.Fprintln(os.Stderr, "Error:", err)
    os.Exit(1)
}

ctx := context.Background()

// Use a Nova model
model, err := provider.LanguageModel(ctx, "amazon.nova-pro-v1:0")
if err != nil {
    fmt.Fprintln(os.Stderr, "Error:", err)
    os.Exit(1)
}

// Generate text
agent := fantasy.NewAgent(model,
    fantasy.WithSystemPrompt("You are a helpful assistant."),
)

result, err := agent.Generate(ctx, fantasy.AgentCall{
    Prompt: "Explain quantum computing in simple terms.",
})
if err != nil {
    fmt.Fprintln(os.Stderr, "Error:", err)
    os.Exit(1)
}
fmt.Println(result.Response.Content.Text())
```
## Work in Progress

We built Fantasy to power [Crush](https://github.com/charmbracelet/crush), a hot coding agent for glamourously invincible development. Given that, Fantasy does not yet support things like:

- Image models
- Audio models
- PDF uploads
- Provider tools (e.g. web_search)

For things you’d like to see supported, PRs are welcome.

## Whatcha think?

We’d love to hear your thoughts on this project. Need help? We gotchu. You can find us on:

- [Slack](https://charm.land/slack)
- [Discord][discord]
- [Twitter](https://twitter.com/charmcli)
- [The Fediverse](https://mastodon.social/@charmcli)
- [Bluesky](https://bsky.app/profile/charm.land)

[discord]: https://charm.land/discord

---

Part of [Charm](https://charm.land).

<a href="https://charm.land/"><img alt="The Charm logo" src="https://stuff.charm.sh/charm-banner-next.jpg" width="400"></a>

Charm热爱开源 • Charm loves open source
