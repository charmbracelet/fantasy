# Bedrock

- Install the AWS CLI
- Log in in `aws configure`

To see available models, run:

```bash
aws bedrock list-inference-profiles --region us-east-1
```

## Amazon Nova Models via Bedrock

Fantasy supports Amazon Nova models through AWS Bedrock using the Converse API.
Crush catalogs **Nova 2 Lite** as the recommended Nova model. Fantasy still routes
any `amazon.*` model ID if you configure one manually.

### Recommended Model

- **amazon.nova-2-lite-v1:0** — Active Nova 2 model with extended thinking, 1M context,
  tool use, and multimodal input (text, image, video)

Nova 1 models (Pro, Lite, Micro, Premier) are no longer listed in Crush. Nova Premier is
[Legacy on Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html)
with EOL on September 14, 2026. AWS recommends migrating to Nova 2 Lite.

### Extended Thinking

Only **Nova 2 Lite** supports Bedrock extended thinking via `reasoningConfig`:

```go
call := fantasy.Call{
    Prompt: prompt,
    ProviderOptions: bedrock.NewProviderOptions(&bedrock.ProviderOptions{
        Thinking: &bedrock.ThinkingProviderOption{
            ReasoningEffort: bedrock.ReasoningEffortMedium, // low, medium, or high
        },
    }),
}
```

When Crush enables thinking for Bedrock models it passes Anthropic-style provider options
(`thinking.budget_tokens`). Fantasy converts those automatically for Nova models.

Reasoning output is returned as `reasoningContent` blocks in the Converse API response.
Nova 2 may redact reasoning text as `[REDACTED]` while still billing reasoning tokens.

### Model ID Format

Nova models use the Bedrock model identifier format: `amazon.nova-{variant}-v{version}:{revision}`

When you create a language model instance, Fantasy automatically applies the appropriate
region prefix (e.g., `us.amazon.nova-2-lite-v1:0` for us-east-1).

### AWS Credential Requirements

To use Nova models, you need AWS credentials configured. Fantasy uses the standard AWS SDK credential chain, which checks:

1. Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
2. AWS credentials file (`~/.aws/credentials`)
3. IAM role (when running on AWS infrastructure)
4. Bearer token (`AWS_BEARER_TOKEN_BEDROCK` for testing/development)

You also need to specify the AWS region:

- Set `AWS_REGION` environment variable (e.g., `us-east-1`)
- Or pass `bedrock.WithRegion("us-east-1")` when creating the provider

### Quick Example

```go
import "charm.land/fantasy/providers/bedrock"

// Create Bedrock provider
provider, err := bedrock.New(bedrock.WithRegion("us-east-1"))
if err != nil {
    fmt.Fprintln(os.Stderr, "Error:", err)
    os.Exit(1)
}

ctx := context.Background()

// Use Nova 2 Lite
model, err := provider.LanguageModel(ctx, "amazon.nova-2-lite-v1:0")
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
