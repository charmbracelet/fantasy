# Bedrock

- Install the AWS CLI
- Log in in `aws configure`

To see available models, run:

```bash
aws bedrock list-inference-profiles --region us-east-1
```

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
