# Bedrock

- Install the AWS CLI
- Log in with `aws login`

To see available models, run:

```bash
aws bedrock list-inference-profiles --region us-east-1
```

## Models

- Anthropic models (`us.anthropic.claude-...`) use the Anthropic Messages API.
- OpenAI GPT models (`us.openai.gpt-6-sol`, `global.openai.gpt-5.6-sol`, ...)
  use the Responses API on the `bedrock-runtime` OpenAI-compatible endpoint.
  Bedrock requires a cross-Region inference profile ID for these.
- `gpt-oss` models (`openai.gpt-oss-120b-1:0`) use Chat Completions, the only
  OpenAI API Bedrock serves them on.

Provider options for OpenAI models are read from the `openai` key, the same
as with the OpenAI provider.
