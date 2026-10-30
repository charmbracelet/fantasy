package openai

import (
	"regexp"
	"strings"
)

// reasoningModelPattern matches the model families known to reason: the
// o-series, gpt-oss, codex, computer-use, and gpt-5 onward. The generation
// stays on a single digit so Azure's gpt-35-turbo is not caught.
var reasoningModelPattern = regexp.MustCompile(`(?:^|[-.])(?:o[134]|oss)|gpt-[5-9]|codex-|computer-use`)

// nonReasoningModelPattern matches the OpenAI models known not to reason:
// gpt-3, gpt-4, and the chat snapshots used in ChatGPT.
var nonReasoningModelPattern = regexp.MustCompile(`gpt-3|gpt-4|chatgpt-|-chat`)

// isReasoningModel reports whether a Chat Completions model reasons. Chat
// Completions serves arbitrary OpenAI-compatible models, so only known
// reasoning families qualify.
func isReasoningModel(modelID string) bool {
	id := strings.ToLower(modelID)
	return reasoningModelPattern.MatchString(id) && !nonReasoningModelPattern.MatchString(id)
}

// isResponsesReasoningModel reports whether a Responses API model reasons.
// Every current OpenAI generation reasons, so only known exceptions don't.
func isResponsesReasoningModel(modelID string) bool {
	return !nonReasoningModelPattern.MatchString(strings.ToLower(modelID))
}
