package openrouter

import (
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"strings"
)

// GeminiCompatibilityTransport wraps an HTTP client to make requests compatible
// with Google Gemini models accessed through OpenRouter.
//
// Google Gemini models use "user" role for tool/function call results, while
// OpenRouter's API spec (following OpenAI's standard) uses "tool" role.
// OpenRouter does not automatically transform this for Gemini models, so we
// need to handle it client-side.
//
// References:
// - OpenRouter API: https://openrouter.ai/docs/features/tool-calling (uses "tool" role)
// - Gemini API: https://ai.google.dev/gemini-api/docs/function-calling (uses "user" role)
type GeminiCompatibilityTransport struct {
	Base *http.Client
}

// Do intercepts HTTP requests to OpenRouter and transforms message roles
// for Gemini model compatibility.
func (t *GeminiCompatibilityTransport) Do(req *http.Request) (*http.Response, error) {
	// Only process POST requests with a body (chat completion requests)
	if req.Method != "POST" || req.Body == nil {
		return t.Base.Do(req)
	}

	// Read the request body
	bodyBytes, err := io.ReadAll(req.Body)
	req.Body.Close()
	if err != nil {
		return t.Base.Do(req)
	}

	// Parse as JSON
	var requestData map[string]any
	if err := json.Unmarshal(bodyBytes, &requestData); err != nil {
		// Not JSON or malformed - pass through unchanged
		req.Body = io.NopCloser(bytes.NewReader(bodyBytes))
		return t.Base.Do(req)
	}

	// Check if this is a Gemini model
	isGemini := false
	if model, ok := requestData["model"].(string); ok {
		modelLower := strings.ToLower(model)
		isGemini = strings.Contains(modelLower, "gemini")
	}

	// Only apply transformation for Gemini models
	if !isGemini {
		req.Body = io.NopCloser(bytes.NewReader(bodyBytes))
		return t.Base.Do(req)
	}

	// Convert role: "tool" to role: "user" for Gemini compatibility
	if messages, ok := requestData["messages"].([]any); ok {
		toolRoleConverted := 0
		for _, msg := range messages {
			if msgMap, ok := msg.(map[string]any); ok {
				if role, ok := msgMap["role"].(string); ok && role == "tool" {
					msgMap["role"] = "user"
					toolRoleConverted++
				}
			}
		}

		if toolRoleConverted > 0 {
			slog.Debug("Converted tool role to user for Gemini compatibility",
				"count", toolRoleConverted,
				"model", requestData["model"])

			// Re-serialize with changes
			modifiedBytes, err := json.Marshal(requestData)
			if err == nil {
				bodyBytes = modifiedBytes
				req.ContentLength = int64(len(modifiedBytes))
				req.Header.Set("Content-Length", string(rune(len(modifiedBytes))))
			}
		}
	}

	// Restore body with potentially modified content
	req.Body = io.NopCloser(bytes.NewReader(bodyBytes))
	return t.Base.Do(req)
}
