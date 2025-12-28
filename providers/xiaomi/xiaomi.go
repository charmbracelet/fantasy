// Package xiaomi provides a fantasy.Provider for Xiaomi API.
package xiaomi

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"strings"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openaicompat"
	"charm.land/fantasy/providers/openai"
	openaisdk "github.com/openai/openai-go/v2"
)

const (
	// Name is the provider type name for Xiaomi.
	Name = "xiaomi"
)

type xiaomiProvider struct {
	fantasy.Provider
	extraBody map[string]any
	thinking  bool
}

type options struct {
	baseURL    string
	apiKey     string
	headers    map[string]string
	httpClient *http.Client
	extraBody  map[string]any
	thinking   bool
}

// Option configures the Xiaomi provider.
type Option = func(*options)

// WithBaseURL sets the base URL for the Xiaomi provider.
func WithBaseURL(baseURL string) Option {
	return func(o *options) {
		o.baseURL = baseURL
	}
}

// WithAPIKey sets the API key for the Xiaomi provider.
func WithAPIKey(apiKey string) Option {
	return func(o *options) {
		o.apiKey = apiKey
	}
}

// WithHeaders sets the headers for the Xiaomi provider.
func WithHeaders(headers map[string]string) Option {
	return func(o *options) {
		o.headers = headers
	}
}

// WithHTTPClient sets the HTTP client for the Xiaomi provider.
func WithHTTPClient(httpClient *http.Client) Option {
	return func(o *options) {
		o.httpClient = httpClient
	}
}

// WithExtraBody sets the extra body parameters for the Xiaomi provider.
func WithExtraBody(extraBody map[string]any) Option {
	return func(o *options) {
		o.extraBody = extraBody
	}
}

// WithThinking enables or disables thinking mode for the Xiaomi provider.
func WithThinking(enabled bool) Option {
	return func(o *options) {
		o.thinking = enabled
	}
}

const (
	xiaomiToolCallPrefix = "xiaomi_tool_calls"
)

// xiaomiToolCall represents a parsed Xiaomi XML tool call
type xiaomiToolCall struct {
	name      string
	arguments string
}

// xiaomiPrepareCallFunc adds Xiaomi-specific parameters to the request
func xiaomiPrepareCallFunc(opts *options) openai.LanguageModelPrepareCallFunc {
	return func(model fantasy.LanguageModel, params *openaisdk.ChatCompletionNewParams, call fantasy.Call) ([]fantasy.CallWarning, error) {
		// First delegate to openaicompat's default prepare
		warnings, err := openaicompat.PrepareCallFunc(model, params, call)
		if err != nil {
			return warnings, err
		}

		// Create extra fields map
		extraFields := make(map[string]any)

		// Add thinking parameter if enabled
		if opts.thinking {
			extraFields["thinking"] = map[string]any{
				"type": "enabled",
			}
		}

		// Add extra body parameters
		if len(opts.extraBody) > 0 {
			for k, v := range opts.extraBody {
				extraFields[k] = v
			}
		}

		// Set extra fields if any
		if len(extraFields) > 0 {
			params.SetExtraFields(extraFields)
		}

		return warnings, nil
	}
}

// xiaomiStreamExtraFunc handles Xiaomi-specific streaming responses, including XML tool call format
func xiaomiStreamExtraFunc(chunk openaisdk.ChatCompletionChunk, yield func(fantasy.StreamPart) bool, ctx map[string]any) (map[string]any, bool) {
	if len(chunk.Choices) == 0 {
		return ctx, true
	}

	for inx, choice := range chunk.Choices {
		// Check for Xiaomi XML tool calls in content
		if choice.Delta.Content != "" && strings.Contains(choice.Delta.Content, "<function=") {
			// Use a wrapper yield that suppresses content when in tool call mode
			return parseXiaomiToolCalls(chunk, wrapYieldToSuppressContent(yield, ctx), ctx, inx)
		}

		// Check for tool calls in the standard ToolCalls field
		// Xiaomi might return tool calls in the standard format with wrapper function names
		if len(choice.Delta.ToolCalls) > 0 {
			return processXiaomiToolCalls(chunk, wrapYieldToExtractToolName(yield), ctx, inx)
		}
	}

	// Delegate to default openaicompat behavior for non-tool-call content
	return openaicompat.StreamExtraFunc(chunk, yield, ctx)
}

// wrapYieldToSuppressContent creates a yield wrapper that suppresses content events when in tool call mode
func wrapYieldToSuppressContent(yield func(fantasy.StreamPart) bool, ctx map[string]any) func(fantasy.StreamPart) bool {
	return func(sp fantasy.StreamPart) bool {
		// Suppress content events when we're processing tool calls
		if sp.Type == fantasy.StreamPartTypeTextDelta {
			// Check if we're in tool call mode by looking at accumulated content
			accumulatedKey := xiaomiToolCallPrefix + "_content"
			if accumulated, ok := ctx[accumulatedKey].(string); ok && accumulated != "" {
				// We're in tool call mode, suppress this content
				return true
			}
		}
		return yield(sp)
	}
}

// wrapYieldToExtractToolName creates a yield wrapper that extracts the actual tool name from wrapper functions
func wrapYieldToExtractToolName(yield func(fantasy.StreamPart) bool) func(fantasy.StreamPart) bool {
	return func(sp fantasy.StreamPart) bool {
		// Only process tool call events
		if sp.Type == fantasy.StreamPartTypeToolCall && sp.ToolCallName != "" {
			// Check if this is a wrapper function
			if sp.ToolCallName == "editor" || sp.ToolCallName == "bash" || sp.ToolCallName == "agent" {
				// Parse the arguments to extract the command parameter
				var argsMap map[string]string
				if err := json.Unmarshal([]byte(sp.ToolCallInput), &argsMap); err == nil {
					if command, ok := argsMap["command"]; ok {
						// Use command parameter as tool name
						sp.ToolCallName = command
						// Remove command from arguments
						delete(argsMap, "command")
						if newArgs, err := json.Marshal(argsMap); err == nil {
							sp.ToolCallInput = string(newArgs)
						}
					}
				}
			}
		}
		return yield(sp)
	}
}

// processXiaomiToolCalls processes tool calls from the standard ToolCalls field
func processXiaomiToolCalls(chunk openaisdk.ChatCompletionChunk, yield func(fantasy.StreamPart) bool, ctx map[string]any, inx int) (map[string]any, bool) {
	// Delegate to default openaicompat behavior, but with tool name extraction
	return openaicompat.StreamExtraFunc(chunk, yield, ctx)
}

// parseXiaomiToolCalls parses Xiaomi's XML tool call format and emits standard tool call events
func parseXiaomiToolCalls(chunk openaisdk.ChatCompletionChunk, yield func(fantasy.StreamPart) bool, ctx map[string]any, inx int) (map[string]any, bool) {
	content := chunk.Choices[0].Delta.Content

	// Accumulate content across chunks
	accumulatedKey := xiaomiToolCallPrefix + "_content"
	accumulated, _ := ctx[accumulatedKey].(string)
	accumulated += content
	ctx[accumulatedKey] = accumulated

	// Try to parse complete tool calls
	toolCalls, remainingContent, err := extractXiaomiToolCalls(accumulated)
	if err != nil {
		yield(fantasy.StreamPart{
			Type:  fantasy.StreamPartTypeError,
			Error: &fantasy.Error{Title: "parse error", Message: "error parsing Xiaomi tool calls", Cause: err},
		})
		return ctx, false
	}

	// Only update context with remaining content if we found tool calls
	// This ensures we keep accumulating when no complete tool calls are found yet
	if len(toolCalls) > 0 {
		ctx[accumulatedKey] = remainingContent
	}

	// Emit tool call events for each parsed tool
	for _, tc := range toolCalls {
		// Xiaomi uses wrapper functions (e.g., "editor") where the actual tool name
		// is in the "command" parameter
		toolName := tc.name
		toolArgs := tc.arguments

		// Parse arguments to extract command if present
		var argsMap map[string]string
		if err := json.Unmarshal([]byte(tc.arguments), &argsMap); err == nil {
			if command, ok := argsMap["command"]; ok && (tc.name == "editor" || tc.name == "bash" || tc.name == "agent") {
				// Use command parameter as tool name for wrapper functions
				toolName = command
				// Remove command from arguments
				delete(argsMap, "command")
				if newArgs, err := json.Marshal(argsMap); err == nil {
					toolArgs = string(newArgs)
				}
			}
		}

		toolCallID := fmt.Sprintf("xiaomi_%d_%s", inx, toolName)

		// Emit tool input start
		if !yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeToolInputStart,
			ID:   toolCallID,
		}) {
			return ctx, false
		}

		// Emit tool input delta (the arguments)
		if !yield(fantasy.StreamPart{
			Type:  fantasy.StreamPartTypeToolInputDelta,
			ID:    toolCallID,
			Delta: toolArgs,
		}) {
			return ctx, false
		}

		// Emit tool input end
		if !yield(fantasy.StreamPart{
			Type: fantasy.StreamPartTypeToolInputEnd,
			ID:   toolCallID,
		}) {
			return ctx, false
		}

		// Emit tool call
		if !yield(fantasy.StreamPart{
			Type:          fantasy.StreamPartTypeToolCall,
			ID:            toolCallID,
			ToolCallName:  toolName,
			ToolCallInput: toolArgs,
		}) {
			return ctx, false
		}
	}

	return ctx, true
}

// extractXiaomiToolCalls extracts complete tool calls from accumulated content
func extractXiaomiToolCalls(content string) ([]xiaomiToolCall, string, error) {
	var toolCalls []xiaomiToolCall

	// Pattern to match <function=name>...</function> (with DOTALL flag to match newlines)
	funcPattern := regexp.MustCompile(`(?s)<function=([^>]+)>(.*?)</function>`)
	paramsPattern := regexp.MustCompile(`<parameter=([^>]+)>([^<]*)</parameter>`)

	// Find all complete tool calls
	matches := funcPattern.FindAllStringSubmatchIndex(content, -1)

	// If no matches, return the original content as remaining
	if len(matches) == 0 {
		return toolCalls, content, nil
	}

	// Process each complete tool call
	for _, match := range matches {
		if match[1] == -1 || match[2] == -1 {
			continue
		}

		funcName := content[match[2]:match[3]]
		funcBody := content[match[4]:match[5]]

		// Parse parameters
		params := make(map[string]string)
		paramMatches := paramsPattern.FindAllStringSubmatch(funcBody, -1)
		for _, pm := range paramMatches {
			if len(pm) == 3 {
				params[pm[1]] = pm[2]
			}
		}

		// Convert to JSON arguments
		argsJSON, err := json.Marshal(params)
		if err != nil {
			return nil, "", fmt.Errorf("failed to marshal parameters: %w", err)
		}

		toolCalls = append(toolCalls, xiaomiToolCall{
			name:      funcName,
			arguments: string(argsJSON),
		})
	}

	// Remaining content is after the last complete tool call
	lastMatch := matches[len(matches)-1]
	remaining := content[lastMatch[1]:]

	return toolCalls, remaining, nil
}

// New creates a new Xiaomi provider.
func New(opts ...Option) (fantasy.Provider, error) {
	o := options{
		baseURL:   "https://api.xiaomimimo.com/v1",
		headers:   make(map[string]string),
		extraBody: make(map[string]any),
	}
	for _, opt := range opts {
		opt(&o)
	}

	// Build OpenAI-compatible provider with Xiaomi-specific configuration
	openaiOpts := []openaicompat.Option{
		openaicompat.WithBaseURL(o.baseURL),
		openaicompat.WithAPIKey(o.apiKey),
	}

	if len(o.headers) > 0 {
		openaiOpts = append(openaiOpts, openaicompat.WithHeaders(o.headers))
	}

	if o.httpClient != nil {
		openaiOpts = append(openaiOpts, openaicompat.WithHTTPClient(o.httpClient))
	}

	// Override PrepareCallFunc to add Xiaomi-specific parameters (thinking, extra body)
	openaiOpts = append(openaiOpts, openaicompat.WithLanguageModelOption(openai.WithLanguageModelPrepareCallFunc(xiaomiPrepareCallFunc(&o))))

	// Override StreamExtraFunc to handle Xiaomi's XML tool call format
	openaiOpts = append(openaiOpts, openaicompat.WithLanguageModelOption(openai.WithLanguageModelStreamExtraFunc(xiaomiStreamExtraFunc)))

	provider, err := openaicompat.New(openaiOpts...)
	if err != nil {
		return nil, err
	}

	// Wrap provider to filter content before streaming
	return &xiaomiProviderWrapper{
		Provider:   provider,
		xiaomiOpts: &o,
	}, nil
}

// xiaomiProviderWrapper wraps the OpenAI-compatible provider to filter XML tool call content
type xiaomiProviderWrapper struct {
	fantasy.Provider
	xiaomiOpts *options
}

// LanguageModel implements fantasy.Provider by wrapping the language model with content filtering
func (w *xiaomiProviderWrapper) LanguageModel(ctx context.Context, modelID string) (fantasy.LanguageModel, error) {
	lm, err := w.Provider.LanguageModel(ctx, modelID)
	if err != nil {
		return nil, err
	}

	return &xiaomiLanguageModel{
		LanguageModel: lm,
		xiaomiOpts:    w.xiaomiOpts,
	}, nil
}

// xiaomiLanguageModel wraps the language model to filter streaming content
type xiaomiLanguageModel struct {
	fantasy.LanguageModel
	xiaomiOpts *options
}

// Stream implements fantasy.LanguageModel by filtering XML tool call content
func (m *xiaomiLanguageModel) Stream(ctx context.Context, call fantasy.Call) (fantasy.StreamResponse, error) {
	stream, err := m.LanguageModel.Stream(ctx, call)
	if err != nil {
		return nil, err
	}
	return xiaomiFilteredStream(stream), nil
}

// xiaomiFilteredStream creates a filtered stream that suppresses tool call content
func xiaomiFilteredStream(original fantasy.StreamResponse) fantasy.StreamResponse {
	return func(yield func(fantasy.StreamPart) bool) {
		// Track accumulated content to detect tool calls
		var accumulatedContent strings.Builder
		var inToolCallMode bool

		// Create a wrapper yield that filters content
		filteredYield := func(sp fantasy.StreamPart) bool {
			// Handle content deltas
			if sp.Type == fantasy.StreamPartTypeTextDelta && sp.Delta != "" {
				// Check if this content contains tool call markers
				if strings.Contains(sp.Delta, "<function=") {
					inToolCallMode = true
					accumulatedContent.WriteString(sp.Delta)
					// Don't yield this content
					return true
				}

				// If we're in tool call mode, accumulate but don't yield
				if inToolCallMode {
					accumulatedContent.WriteString(sp.Delta)
					return true
				}
			}

			// Handle finish_reason - reset tool call mode on stream end
			if sp.Type == fantasy.StreamPartTypeFinish {
				inToolCallMode = false
				accumulatedContent.Reset()
			}

			// Handle error - reset tool call mode
			if sp.Type == fantasy.StreamPartTypeError {
				inToolCallMode = false
				accumulatedContent.Reset()
			}

			// Yield all other stream parts
			return yield(sp)
		}

		// Call original stream with filtered yield
		original(filteredYield)
	}
}