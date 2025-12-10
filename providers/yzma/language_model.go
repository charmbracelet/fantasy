package yzma

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"

	"charm.land/fantasy"
	"charm.land/fantasy/object"
	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/message"
	"github.com/hybridgroup/yzma/pkg/template"
)

const (
	defaultTemperature       = 0.8
	defaultTopK        int32 = 40
	defaultTopP              = 0.9
)

type yzmaModel struct {
	provider string
	modelID  string

	model   llama.Model
	context llama.Context
	vocab   llama.Vocab
}

func newModel(modelID string, modelsPath string) (fantasy.LanguageModel, error) {
	filePath, err := ensureModelExists(modelID, modelsPath)
	if err != nil {
		return nil, fmt.Errorf("model file does not exist: %w", err)
	}

	model, err := llama.ModelLoadFromFile(filePath, llama.ModelDefaultParams())
	if err != nil {
		return nil, err
	}

	// TODO: allow for setting any passed context options
	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = uint32(4096)
	ctxParams.NBatch = uint32(2048)

	lctx, err := llama.InitFromModel(model, ctxParams)
	if err != nil {
		llama.ModelFree(model)
		return nil, err
	}

	vocab := llama.ModelGetVocab(model)

	return &yzmaModel{
		provider: Name,
		modelID:  modelID,
		model:    model,
		context:  lctx,
		vocab:    vocab,
	}, nil
}

func (m *yzmaModel) Close() {
	if m.context != 0 {
		llama.Free(m.context)
	}

	if m.model != 0 {
		llama.ModelFree(m.model)
	}
}

// Generate calls the language model to generate a response.
func (m *yzmaModel) Generate(ctx context.Context, call fantasy.Call) (*fantasy.Response, error) {
	// Clear cache before each generation
	mem, _ := llama.GetMemory(m.context)
	llama.MemoryClear(mem, true)

	sampler := initSampler(m.model, call)
	messages := convertMessageContent(call.Prompt)

	// Convert tools to message format for the prompt
	// We need to add tool definitions as a system message so the model knows what tools are available
	if len(call.Tools) > 0 {
		toolDefs := buildToolDefinitions(call.Tools)
		// Prepend tool definitions as a system message
		messages = append([]message.Message{
			message.Chat{
				Role:    "system",
				Content: toolDefs,
			},
		}, messages...)
	}

	tmpl := templateForModel(m.model)
	msg := chatTemplate(tmpl, messages, true)

	// call once to get the size of the tokens from the prompt
	tokens := llama.Tokenize(m.vocab, msg, true, true)
	batch := llama.BatchGetOne(tokens)

	result := decodeResults(m.context, m.vocab, batch, sampler, int32(*call.MaxOutputTokens))

	// Parse tool calls from the response
	toolCalls := parseToolCalls(result)

	content := make([]fantasy.Content, 0)

	if len(toolCalls) > 0 {
		// Return tool calls
		for i, tc := range toolCalls {
			content = append(content, fantasy.ToolCallContent{
				ProviderExecuted: false,
				ToolCallID:       strconv.Itoa(i),
				ToolName:         tc.Name,
				Input:            toJSONString(tc.Arguments),
			})
		}
		return &fantasy.Response{
			Content:      content,
			FinishReason: fantasy.FinishReasonToolCalls,
		}, nil
	}

	// No tool calls, return text
	content = append(content, fantasy.TextContent{Text: result})

	response := &fantasy.Response{
		Content:      content,
		FinishReason: fantasy.FinishReasonStop,
	}

	return response, nil
}

func (m *yzmaModel) GenerateObject(ctx context.Context, call fantasy.ObjectCall) (*fantasy.ObjectResponse, error) {
	// For simplicity, we treat object generation the same as text generation.
	resp, err := m.Generate(ctx, fantasy.Call{
		Prompt:          call.Prompt,
		MaxOutputTokens: call.MaxOutputTokens,
		Temperature:     call.Temperature,
		TopP:            call.TopP,
		TopK:            call.TopK,
	})
	if err != nil {
		return nil, err
	}

	return &fantasy.ObjectResponse{
		Object: resp.Content,
	}, nil
}

func (m *yzmaModel) Stream(ctx context.Context, call fantasy.Call) (fantasy.StreamResponse, error) {
	// Clear cache before each generation
	mem, _ := llama.GetMemory(m.context)
	llama.MemoryClear(mem, true)

	sampler := initSampler(m.model, call)
	tmpl := templateForModel(m.model)

	messages := convertMessageContent(call.Prompt)

	// Convert tools to message format for the prompt
	hasTools := len(call.Tools) > 0
	if hasTools {
		toolDefs := buildToolDefinitions(call.Tools)
		// Prepend tool definitions as a system message
		messages = append([]message.Message{
			message.Chat{
				Role:    "system",
				Content: toolDefs,
			},
		}, messages...)
	}

	return func(yield func(fantasy.StreamPart) bool) {
		prompt, err := template.Apply(tmpl, messages, true)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to apply template: %v\n", err)
			return
		}

		tokens := llama.Tokenize(m.vocab, prompt, true, true)
		batch := llama.BatchGetOne(tokens)

		maxTokens := int32(2048)
		if call.MaxOutputTokens != nil {
			maxTokens = int32(*call.MaxOutputTokens)
		}

		// If no tools, stream directly without buffering
		if !hasTools {
			if !yield(fantasy.StreamPart{
				Type: fantasy.StreamPartTypeTextStart,
				ID:   "0",
			}) {
				return
			}

			for pos := int32(0); pos < maxTokens; pos += batch.NTokens {
				llama.Decode(m.context, batch)
				token := llama.SamplerSample(sampler, m.context, -1)

				if llama.VocabIsEOG(m.vocab, token) {
					break
				}

				buf := make([]byte, 64)
				length := llama.TokenToPiece(m.vocab, token, buf, 0, true)
				delta := string(buf[:length])

				if !yield(fantasy.StreamPart{
					Type:  fantasy.StreamPartTypeTextDelta,
					ID:    "0",
					Delta: delta,
				}) {
					return
				}

				batch = llama.BatchGetOne([]llama.Token{token})
			}

			if !yield(fantasy.StreamPart{
				Type: fantasy.StreamPartTypeTextEnd,
				ID:   "0",
			}) {
				return
			}

			yield(fantasy.StreamPart{
				Type:         fantasy.StreamPartTypeFinish,
				FinishReason: fantasy.FinishReasonStop,
				Usage:        fantasy.Usage{},
			})
			return
		}

		// With tools: stream but also buffer to detect tool calls
		// We use a small lookahead buffer to detect JSON start
		var fullResponse strings.Builder
		var pendingDeltas []string
		textStarted := false
		inPotentialToolCall := false

		for pos := int32(0); pos < maxTokens; pos += batch.NTokens {
			llama.Decode(m.context, batch)
			token := llama.SamplerSample(sampler, m.context, -1)

			if llama.VocabIsEOG(m.vocab, token) {
				break
			}

			buf := make([]byte, 64)
			length := llama.TokenToPiece(m.vocab, token, buf, 0, true)
			delta := string(buf[:length])

			fullResponse.WriteString(delta)
			currentResponse := fullResponse.String()

			// Check if response looks like it might contain tool calls
			trimmed := strings.TrimSpace(currentResponse)
			looksLikeToolCall := strings.HasPrefix(trimmed, "{") &&
				(strings.Contains(trimmed, "\"name\"") || len(trimmed) < 20)

			if looksLikeToolCall && !textStarted {
				// Buffer tokens until we know if it's a tool call
				inPotentialToolCall = true
				pendingDeltas = append(pendingDeltas, delta)
			} else if inPotentialToolCall {
				// Still buffering, check if we can determine yet
				pendingDeltas = append(pendingDeltas, delta)

				// If we see "arguments", it's definitely a tool call - keep buffering
				if strings.Contains(trimmed, "\"arguments\"") {
					continue
				}

				// If we see enough non-JSON content, it's not a tool call
				if len(trimmed) > 50 && !strings.Contains(trimmed, "\"name\"") {
					// Flush pending deltas as text
					if !textStarted {
						if !yield(fantasy.StreamPart{
							Type: fantasy.StreamPartTypeTextStart,
							ID:   "0",
						}) {
							return
						}
						textStarted = true
					}
					for _, pd := range pendingDeltas {
						if !yield(fantasy.StreamPart{
							Type:  fantasy.StreamPartTypeTextDelta,
							ID:    "0",
							Delta: pd,
						}) {
							return
						}
					}
					pendingDeltas = nil
					inPotentialToolCall = false
				}
			} else {
				// Normal text streaming
				if !textStarted {
					if !yield(fantasy.StreamPart{
						Type: fantasy.StreamPartTypeTextStart,
						ID:   "0",
					}) {
						return
					}
					textStarted = true
				}

				if !yield(fantasy.StreamPart{
					Type:  fantasy.StreamPartTypeTextDelta,
					ID:    "0",
					Delta: delta,
				}) {
					return
				}
			}

			batch = llama.BatchGetOne([]llama.Token{token})
		}

		// Generation complete - check if we have tool calls
		response := fullResponse.String()
		toolCalls := parseToolCalls(response)

		if len(toolCalls) > 0 {
			// Emit tool calls (don't emit any text that was buffered)
			for i, tc := range toolCalls {
				if !yield(fantasy.StreamPart{
					Type:         fantasy.StreamPartTypeToolInputStart,
					ID:           strconv.Itoa(i),
					ToolCallName: tc.Name,
				}) {
					return
				}

				argsJSON := toJSONString(tc.Arguments)
				if !yield(fantasy.StreamPart{
					Type:  fantasy.StreamPartTypeToolInputDelta,
					ID:    strconv.Itoa(i),
					Delta: argsJSON,
				}) {
					return
				}

				if !yield(fantasy.StreamPart{
					Type: fantasy.StreamPartTypeToolInputEnd,
					ID:   strconv.Itoa(i),
				}) {
					return
				}

				if !yield(fantasy.StreamPart{
					Type:          fantasy.StreamPartTypeToolCall,
					ID:            strconv.Itoa(i),
					ToolCallName:  tc.Name,
					ToolCallInput: argsJSON,
				}) {
					return
				}
			}

			yield(fantasy.StreamPart{
				Type:         fantasy.StreamPartTypeFinish,
				FinishReason: fantasy.FinishReasonToolCalls,
				Usage:        fantasy.Usage{},
			})
			return
		}

		// No tool calls - flush any pending deltas
		if len(pendingDeltas) > 0 {
			if !textStarted {
				if !yield(fantasy.StreamPart{
					Type: fantasy.StreamPartTypeTextStart,
					ID:   "0",
				}) {
					return
				}
				textStarted = true
			}
			for _, pd := range pendingDeltas {
				if !yield(fantasy.StreamPart{
					Type:  fantasy.StreamPartTypeTextDelta,
					ID:    "0",
					Delta: pd,
				}) {
					return
				}
			}
		}

		if textStarted {
			if !yield(fantasy.StreamPart{
				Type: fantasy.StreamPartTypeTextEnd,
				ID:   "0",
			}) {
				return
			}
		}

		yield(fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeFinish,
			FinishReason: fantasy.FinishReasonStop,
			Usage:        fantasy.Usage{},
		})
	}, nil
}

func (m *yzmaModel) StreamObject(ctx context.Context, call fantasy.ObjectCall) (fantasy.ObjectStreamResponse, error) {
	return object.StreamWithText(ctx, m, call)
}

func (m *yzmaModel) Provider() string {
	return m.provider
}

func (m *yzmaModel) Model() string {
	return m.modelID
}

func initSampler(model llama.Model, call fantasy.Call) llama.Sampler {
	temperature := defaultTemperature
	if call.Temperature != nil && *call.Temperature > 0 {
		temperature = *call.Temperature
	}
	topK := defaultTopK
	if call.TopK != nil && *call.TopK > 0 {
		topK = int32(*call.TopK)
	}

	minP := 0.1

	topP := defaultTopP
	if call.TopP != nil && *call.TopP > 0 {
		topP = *call.TopP
	}

	sp := llama.DefaultSamplerParams()
	sp.Temp = float32(temperature)
	sp.TopK = topK
	sp.TopP = float32(topP)
	sp.MinP = float32(minP)
	sp.Seed = llama.DefaultSeed

	sampler := llama.NewSampler(model, llama.DefaultSamplers, sp)

	return sampler
}

func templateForModel(model llama.Model) string {
	return llama.ModelChatTemplate(model, "")
}

func chatTemplate(tmpl string, msgs []message.Message, add bool) string {
	prompt, err := template.Apply(tmpl, msgs, true)
	if err != nil {
		return ""
	}
	return prompt
}

func convertMessageContent(prompt fantasy.Prompt) []message.Message {
	chatMsgs := []message.Message{}
	for _, m := range prompt {
		for _, p := range m.Content {
			switch p.GetType() {
			case fantasy.ContentTypeText:
				text, _ := fantasy.AsMessagePart[fantasy.TextPart](p)
				chatMsgs = append(chatMsgs, message.Chat{
					Role:    string(m.Role),
					Content: text.Text,
				})
			case fantasy.ContentTypeToolCall:
				toolCall, _ := fantasy.AsMessagePart[fantasy.ToolCallPart](p)
				// Format tool call as a message the model can understand
				toolCallMap := map[string]string{}
				// Assuming the input is a JSON string, we can parse it into a map

				_ = json.Unmarshal([]byte(toolCall.Input), &toolCallMap)
				chatMsgs = append(chatMsgs, message.Tool{
					Role: string(m.Role),
					ToolCalls: []message.ToolCall{{
						Type:     "function",
						Function: message.ToolFunction{Name: toolCall.ToolName, Arguments: toolCallMap},
					}},
				})
			case fantasy.ContentTypeToolResult:
				toolResult, _ := fantasy.AsMessagePart[fantasy.ToolResultPart](p)
				var resultText string
				switch output := toolResult.Output.(type) {
				case fantasy.ToolResultOutputContentText:
					resultText = output.Text
				case fantasy.ToolResultOutputContentError:
					resultText = output.Error.Error()
				}
				chatMsgs = append(chatMsgs, message.ToolResponse{
					Role:    string(m.Role),
					Name:    toolResult.ToolCallID,
					Content: resultText,
				})
			}
		}
	}
	return chatMsgs
}

func decodeResults(lctx llama.Context, vocab llama.Vocab, batch llama.Batch, sampler llama.Sampler, maxTokens int32) string {
	result := ""

	for pos := int32(0); pos < maxTokens; pos += batch.NTokens {
		llama.Decode(lctx, batch)
		token := llama.SamplerSample(sampler, lctx, -1)

		if llama.VocabIsEOG(vocab, token) {
			break
		}

		buf := make([]byte, 64)
		len := llama.TokenToPiece(vocab, token, buf, 0, true)

		result = result + string(buf[:len])
		batch = llama.BatchGetOne([]llama.Token{token})
	}

	return result
}

func toJSONString(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return "{}"
	}
	return string(b)
}

// buildToolDefinitions creates a system prompt describing available tools
func buildToolDefinitions(tools []fantasy.Tool) string {
	type toolDef struct {
		Type     string `json:"type"`
		Function struct {
			Name        string         `json:"name"`
			Description string         `json:"description"`
			Parameters  map[string]any `json:"parameters"`
		} `json:"function"`
	}

	var defs []toolDef
	for _, tool := range tools {
		if ft, ok := tool.(fantasy.FunctionTool); ok {
			def := toolDef{Type: "function"}
			def.Function.Name = ft.Name
			def.Function.Description = ft.Description
			def.Function.Parameters = ft.InputSchema
			defs = append(defs, def)
		}
	}

	prompt := "You have access to the following tools. To use a tool, respond with a JSON object in this format:\n"
	prompt += `{"name": "<tool_name>", "arguments": {<tool_arguments>}}` + "\n\n"
	prompt += "Available tools:\n"

	for _, def := range defs {
		toolJSON, _ := json.MarshalIndent(def, "", "  ")
		prompt += string(toolJSON) + "\n\n"
	}

	return prompt
}

// toolCall represents a parsed tool call from the model response
type toolCall struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

// parseToolCalls extracts tool calls from the model's response text
// It looks for JSON objects with "name" and "arguments" fields
func parseToolCalls(response string) []toolCall {
	var calls []toolCall

	// Try to find JSON objects in the response
	// Pattern matches {"name": "...", "arguments": {...}}
	jsonPattern := regexp.MustCompile(`\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}`)
	matches := jsonPattern.FindAllString(response, -1)

	for _, match := range matches {
		var tc toolCall
		if err := json.Unmarshal([]byte(match), &tc); err == nil && tc.Name != "" {
			calls = append(calls, tc)
		}
	}

	// If regex didn't work, try line-by-line parsing
	if len(calls) == 0 {
		lines := strings.Split(response, "\n")
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if strings.HasPrefix(line, "{") && strings.Contains(line, "\"name\"") {
				var tc toolCall
				if err := json.Unmarshal([]byte(line), &tc); err == nil && tc.Name != "" {
					calls = append(calls, tc)
				}
			}
		}
	}

	return calls
}
