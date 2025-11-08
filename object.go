package fantasy

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"reflect"
)

// ObjectMode specifies how structured output should be generated.
type ObjectMode string

const (
	// ObjectModeAuto lets the provider choose the best approach.
	ObjectModeAuto ObjectMode = "auto"

	// ObjectModeJSON forces the use of native JSON mode (if supported).
	ObjectModeJSON ObjectMode = "json"

	// ObjectModeTool forces the use of tool-based approach.
	ObjectModeTool ObjectMode = "tool"

	// ObjectModeText uses text generation with schema in prompt (fallback for models without tool/JSON support).
	ObjectModeText ObjectMode = "text"
)

// ObjectRepairFunc is a function that attempts to repair invalid JSON output.
// It receives the raw text and the error encountered during parsing or validation,
// and returns repaired text or an error if repair is not possible.
type ObjectRepairFunc func(ctx context.Context, text string, err error) (string, error)

// ObjectCall represents a request to generate a structured object.
type ObjectCall struct {
	Prompt            Prompt
	Schema            Schema
	SchemaName        string
	SchemaDescription string

	MaxOutputTokens  *int64
	Temperature      *float64
	TopP             *float64
	TopK             *int64
	PresencePenalty  *float64
	FrequencyPenalty *float64

	ProviderOptions ProviderOptions

	RepairText ObjectRepairFunc
}

// ObjectResponse represents the response from a structured object generation.
type ObjectResponse struct {
	Object           any
	RawText          string
	Usage            Usage
	FinishReason     FinishReason
	Warnings         []CallWarning
	ProviderMetadata ProviderMetadata
}

// ObjectStreamPartType indicates the type of stream part.
type ObjectStreamPartType string

const (
	// ObjectStreamPartTypeObject is emitted when a new partial object is available.
	ObjectStreamPartTypeObject ObjectStreamPartType = "object"

	// ObjectStreamPartTypeTextDelta is emitted for text deltas (if model generates text).
	ObjectStreamPartTypeTextDelta ObjectStreamPartType = "text-delta"

	// ObjectStreamPartTypeError is emitted when an error occurs.
	ObjectStreamPartTypeError ObjectStreamPartType = "error"

	// ObjectStreamPartTypeFinish is emitted when streaming completes.
	ObjectStreamPartTypeFinish ObjectStreamPartType = "finish"
)

// ObjectStreamPart represents a single chunk in the object stream.
type ObjectStreamPart struct {
	Type             ObjectStreamPartType
	Object           any
	Delta            string
	Error            error
	Usage            Usage
	FinishReason     FinishReason
	Warnings         []CallWarning
	ProviderMetadata ProviderMetadata
}

// ObjectStreamResponse is an iterator over ObjectStreamPart.
type ObjectStreamResponse = iter.Seq[ObjectStreamPart]

// ObjectResult is a typed result wrapper returned by GenerateObject[T].
type ObjectResult[T any] struct {
	Object           T
	RawText          string
	Usage            Usage
	FinishReason     FinishReason
	Warnings         []CallWarning
	ProviderMetadata ProviderMetadata
}

// StreamObjectResult provides typed access to a streaming object generation result.
type StreamObjectResult[T any] struct {
	stream ObjectStreamResponse
	ctx    context.Context
}

// NewStreamObjectResult creates a typed stream result from an untyped stream.
func NewStreamObjectResult[T any](ctx context.Context, stream ObjectStreamResponse) *StreamObjectResult[T] {
	return &StreamObjectResult[T]{
		stream: stream,
		ctx:    ctx,
	}
}

// PartialObjectStream returns an iterator that yields progressively more complete objects.
// Only emits when the object actually changes (deduplication).
func (s *StreamObjectResult[T]) PartialObjectStream() iter.Seq[T] {
	return func(yield func(T) bool) {
		var lastObject T
		var hasEmitted bool

		for part := range s.stream {
			if part.Type == ObjectStreamPartTypeObject && part.Object != nil {
				var current T
				if err := unmarshalObject(part.Object, &current); err != nil {
					continue
				}

				if !hasEmitted || !reflect.DeepEqual(current, lastObject) {
					if !yield(current) {
						return
					}
					lastObject = current
					hasEmitted = true
				}
			}
		}
	}
}

// TextStream returns an iterator that yields text deltas.
// Useful if the model generates explanatory text alongside the object.
func (s *StreamObjectResult[T]) TextStream() iter.Seq[string] {
	return func(yield func(string) bool) {
		for part := range s.stream {
			if part.Type == ObjectStreamPartTypeTextDelta && part.Delta != "" {
				if !yield(part.Delta) {
					return
				}
			}
		}
	}
}

// FullStream returns an iterator that yields all stream parts including errors and metadata.
func (s *StreamObjectResult[T]) FullStream() iter.Seq[ObjectStreamPart] {
	return s.stream
}

// Object waits for the stream to complete and returns the final object.
// Returns an error if streaming fails or no valid object was generated.
func (s *StreamObjectResult[T]) Object() (*ObjectResult[T], error) {
	var finalObject T
	var usage Usage
	var finishReason FinishReason
	var warnings []CallWarning
	var providerMetadata ProviderMetadata
	var rawText string
	var lastError error
	hasObject := false

	for part := range s.stream {
		switch part.Type {
		case ObjectStreamPartTypeObject:
			if part.Object != nil {
				if err := unmarshalObject(part.Object, &finalObject); err == nil {
					hasObject = true
					if jsonBytes, err := json.Marshal(part.Object); err == nil {
						rawText = string(jsonBytes)
					}
				}
			}

		case ObjectStreamPartTypeError:
			lastError = part.Error

		case ObjectStreamPartTypeFinish:
			usage = part.Usage
			finishReason = part.FinishReason
			if len(part.Warnings) > 0 {
				warnings = part.Warnings
			}
			if len(part.ProviderMetadata) > 0 {
				providerMetadata = part.ProviderMetadata
			}
		}
	}

	if lastError != nil {
		return nil, lastError
	}

	if !hasObject {
		return nil, &NoObjectGeneratedError{
			RawText:      rawText,
			ParseError:   fmt.Errorf("no valid object generated in stream"),
			Usage:        usage,
			FinishReason: finishReason,
		}
	}

	return &ObjectResult[T]{
		Object:           finalObject,
		RawText:          rawText,
		Usage:            usage,
		FinishReason:     finishReason,
		Warnings:         warnings,
		ProviderMetadata: providerMetadata,
	}, nil
}

// GenerateObject generates a structured object that matches the given type T.
// The schema is automatically generated from T using reflection.
//
// Example:
//
//	type Recipe struct {
//	    Name        string   `json:"name"`
//	    Ingredients []string `json:"ingredients"`
//	}
//
//	result, err := fantasy.GenerateObject[Recipe](ctx, model, fantasy.ObjectCall{
//	    Prompt: fantasy.Prompt{fantasy.NewUserMessage("Generate a lasagna recipe")},
//	})
func GenerateObject[T any](
	ctx context.Context,
	model LanguageModel,
	opts ObjectCall,
) (*ObjectResult[T], error) {
	var zero T
	schema := generateSchema(reflect.TypeOf(zero))
	opts.Schema = schema

	resp, err := model.GenerateObject(ctx, opts)
	if err != nil {
		return nil, err
	}

	var result T
	if err := unmarshalObject(resp.Object, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal to %T: %w", result, err)
	}

	return &ObjectResult[T]{
		Object:           result,
		RawText:          resp.RawText,
		Usage:            resp.Usage,
		FinishReason:     resp.FinishReason,
		Warnings:         resp.Warnings,
		ProviderMetadata: resp.ProviderMetadata,
	}, nil
}

func unmarshalObject(obj any, target any) error {
	jsonBytes, err := json.Marshal(obj)
	if err != nil {
		return fmt.Errorf("failed to marshal object: %w", err)
	}

	if err := json.Unmarshal(jsonBytes, target); err != nil {
		return fmt.Errorf("failed to unmarshal into target type: %w", err)
	}

	return nil
}

// StreamObject streams a structured object that matches the given type T.
// Returns a StreamObjectResult[T] with progressive updates and deduplication.
//
// Example:
//
//	stream, err := fantasy.StreamObject[Recipe](ctx, model, fantasy.ObjectCall{
//	    Prompt: fantasy.Prompt{fantasy.NewUserMessage("Generate a lasagna recipe")},
//	})
//
//	for partial := range stream.PartialObjectStream() {
//	    fmt.Printf("Progress: %s\n", partial.Name)
//	}
//
//	result, err := stream.Object()  // Wait for final result
func StreamObject[T any](
	ctx context.Context,
	model LanguageModel,
	opts ObjectCall,
) (*StreamObjectResult[T], error) {
	var zero T
	schema := generateSchema(reflect.TypeOf(zero))
	opts.Schema = schema

	stream, err := model.StreamObject(ctx, opts)
	if err != nil {
		return nil, err
	}

	return NewStreamObjectResult[T](ctx, stream), nil
}

// GenerateObjectWithTool is a helper for providers without native JSON mode.
// It converts the schema to a tool definition, forces the model to call it,
// and extracts the tool's input as the structured output.
func GenerateObjectWithTool(
	ctx context.Context,
	model LanguageModel,
	call ObjectCall,
) (*ObjectResponse, error) {
	toolName := call.SchemaName
	if toolName == "" {
		toolName = "generate_object"
	}

	toolDescription := call.SchemaDescription
	if toolDescription == "" {
		toolDescription = "Generate a structured object matching the schema"
	}

	tool := FunctionTool{
		Name:        toolName,
		Description: toolDescription,
		InputSchema: SchemaToMap(call.Schema),
	}

	toolChoice := SpecificToolChoice(tool.Name)
	resp, err := model.Generate(ctx, Call{
		Prompt:           call.Prompt,
		Tools:            []Tool{tool},
		ToolChoice:       &toolChoice,
		MaxOutputTokens:  call.MaxOutputTokens,
		Temperature:      call.Temperature,
		TopP:             call.TopP,
		TopK:             call.TopK,
		PresencePenalty:  call.PresencePenalty,
		FrequencyPenalty: call.FrequencyPenalty,
		ProviderOptions:  call.ProviderOptions,
	})
	if err != nil {
		return nil, fmt.Errorf("tool-based generation failed: %w", err)
	}

	toolCalls := resp.Content.ToolCalls()
	if len(toolCalls) == 0 {
		return nil, &NoObjectGeneratedError{
			RawText:      resp.Content.Text(),
			ParseError:   fmt.Errorf("no tool call generated"),
			Usage:        resp.Usage,
			FinishReason: resp.FinishReason,
		}
	}

	toolCall := toolCalls[0]

	var obj any
	if call.RepairText != nil {
		obj, err = ParseAndValidateWithRepair(ctx, toolCall.Input, call.Schema, call.RepairText)
	} else {
		obj, err = ParseAndValidate(toolCall.Input, call.Schema)
	}

	if err != nil {
		if nogErr, ok := err.(*NoObjectGeneratedError); ok {
			nogErr.Usage = resp.Usage
			nogErr.FinishReason = resp.FinishReason
		}
		return nil, err
	}

	return &ObjectResponse{
		Object:           obj,
		RawText:          toolCall.Input,
		Usage:            resp.Usage,
		FinishReason:     resp.FinishReason,
		Warnings:         resp.Warnings,
		ProviderMetadata: resp.ProviderMetadata,
	}, nil
}

// GenerateObjectWithText is a helper for providers without tool or JSON mode support.
// It adds the schema to the system prompt and parses the text response as JSON.
// This is a fallback for older models or simple providers.
func GenerateObjectWithText(
	ctx context.Context,
	model LanguageModel,
	call ObjectCall,
) (*ObjectResponse, error) {
	jsonSchemaBytes, err := json.Marshal(call.Schema)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal schema: %w", err)
	}

	schemaInstruction := fmt.Sprintf(
		"You must respond with valid JSON that matches this schema: %s\n"+
			"Respond ONLY with the JSON object, no additional text or explanation.",
		string(jsonSchemaBytes),
	)

	enhancedPrompt := make(Prompt, 0, len(call.Prompt)+1)

	hasSystem := false
	for _, msg := range call.Prompt {
		if msg.Role == MessageRoleSystem {
			hasSystem = true
			existingText := ""
			if len(msg.Content) > 0 {
				if textPart, ok := msg.Content[0].(TextPart); ok {
					existingText = textPart.Text
				}
			}
			enhancedPrompt = append(enhancedPrompt, NewSystemMessage(existingText+"\n\n"+schemaInstruction))
		} else {
			enhancedPrompt = append(enhancedPrompt, msg)
		}
	}

	if !hasSystem {
		enhancedPrompt = append(Prompt{NewSystemMessage(schemaInstruction)}, call.Prompt...)
	}

	resp, err := model.Generate(ctx, Call{
		Prompt:           enhancedPrompt,
		MaxOutputTokens:  call.MaxOutputTokens,
		Temperature:      call.Temperature,
		TopP:             call.TopP,
		TopK:             call.TopK,
		PresencePenalty:  call.PresencePenalty,
		FrequencyPenalty: call.FrequencyPenalty,
		ProviderOptions:  call.ProviderOptions,
	})
	if err != nil {
		return nil, fmt.Errorf("text-based generation failed: %w", err)
	}

	textContent := resp.Content.Text()
	if textContent == "" {
		return nil, &NoObjectGeneratedError{
			RawText:      "",
			ParseError:   fmt.Errorf("no text content in response"),
			Usage:        resp.Usage,
			FinishReason: resp.FinishReason,
		}
	}

	var obj any
	if call.RepairText != nil {
		obj, err = ParseAndValidateWithRepair(ctx, textContent, call.Schema, call.RepairText)
	} else {
		obj, err = ParseAndValidate(textContent, call.Schema)
	}

	if err != nil {
		if nogErr, ok := err.(*NoObjectGeneratedError); ok {
			nogErr.Usage = resp.Usage
			nogErr.FinishReason = resp.FinishReason
		}
		return nil, err
	}

	return &ObjectResponse{
		Object:           obj,
		RawText:          textContent,
		Usage:            resp.Usage,
		FinishReason:     resp.FinishReason,
		Warnings:         resp.Warnings,
		ProviderMetadata: resp.ProviderMetadata,
	}, nil
}

// StreamObjectWithTool is a helper for providers without native JSON streaming.
// It uses streaming tool calls to extract and parse the structured output progressively.
func StreamObjectWithTool(
	ctx context.Context,
	model LanguageModel,
	call ObjectCall,
) (ObjectStreamResponse, error) {
	// Create a tool from the schema
	toolName := call.SchemaName
	if toolName == "" {
		toolName = "generate_object"
	}

	toolDescription := call.SchemaDescription
	if toolDescription == "" {
		toolDescription = "Generate a structured object matching the schema"
	}

	tool := FunctionTool{
		Name:        toolName,
		Description: toolDescription,
		InputSchema: SchemaToMap(call.Schema),
	}

	// Make a streaming Generate call with forced tool choice
	toolChoice := SpecificToolChoice(tool.Name)
	stream, err := model.Stream(ctx, Call{
		Prompt:           call.Prompt,
		Tools:            []Tool{tool},
		ToolChoice:       &toolChoice,
		MaxOutputTokens:  call.MaxOutputTokens,
		Temperature:      call.Temperature,
		TopP:             call.TopP,
		TopK:             call.TopK,
		PresencePenalty:  call.PresencePenalty,
		FrequencyPenalty: call.FrequencyPenalty,
		ProviderOptions:  call.ProviderOptions,
	})
	if err != nil {
		return nil, fmt.Errorf("tool-based streaming failed: %w", err)
	}

	// Convert the text stream to object stream parts
	return func(yield func(ObjectStreamPart) bool) {
		var accumulated string
		var lastParsedObject any
		var usage Usage
		var finishReason FinishReason
		var warnings []CallWarning
		var providerMetadata ProviderMetadata
		var streamErr error

		for part := range stream {
			switch part.Type {
			case StreamPartTypeTextDelta:
				accumulated += part.Delta

				obj, state, parseErr := ParsePartialJSON(accumulated)

				if state == ParseStateSuccessful || state == ParseStateRepaired {
					if err := validateAgainstSchema(obj, call.Schema); err == nil {
						if !reflect.DeepEqual(obj, lastParsedObject) {
							if !yield(ObjectStreamPart{
								Type:   ObjectStreamPartTypeObject,
								Object: obj,
							}) {
								return
							}
							lastParsedObject = obj
						}
					}
				}

				if state == ParseStateFailed && call.RepairText != nil {
					repairedText, repairErr := call.RepairText(ctx, accumulated, parseErr)
					if repairErr == nil {
						obj2, state2, _ := ParsePartialJSON(repairedText)
						if (state2 == ParseStateSuccessful || state2 == ParseStateRepaired) &&
							validateAgainstSchema(obj2, call.Schema) == nil {
							if !reflect.DeepEqual(obj2, lastParsedObject) {
								if !yield(ObjectStreamPart{
									Type:   ObjectStreamPartTypeObject,
									Object: obj2,
								}) {
									return
								}
								lastParsedObject = obj2
							}
						}
					}
				}

			case StreamPartTypeToolInputDelta:
				accumulated += part.Delta

				obj, state, parseErr := ParsePartialJSON(accumulated)
				if state == ParseStateSuccessful || state == ParseStateRepaired {
					if err := validateAgainstSchema(obj, call.Schema); err == nil {
						if !reflect.DeepEqual(obj, lastParsedObject) {
							if !yield(ObjectStreamPart{
								Type:   ObjectStreamPartTypeObject,
								Object: obj,
							}) {
								return
							}
							lastParsedObject = obj
						}
					}
				}

				if state == ParseStateFailed && call.RepairText != nil {
					repairedText, repairErr := call.RepairText(ctx, accumulated, parseErr)
					if repairErr == nil {
						obj2, state2, _ := ParsePartialJSON(repairedText)
						if (state2 == ParseStateSuccessful || state2 == ParseStateRepaired) &&
							validateAgainstSchema(obj2, call.Schema) == nil {
							if !reflect.DeepEqual(obj2, lastParsedObject) {
								if !yield(ObjectStreamPart{
									Type:   ObjectStreamPartTypeObject,
									Object: obj2,
								}) {
									return
								}
								lastParsedObject = obj2
							}
						}
					}
				}

			case StreamPartTypeToolCall:
				toolInput := part.ToolCallInput

				var obj any
				var err error
				if call.RepairText != nil {
					obj, err = ParseAndValidateWithRepair(ctx, toolInput, call.Schema, call.RepairText)
				} else {
					obj, err = ParseAndValidate(toolInput, call.Schema)
				}

				if err == nil {
					if !reflect.DeepEqual(obj, lastParsedObject) {
						if !yield(ObjectStreamPart{
							Type:   ObjectStreamPartTypeObject,
							Object: obj,
						}) {
							return
						}
						lastParsedObject = obj
					}
				}

			case StreamPartTypeError:
				streamErr = part.Error
				if !yield(ObjectStreamPart{
					Type:  ObjectStreamPartTypeError,
					Error: part.Error,
				}) {
					return
				}

			case StreamPartTypeFinish:
				usage = part.Usage
				finishReason = part.FinishReason

			case StreamPartTypeWarnings:
				warnings = part.Warnings
			}

			if len(part.ProviderMetadata) > 0 {
				providerMetadata = part.ProviderMetadata
			}
		}

		if streamErr == nil && lastParsedObject != nil {
			yield(ObjectStreamPart{
				Type:             ObjectStreamPartTypeFinish,
				Usage:            usage,
				FinishReason:     finishReason,
				Warnings:         warnings,
				ProviderMetadata: providerMetadata,
			})
		} else if streamErr == nil && lastParsedObject == nil {
			yield(ObjectStreamPart{
				Type: ObjectStreamPartTypeError,
				Error: &NoObjectGeneratedError{
					RawText:      accumulated,
					ParseError:   fmt.Errorf("no valid object generated in stream"),
					Usage:        usage,
					FinishReason: finishReason,
				},
			})
		}
	}, nil
}

// StreamObjectWithText is a helper for providers without tool or JSON streaming support.
// It adds the schema to the system prompt and parses the streamed text as JSON progressively.
func StreamObjectWithText(
	ctx context.Context,
	model LanguageModel,
	call ObjectCall,
) (ObjectStreamResponse, error) {
	jsonSchemaMap := SchemaToMap(call.Schema)
	jsonSchemaBytes, err := json.Marshal(jsonSchemaMap)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal schema: %w", err)
	}

	schemaInstruction := fmt.Sprintf(
		"You must respond with valid JSON that matches this schema: %s\n"+
			"Respond ONLY with the JSON object, no additional text or explanation.",
		string(jsonSchemaBytes),
	)

	enhancedPrompt := make(Prompt, 0, len(call.Prompt)+1)

	hasSystem := false
	for _, msg := range call.Prompt {
		if msg.Role == MessageRoleSystem {
			hasSystem = true
			existingText := ""
			if len(msg.Content) > 0 {
				if textPart, ok := msg.Content[0].(TextPart); ok {
					existingText = textPart.Text
				}
			}
			enhancedPrompt = append(enhancedPrompt, NewSystemMessage(existingText+"\n\n"+schemaInstruction))
		} else {
			enhancedPrompt = append(enhancedPrompt, msg)
		}
	}

	if !hasSystem {
		enhancedPrompt = append(Prompt{NewSystemMessage(schemaInstruction)}, call.Prompt...)
	}

	stream, err := model.Stream(ctx, Call{
		Prompt:           enhancedPrompt,
		MaxOutputTokens:  call.MaxOutputTokens,
		Temperature:      call.Temperature,
		TopP:             call.TopP,
		TopK:             call.TopK,
		PresencePenalty:  call.PresencePenalty,
		FrequencyPenalty: call.FrequencyPenalty,
		ProviderOptions:  call.ProviderOptions,
	})
	if err != nil {
		return nil, fmt.Errorf("text-based streaming failed: %w", err)
	}

	return func(yield func(ObjectStreamPart) bool) {
		var accumulated string
		var lastParsedObject any
		var usage Usage
		var finishReason FinishReason
		var warnings []CallWarning
		var providerMetadata ProviderMetadata
		var streamErr error

		for part := range stream {
			switch part.Type {
			case StreamPartTypeTextDelta:
				accumulated += part.Delta

				obj, state, parseErr := ParsePartialJSON(accumulated)

				if state == ParseStateSuccessful || state == ParseStateRepaired {
					if err := validateAgainstSchema(obj, call.Schema); err == nil {
						if !reflect.DeepEqual(obj, lastParsedObject) {
							if !yield(ObjectStreamPart{
								Type:   ObjectStreamPartTypeObject,
								Object: obj,
							}) {
								return
							}
							lastParsedObject = obj
						}
					}
				}

				if state == ParseStateFailed && call.RepairText != nil {
					repairedText, repairErr := call.RepairText(ctx, accumulated, parseErr)
					if repairErr == nil {
						obj2, state2, _ := ParsePartialJSON(repairedText)
						if (state2 == ParseStateSuccessful || state2 == ParseStateRepaired) &&
							validateAgainstSchema(obj2, call.Schema) == nil {
							if !reflect.DeepEqual(obj2, lastParsedObject) {
								if !yield(ObjectStreamPart{
									Type:   ObjectStreamPartTypeObject,
									Object: obj2,
								}) {
									return
								}
								lastParsedObject = obj2
							}
						}
					}
				}

			case StreamPartTypeError:
				streamErr = part.Error
				if !yield(ObjectStreamPart{
					Type:  ObjectStreamPartTypeError,
					Error: part.Error,
				}) {
					return
				}

			case StreamPartTypeFinish:
				usage = part.Usage
				finishReason = part.FinishReason

			case StreamPartTypeWarnings:
				warnings = part.Warnings
			}

			if len(part.ProviderMetadata) > 0 {
				providerMetadata = part.ProviderMetadata
			}
		}

		if streamErr == nil && lastParsedObject != nil {
			yield(ObjectStreamPart{
				Type:             ObjectStreamPartTypeFinish,
				Usage:            usage,
				FinishReason:     finishReason,
				Warnings:         warnings,
				ProviderMetadata: providerMetadata,
			})
		} else if streamErr == nil && lastParsedObject == nil {
			yield(ObjectStreamPart{
				Type: ObjectStreamPartTypeError,
				Error: &NoObjectGeneratedError{
					RawText:      accumulated,
					ParseError:   fmt.Errorf("no valid object generated in stream"),
					Usage:        usage,
					FinishReason: finishReason,
				},
			})
		}
	}, nil
}
