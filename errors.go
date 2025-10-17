package fantasy

import (
	"encoding/json"
	"errors"
	"fmt"
)

// markerSymbol is used for identifying AI SDK Error instances.
var markerSymbol = "fantasy.error"

// AIError is a custom error type for AI SDK related errors.
type AIError struct {
	Name    string
	Message string
	Cause   error
	marker  string
}

// Error implements the error interface.
func (e *AIError) Error() string {
	return e.Message
}

// Unwrap returns the underlying cause of the error.
func (e *AIError) Unwrap() error {
	return e.Cause
}

// NewAIError creates a new AI SDK Error.
func NewAIError(name, message string, cause error) *AIError {
	return &AIError{
		Name:    name,
		Message: message,
		Cause:   cause,
		marker:  markerSymbol,
	}
}

// IsAIError checks if the given error is an AI SDK Error.
func IsAIError(err error) bool {
	var sdkErr *AIError
	return errors.As(err, &sdkErr) && sdkErr.marker == markerSymbol
}

// APICallError represents an error from an API call.
type APICallError struct {
	*AIError
	URL             string
	RequestDump     string
	StatusCode      int
	ResponseHeaders map[string]string
	ResponseDump    string
	IsRetryable     bool
}

// NewAPICallError creates a new API call error.
func NewAPICallError(message, url string, requestDump string, statusCode int, responseHeaders map[string]string, responseDump string, cause error, isRetryable bool) *APICallError {
	if !isRetryable && statusCode != 0 {
		isRetryable = statusCode == 408 || statusCode == 409 || statusCode == 429 || statusCode >= 500
	}

	return &APICallError{
		AIError:         NewAIError("AI_APICallError", message, cause),
		URL:             url,
		RequestDump:     requestDump,
		StatusCode:      statusCode,
		ResponseHeaders: responseHeaders,
		ResponseDump:    responseDump,
		IsRetryable:     isRetryable,
	}
}

// EmptyResponseBodyError represents an empty response body error.
type EmptyResponseBodyError struct {
	*AIError
}

// NewEmptyResponseBodyError creates a new empty response body error.
func NewEmptyResponseBodyError(message string) *EmptyResponseBodyError {
	if message == "" {
		message = "Empty response body"
	}
	return &EmptyResponseBodyError{
		AIError: NewAIError("AI_EmptyResponseBodyError", message, nil),
	}
}

// InvalidArgumentError represents an invalid function argument error.
type InvalidArgumentError struct {
	*AIError
	Argument string
}

// NewInvalidArgumentError creates a new invalid argument error.
func NewInvalidArgumentError(argument, message string, cause error) *InvalidArgumentError {
	return &InvalidArgumentError{
		AIError:  NewAIError("AI_InvalidArgumentError", message, cause),
		Argument: argument,
	}
}

// InvalidPromptError represents an invalid prompt error.
type InvalidPromptError struct {
	*AIError
	Prompt any
}

// NewInvalidPromptError creates a new invalid prompt error.
func NewInvalidPromptError(prompt any, message string, cause error) *InvalidPromptError {
	return &InvalidPromptError{
		AIError: NewAIError("AI_InvalidPromptError", fmt.Sprintf("Invalid prompt: %s", message), cause),
		Prompt:  prompt,
	}
}

// InvalidResponseDataError represents invalid response data from the server.
type InvalidResponseDataError struct {
	*AIError
	Data any
}

// NewInvalidResponseDataError creates a new invalid response data error.
func NewInvalidResponseDataError(data any, message string) *InvalidResponseDataError {
	if message == "" {
		dataJSON, _ := json.Marshal(data)
		message = fmt.Sprintf("Invalid response data: %s.", string(dataJSON))
	}
	return &InvalidResponseDataError{
		AIError: NewAIError("AI_InvalidResponseDataError", message, nil),
		Data:    data,
	}
}

// JSONParseError represents a JSON parsing error.
type JSONParseError struct {
	*AIError
	Text string
}

// NewJSONParseError creates a new JSON parse error.
func NewJSONParseError(text string, cause error) *JSONParseError {
	message := fmt.Sprintf("JSON parsing failed: Text: %s.\nError message: %s", text, GetErrorMessage(cause))
	return &JSONParseError{
		AIError: NewAIError("AI_JSONParseError", message, cause),
		Text:    text,
	}
}

// LoadAPIKeyError represents an error loading an API key.
type LoadAPIKeyError struct {
	*AIError
}

// NewLoadAPIKeyError creates a new load API key error.
func NewLoadAPIKeyError(message string) *LoadAPIKeyError {
	return &LoadAPIKeyError{
		AIError: NewAIError("AI_LoadAPIKeyError", message, nil),
	}
}

// LoadSettingError represents an error loading a setting.
type LoadSettingError struct {
	*AIError
}

// NewLoadSettingError creates a new load setting error.
func NewLoadSettingError(message string) *LoadSettingError {
	return &LoadSettingError{
		AIError: NewAIError("AI_LoadSettingError", message, nil),
	}
}

// NoContentGeneratedError is thrown when the AI provider fails to generate any content.
type NoContentGeneratedError struct {
	*AIError
}

// NewNoContentGeneratedError creates a new no content generated error.
func NewNoContentGeneratedError(message string) *NoContentGeneratedError {
	if message == "" {
		message = "No content generated."
	}
	return &NoContentGeneratedError{
		AIError: NewAIError("AI_NoContentGeneratedError", message, nil),
	}
}

// ModelType represents the type of model.
type ModelType string

const (
	ModelTypeLanguage      ModelType = "languageModel"
	ModelTypeTextEmbedding ModelType = "textEmbeddingModel"
	ModelTypeImage         ModelType = "imageModel"
	ModelTypeTranscription ModelType = "transcriptionModel"
	ModelTypeSpeech        ModelType = "speechModel"
)

// NoSuchModelError represents an error when a model is not found.
type NoSuchModelError struct {
	*AIError
	ModelID   string
	ModelType ModelType
}

// NewNoSuchModelError creates a new no such model error.
func NewNoSuchModelError(modelID string, modelType ModelType, message string) *NoSuchModelError {
	if message == "" {
		message = fmt.Sprintf("No such %s: %s", modelType, modelID)
	}
	return &NoSuchModelError{
		AIError:   NewAIError("AI_NoSuchModelError", message, nil),
		ModelID:   modelID,
		ModelType: modelType,
	}
}

// TooManyEmbeddingValuesForCallError represents an error when too many values are provided for embedding.
type TooManyEmbeddingValuesForCallError struct {
	*AIError
	Provider             string
	ModelID              string
	MaxEmbeddingsPerCall int
	Values               []any
}

// NewTooManyEmbeddingValuesForCallError creates a new too many embedding values error.
func NewTooManyEmbeddingValuesForCallError(provider, modelID string, maxEmbeddingsPerCall int, values []any) *TooManyEmbeddingValuesForCallError {
	message := fmt.Sprintf(
		"Too many values for a single embedding call. The %s model \"%s\" can only embed up to %d values per call, but %d values were provided.",
		provider, modelID, maxEmbeddingsPerCall, len(values),
	)
	return &TooManyEmbeddingValuesForCallError{
		AIError:              NewAIError("AI_TooManyEmbeddingValuesForCallError", message, nil),
		Provider:             provider,
		ModelID:              modelID,
		MaxEmbeddingsPerCall: maxEmbeddingsPerCall,
		Values:               values,
	}
}

// TypeValidationError represents a type validation error.
type TypeValidationError struct {
	*AIError
	Value any
}

// NewTypeValidationError creates a new type validation error.
func NewTypeValidationError(value any, cause error) *TypeValidationError {
	valueJSON, _ := json.Marshal(value)
	message := fmt.Sprintf(
		"Type validation failed: Value: %s.\nError message: %s",
		string(valueJSON), GetErrorMessage(cause),
	)
	return &TypeValidationError{
		AIError: NewAIError("AI_TypeValidationError", message, cause),
		Value:   value,
	}
}

// WrapTypeValidationError wraps an error into a TypeValidationError.
func WrapTypeValidationError(value any, cause error) *TypeValidationError {
	if tvErr, ok := cause.(*TypeValidationError); ok && tvErr.Value == value {
		return tvErr
	}
	return NewTypeValidationError(value, cause)
}

// UnsupportedFunctionalityError represents an unsupported functionality error.
type UnsupportedFunctionalityError struct {
	*AIError
	Functionality string
}

// NewUnsupportedFunctionalityError creates a new unsupported functionality error.
func NewUnsupportedFunctionalityError(functionality, message string) *UnsupportedFunctionalityError {
	if message == "" {
		message = fmt.Sprintf("'%s' functionality not supported.", functionality)
	}
	return &UnsupportedFunctionalityError{
		AIError:       NewAIError("AI_UnsupportedFunctionalityError", message, nil),
		Functionality: functionality,
	}
}

// GetErrorMessage extracts a message from an error.
func GetErrorMessage(err error) string {
	if err == nil {
		return "unknown error"
	}
	return err.Error()
}
