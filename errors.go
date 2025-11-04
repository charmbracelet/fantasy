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
