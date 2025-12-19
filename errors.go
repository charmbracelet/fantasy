package fantasy

import (
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/charmbracelet/x/exp/slice"
)

// Error is a custom error type for the fantasy package.
type Error struct {
	Message string
	Title   string
	Cause   error
}

// NewAIError creates a new Error for AI/provider failures.
func NewAIError(title, message string, cause error) *Error {
	return &Error{
		Title:   title,
		Message: message,
		Cause:   cause,
	}
}

// InvalidArgumentError represents an error caused by invalid input.
type InvalidArgumentError struct {
	Argument string
	Message  string
	Cause    error
}

// Error implements the error interface.
func (err *InvalidArgumentError) Error() string {
	if err.Argument != "" {
		return fmt.Sprintf("invalid argument %s: %s", err.Argument, err.Message)
	}
	return fmt.Sprintf("invalid argument: %s", err.Message)
}

// Unwrap returns the underlying cause.
func (err InvalidArgumentError) Unwrap() error {
	return err.Cause
}

// NewInvalidArgumentError constructs an InvalidArgumentError.
func NewInvalidArgumentError(argument, message string, cause error) error {
	return &InvalidArgumentError{
		Argument: argument,
		Message:  message,
		Cause:    cause,
	}
}

// APICallError represents an error from an API call.
type APICallError struct {
	*Error
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
		isRetryable = statusCode == http.StatusRequestTimeout || statusCode == http.StatusConflict || statusCode == http.StatusTooManyRequests || statusCode >= 500
	}

	return &APICallError{
		Error:           NewAIError("AI_APICallError", message, cause),
		URL:             url,
		RequestDump:     requestDump,
		StatusCode:      statusCode,
		ResponseHeaders: responseHeaders,
		ResponseDump:    responseDump,
		IsRetryable:     isRetryable,
	}
}

func (err *Error) Error() string {
	if err.Title == "" {
		return err.Message
	}
	return fmt.Sprintf("%s: %s", err.Title, err.Message)
}

func (err Error) Unwrap() error {
	return err.Cause
}

// ProviderError represents an error returned by an external provider.
type ProviderError struct {
	Message string
	Title   string
	Cause   error

	URL             string
	StatusCode      int
	RequestBody     []byte
	ResponseHeaders map[string]string
	ResponseBody    []byte
}

func (m *ProviderError) Error() string {
	if m.Title == "" {
		return m.Message
	}
	return fmt.Sprintf("%s: %s", m.Title, m.Message)
}

// IsRetryable checks if the error is retryable based on the status code.
func (m *ProviderError) IsRetryable() bool {
	return m.StatusCode == http.StatusRequestTimeout || m.StatusCode == http.StatusConflict || m.StatusCode == http.StatusTooManyRequests
}

// RetryError represents an error that occurred during retry operations.
type RetryError struct {
	Errors []error
}

func (e *RetryError) Error() string {
	if err, ok := slice.Last(e.Errors); ok {
		return fmt.Sprintf("retry error: %v", err)
	}
	return "retry error: no underlying errors"
}

func (e RetryError) Unwrap() error {
	if err, ok := slice.Last(e.Errors); ok {
		return err
	}
	return nil
}

// ErrorTitleForStatusCode returns a human-readable title for a given HTTP status code.
func ErrorTitleForStatusCode(statusCode int) string {
	return strings.ToLower(http.StatusText(statusCode))
}

// NoObjectGeneratedError is returned when object generation fails
// due to parsing errors, validation errors, or model failures.
type NoObjectGeneratedError struct {
	RawText         string
	ParseError      error
	ValidationError error
	Usage           Usage
	FinishReason    FinishReason
}

// Error implements the error interface.
func (e *NoObjectGeneratedError) Error() string {
	if e.ValidationError != nil {
		return fmt.Sprintf("object validation failed: %v", e.ValidationError)
	}
	if e.ParseError != nil {
		return fmt.Sprintf("failed to parse object: %v", e.ParseError)
	}
	return "failed to generate object"
}

// IsNoObjectGeneratedError checks if an error is of type NoObjectGeneratedError.
func IsNoObjectGeneratedError(err error) bool {
	var target *NoObjectGeneratedError
	return errors.As(err, &target)
}
