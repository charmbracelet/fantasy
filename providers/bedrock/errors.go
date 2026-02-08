package bedrock

import (
	"errors"
	"net/http"

	"charm.land/fantasy"
	"github.com/aws/smithy-go"
)

// convertAWSError converts AWS SDK errors to fantasy.ProviderError.
// It maps AWS error codes to appropriate HTTP status codes and extracts
// error messages from AWS errors.
func convertAWSError(err error) error {
	if err == nil {
		return nil
	}

	// Check for smithy.APIError (the base error type for AWS SDK v2)
	var apiErr smithy.APIError
	if errors.As(err, &apiErr) {
		statusCode := getStatusCodeFromAWSError(apiErr)
		return &fantasy.ProviderError{
			Title:      fantasy.ErrorTitleForStatusCode(statusCode),
			Message:    apiErr.ErrorMessage(),
			Cause:      err,
			StatusCode: statusCode,
		}
	}

	// Generic error - wrap it in a ProviderError
	return &fantasy.ProviderError{
		Title:   "AWS Error",
		Message: err.Error(),
		Cause:   err,
	}
}

// getStatusCodeFromAWSError maps AWS error codes to HTTP status codes.
func getStatusCodeFromAWSError(apiErr smithy.APIError) int {
	errorCode := apiErr.ErrorCode()

	// Map common AWS error codes to HTTP status codes
	switch errorCode {
	// Authentication errors (401)
	case "UnrecognizedClientException",
		"InvalidSignatureException",
		"ExpiredTokenException",
		"InvalidAccessKeyId",
		"InvalidToken",
		"AccessDeniedException":
		return http.StatusUnauthorized

	// Throttling errors (429)
	case "ThrottlingException",
		"TooManyRequestsException",
		"ProvisionedThroughputExceededException",
		"RequestLimitExceeded",
		"Throttling":
		return http.StatusTooManyRequests

	// Validation errors (400)
	case "ValidationException",
		"InvalidParameterException",
		"InvalidRequestException",
		"MissingParameter",
		"InvalidInput",
		"BadRequestException":
		return http.StatusBadRequest

	// Service errors (500)
	case "InternalServerError",
		"ServiceUnavailableException",
		"InternalFailure",
		"ServiceException":
		return http.StatusInternalServerError

	// Resource not found (404)
	case "ResourceNotFoundException",
		"ModelNotFoundException",
		"NotFoundException":
		return http.StatusNotFound

	// Default to 500 for unknown errors
	default:
		return http.StatusInternalServerError
	}
}
