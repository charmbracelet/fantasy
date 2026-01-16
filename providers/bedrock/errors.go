package bedrock

import (
	"errors"

	"charm.land/fantasy"
	"github.com/aws/smithy-go"
)

// convertAWSError converts AWS SDK errors to fantasy.ProviderError.
// This provides a basic implementation for task 8; full implementation in task 11.
func convertAWSError(err error) error {
	if err == nil {
		return nil
	}

	// Check for specific AWS error types
	var apiErr smithy.APIError
	if errors.As(err, &apiErr) {
		return &fantasy.ProviderError{
			Title:      apiErr.ErrorCode(),
			Message:    apiErr.ErrorMessage(),
			StatusCode: getStatusCodeFromAWSError(apiErr),
		}
	}

	// Generic error
	return &fantasy.ProviderError{
		Title:   "AWS Error",
		Message: err.Error(),
	}
}

// getStatusCodeFromAWSError maps AWS error codes to HTTP status codes.
// This is a basic implementation; full implementation in task 11.
func getStatusCodeFromAWSError(apiErr smithy.APIError) int {
	errorCode := apiErr.ErrorCode()

	// Map common AWS error codes to HTTP status codes
	switch errorCode {
	case "UnauthorizedException", "InvalidSignatureException", "ExpiredTokenException":
		return 401
	case "ThrottlingException", "TooManyRequestsException":
		return 429
	case "ValidationException", "InvalidRequestException":
		return 400
	case "ServiceUnavailableException", "InternalServerException":
		return 500
	default:
		return 500
	}
}
