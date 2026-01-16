package bedrock

import (
	"errors"
	"net/http"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

// Unit tests for specific error types

func TestConvertAWSError_AuthenticationError(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name      string
		errorCode string
	}{
		{"UnrecognizedClientException", "UnrecognizedClientException"},
		{"InvalidSignatureException", "InvalidSignatureException"},
		{"ExpiredTokenException", "ExpiredTokenException"},
		{"InvalidAccessKeyId", "InvalidAccessKeyId"},
		{"InvalidToken", "InvalidToken"},
		{"AccessDeniedException", "AccessDeniedException"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			awsErr := &mockAPIError{
				code:    tc.errorCode,
				message: "Authentication failed",
			}

			convertedErr := convertAWSError(awsErr)
			require.NotNil(t, convertedErr)

			providerErr, ok := convertedErr.(*fantasy.ProviderError)
			require.True(t, ok, "Expected ProviderError")
			require.Equal(t, http.StatusUnauthorized, providerErr.StatusCode)
			require.Equal(t, "Authentication failed", providerErr.Message)
			require.NotEmpty(t, providerErr.Title)
			require.Equal(t, awsErr, providerErr.Cause)
		})
	}
}

func TestConvertAWSError_ThrottlingError(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name      string
		errorCode string
	}{
		{"ThrottlingException", "ThrottlingException"},
		{"TooManyRequestsException", "TooManyRequestsException"},
		{"ProvisionedThroughputExceededException", "ProvisionedThroughputExceededException"},
		{"RequestLimitExceeded", "RequestLimitExceeded"},
		{"Throttling", "Throttling"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			awsErr := &mockAPIError{
				code:    tc.errorCode,
				message: "Rate limit exceeded",
			}

			convertedErr := convertAWSError(awsErr)
			require.NotNil(t, convertedErr)

			providerErr, ok := convertedErr.(*fantasy.ProviderError)
			require.True(t, ok, "Expected ProviderError")
			require.Equal(t, http.StatusTooManyRequests, providerErr.StatusCode)
			require.Equal(t, "Rate limit exceeded", providerErr.Message)
			require.NotEmpty(t, providerErr.Title)
			require.Equal(t, awsErr, providerErr.Cause)
		})
	}
}

func TestConvertAWSError_ValidationError(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name      string
		errorCode string
	}{
		{"ValidationException", "ValidationException"},
		{"InvalidParameterException", "InvalidParameterException"},
		{"InvalidRequestException", "InvalidRequestException"},
		{"MissingParameter", "MissingParameter"},
		{"InvalidInput", "InvalidInput"},
		{"BadRequestException", "BadRequestException"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			awsErr := &mockAPIError{
				code:    tc.errorCode,
				message: "Invalid request parameters",
			}

			convertedErr := convertAWSError(awsErr)
			require.NotNil(t, convertedErr)

			providerErr, ok := convertedErr.(*fantasy.ProviderError)
			require.True(t, ok, "Expected ProviderError")
			require.Equal(t, http.StatusBadRequest, providerErr.StatusCode)
			require.Equal(t, "Invalid request parameters", providerErr.Message)
			require.NotEmpty(t, providerErr.Title)
			require.Equal(t, awsErr, providerErr.Cause)
		})
	}
}

func TestConvertAWSError_ServiceError(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name      string
		errorCode string
	}{
		{"InternalServerError", "InternalServerError"},
		{"ServiceUnavailableException", "ServiceUnavailableException"},
		{"InternalFailure", "InternalFailure"},
		{"ServiceException", "ServiceException"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			awsErr := &mockAPIError{
				code:    tc.errorCode,
				message: "Internal service error",
			}

			convertedErr := convertAWSError(awsErr)
			require.NotNil(t, convertedErr)

			providerErr, ok := convertedErr.(*fantasy.ProviderError)
			require.True(t, ok, "Expected ProviderError")
			require.Equal(t, http.StatusInternalServerError, providerErr.StatusCode)
			require.Equal(t, "Internal service error", providerErr.Message)
			require.NotEmpty(t, providerErr.Title)
			require.Equal(t, awsErr, providerErr.Cause)
		})
	}
}

func TestConvertAWSError_ResourceNotFoundError(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name      string
		errorCode string
	}{
		{"ResourceNotFoundException", "ResourceNotFoundException"},
		{"ModelNotFoundException", "ModelNotFoundException"},
		{"NotFoundException", "NotFoundException"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			awsErr := &mockAPIError{
				code:    tc.errorCode,
				message: "Resource not found",
			}

			convertedErr := convertAWSError(awsErr)
			require.NotNil(t, convertedErr)

			providerErr, ok := convertedErr.(*fantasy.ProviderError)
			require.True(t, ok, "Expected ProviderError")
			require.Equal(t, http.StatusNotFound, providerErr.StatusCode)
			require.Equal(t, "Resource not found", providerErr.Message)
			require.NotEmpty(t, providerErr.Title)
			require.Equal(t, awsErr, providerErr.Cause)
		})
	}
}

func TestConvertAWSError_GenericError(t *testing.T) {
	t.Parallel()

	// Test with a generic error that doesn't implement smithy.APIError
	genericErr := errors.New("generic error message")

	convertedErr := convertAWSError(genericErr)
	require.NotNil(t, convertedErr)

	providerErr, ok := convertedErr.(*fantasy.ProviderError)
	require.True(t, ok, "Expected ProviderError")
	require.Equal(t, "generic error message", providerErr.Message)
	require.Equal(t, "AWS Error", providerErr.Title)
	require.Equal(t, genericErr, providerErr.Cause)
	require.Equal(t, 0, providerErr.StatusCode, "Generic errors should not have a status code set")
}

func TestConvertAWSError_UnknownErrorCode(t *testing.T) {
	t.Parallel()

	// Test with an unknown AWS error code
	awsErr := &mockAPIError{
		code:    "UnknownErrorCode",
		message: "Unknown error occurred",
	}

	convertedErr := convertAWSError(awsErr)
	require.NotNil(t, convertedErr)

	providerErr, ok := convertedErr.(*fantasy.ProviderError)
	require.True(t, ok, "Expected ProviderError")
	require.Equal(t, http.StatusInternalServerError, providerErr.StatusCode,
		"Unknown error codes should default to 500")
	require.Equal(t, "Unknown error occurred", providerErr.Message)
	require.NotEmpty(t, providerErr.Title)
	require.Equal(t, awsErr, providerErr.Cause)
}

func TestConvertAWSError_NilError(t *testing.T) {
	t.Parallel()

	// Test with nil error
	convertedErr := convertAWSError(nil)
	require.Nil(t, convertedErr, "convertAWSError should return nil for nil input")
}

func TestConvertAWSError_ErrorMessagePreservation(t *testing.T) {
	t.Parallel()

	// Test that error messages are preserved exactly
	testMessages := []string{
		"Simple error message",
		"Error with special characters: !@#$%^&*()",
		"Multi-line\nerror\nmessage",
		"Error with unicode: 你好世界",
		"",
	}

	for _, msg := range testMessages {
		t.Run(msg, func(t *testing.T) {
			awsErr := &mockAPIError{
				code:    "ValidationException",
				message: msg,
			}

			convertedErr := convertAWSError(awsErr)
			require.NotNil(t, convertedErr)

			providerErr, ok := convertedErr.(*fantasy.ProviderError)
			require.True(t, ok, "Expected ProviderError")
			require.Equal(t, msg, providerErr.Message, "Error message should be preserved exactly")
		})
	}
}

func TestGetStatusCodeFromAWSError(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		errorCode          string
		expectedStatusCode int
	}{
		// Authentication errors
		{"UnrecognizedClientException", http.StatusUnauthorized},
		{"InvalidSignatureException", http.StatusUnauthorized},
		{"ExpiredTokenException", http.StatusUnauthorized},
		{"InvalidAccessKeyId", http.StatusUnauthorized},
		{"InvalidToken", http.StatusUnauthorized},
		{"AccessDeniedException", http.StatusUnauthorized},

		// Throttling errors
		{"ThrottlingException", http.StatusTooManyRequests},
		{"TooManyRequestsException", http.StatusTooManyRequests},
		{"ProvisionedThroughputExceededException", http.StatusTooManyRequests},
		{"RequestLimitExceeded", http.StatusTooManyRequests},
		{"Throttling", http.StatusTooManyRequests},

		// Validation errors
		{"ValidationException", http.StatusBadRequest},
		{"InvalidParameterException", http.StatusBadRequest},
		{"InvalidRequestException", http.StatusBadRequest},
		{"MissingParameter", http.StatusBadRequest},
		{"InvalidInput", http.StatusBadRequest},
		{"BadRequestException", http.StatusBadRequest},

		// Service errors
		{"InternalServerError", http.StatusInternalServerError},
		{"ServiceUnavailableException", http.StatusInternalServerError},
		{"InternalFailure", http.StatusInternalServerError},
		{"ServiceException", http.StatusInternalServerError},

		// Resource not found
		{"ResourceNotFoundException", http.StatusNotFound},
		{"ModelNotFoundException", http.StatusNotFound},
		{"NotFoundException", http.StatusNotFound},

		// Unknown error code
		{"UnknownErrorCode", http.StatusInternalServerError},
		{"", http.StatusInternalServerError},
	}

	for _, tc := range testCases {
		t.Run(tc.errorCode, func(t *testing.T) {
			awsErr := &mockAPIError{
				code:    tc.errorCode,
				message: "test error",
			}

			statusCode := getStatusCodeFromAWSError(awsErr)
			require.Equal(t, tc.expectedStatusCode, statusCode,
				"Status code mismatch for error code: %s", tc.errorCode)
		})
	}
}

// Note: mockAPIError is defined in properties_test.go and shared across test files
