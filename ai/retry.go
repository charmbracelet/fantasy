package ai

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"time"
)

// RetryFn is a function that returns a value and an error.
type RetryFn[T any] func() (T, error)

// RetryFunction is a function that retries another function.
type RetryFunction[T any] func(ctx context.Context, fn RetryFn[T]) (T, error)

// RetryReason represents the reason why a retry operation failed.
type RetryReason string

const (
	RetryReasonMaxRetriesExceeded RetryReason = "maxRetriesExceeded"
	RetryReasonErrorNotRetryable  RetryReason = "errorNotRetryable"
)

// RetryError represents an error that occurred during retry operations.
type RetryError struct {
	*AIError
	Reason RetryReason
	Errors []error
}

// NewRetryError creates a new retry error.
func NewRetryError(message string, reason RetryReason, errors []error) *RetryError {
	return &RetryError{
		AIError: NewAIError("AI_RetryError", message, nil),
		Reason:  reason,
		Errors:  errors,
	}
}

// getRetryDelayInMs calculates the retry delay based on error headers and exponential backoff.
func getRetryDelayInMs(err error, exponentialBackoffDelay time.Duration) time.Duration {
	var apiErr *APICallError
	if !errors.As(err, &apiErr) || apiErr.ResponseHeaders == nil {
		return exponentialBackoffDelay
	}

	headers := apiErr.ResponseHeaders
	var ms time.Duration

	// retry-ms is more precise than retry-after and used by e.g. OpenAI
	if retryAfterMs, exists := headers["retry-after-ms"]; exists {
		if timeoutMs, err := strconv.ParseFloat(retryAfterMs, 64); err == nil {
			ms = time.Duration(timeoutMs) * time.Millisecond
		}
	}

	// About the Retry-After header: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Retry-After
	if retryAfter, exists := headers["retry-after"]; exists && ms == 0 {
		if timeoutSeconds, err := strconv.ParseFloat(retryAfter, 64); err == nil {
			ms = time.Duration(timeoutSeconds) * time.Second
		} else {
			// Try parsing as HTTP date
			if t, err := time.Parse(time.RFC1123, retryAfter); err == nil {
				ms = time.Until(t)
			}
		}
	}

	// Check that the delay is reasonable:
	// 0 <= ms < 60 seconds or ms < exponentialBackoffDelay
	if ms > 0 && (ms < 60*time.Second || ms < exponentialBackoffDelay) {
		return ms
	}

	return exponentialBackoffDelay
}

// isAbortError checks if the error is a context cancellation error.
func isAbortError(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}

// RetryWithExponentialBackoffRespectingRetryHeaders creates a retry function that retries
// a failed operation with exponential backoff, while respecting rate limit headers
// (retry-after-ms and retry-after) if they are provided and reasonable (0-60 seconds).
func RetryWithExponentialBackoffRespectingRetryHeaders[T any](options RetryOptions) RetryFunction[T] {
	return func(ctx context.Context, fn RetryFn[T]) (T, error) {
		return retryWithExponentialBackoff(ctx, fn, options, nil)
	}
}

// RetryOptions configures the retry behavior.
type RetryOptions struct {
	MaxRetries     int
	InitialDelayIn time.Duration
	BackoffFactor  float64
	OnRetry        OnRetryCallback
}

type OnRetryCallback = func(err *APICallError, delay time.Duration)

// DefaultRetryOptions returns the default retry options.
func DefaultRetryOptions() RetryOptions {
	return RetryOptions{
		MaxRetries:     2,
		InitialDelayIn: 2000 * time.Millisecond,
		BackoffFactor:  2.0,
	}
}

// retryWithExponentialBackoff implements the retry logic with exponential backoff.
func retryWithExponentialBackoff[T any](ctx context.Context, fn RetryFn[T], options RetryOptions, allErrors []error) (T, error) {
	var zero T
	result, err := fn()
	if err == nil {
		return result, nil
	}

	if isAbortError(err) {
		return zero, err // don't retry when the request was aborted
	}

	if options.MaxRetries == 0 {
		return zero, err // don't wrap the error when retries are disabled
	}

	errorMessage := GetErrorMessage(err)
	newErrors := append(allErrors, err)
	tryNumber := len(newErrors)

	if tryNumber > options.MaxRetries {
		return zero, NewRetryError(
			fmt.Sprintf("Failed after %d attempts. Last error: %s", tryNumber, errorMessage),
			RetryReasonMaxRetriesExceeded,
			newErrors,
		)
	}

	var apiErr *APICallError
	if errors.As(err, &apiErr) && apiErr.IsRetryable && tryNumber <= options.MaxRetries {
		delay := getRetryDelayInMs(err, options.InitialDelayIn)
		if options.OnRetry != nil {
			options.OnRetry(apiErr, delay)
		}

		select {
		case <-time.After(delay):
			// Continue with retry
		case <-ctx.Done():
			return zero, ctx.Err()
		}

		newOptions := options
		newOptions.InitialDelayIn = time.Duration(float64(options.InitialDelayIn) * options.BackoffFactor)

		return retryWithExponentialBackoff(ctx, fn, newOptions, newErrors)
	}

	if tryNumber == 1 {
		return zero, err // don't wrap the error when a non-retryable error occurs on the first try
	}

	return zero, NewRetryError(
		fmt.Sprintf("Failed after %d attempts with non-retryable error: '%s'", tryNumber, errorMessage),
		RetryReasonErrorNotRetryable,
		newErrors,
	)
}
