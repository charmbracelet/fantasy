package fantasy

import (
	"context"
	"errors"
	"net"
	"net/http"
	"strconv"
	"time"
)

// RetryFn is a function that returns a value and an error.
type RetryFn[T any] func() (T, error)

// RetryFunction is a function that retries another function.
type RetryFunction[T any] func(ctx context.Context, fn RetryFn[T]) (T, error)

// getRetryDelayInMs calculates the retry delay based on error headers and exponential backoff.
func getRetryDelayInMs(err error, exponentialBackoffDelay time.Duration) time.Duration {
	var providerErr *ProviderError
	if !errors.As(err, &providerErr) || providerErr.ResponseHeaders == nil {
		return exponentialBackoffDelay
	}

	headers := providerErr.ResponseHeaders
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

// RetryWithExponentialBackoffRespectingRetryHeaders creates a retry function that retries
// a failed operation with exponential backoff, while respecting rate limit headers
// (retry-after-ms and retry-after) if they are provided and reasonable (0-60 seconds).
//
// When OnAuthRefresh is set and the operation ends in an authentication error,
// the hook is given one chance to refresh credentials. On success the entire
// retry pass runs again with a fresh budget; on failure the original auth
// error is returned. At most one refresh is attempted, so a credential that
// stays invalid cannot spin.
func RetryWithExponentialBackoffRespectingRetryHeaders[T any](options RetryOptions) RetryFunction[T] {
	return func(ctx context.Context, fn RetryFn[T]) (T, error) {
		result, err := retryWithExponentialBackoff(ctx, fn, options, nil)
		if err == nil || options.OnAuthRefresh == nil {
			return result, err
		}
		var authErr *ProviderError
		if !errors.As(err, &authErr) || !isAuthError(authErr) {
			return result, err
		}
		if refreshErr := options.OnAuthRefresh(ctx, authErr); refreshErr != nil {
			return result, err // refresh failed: surface the original auth error
		}
		return retryWithExponentialBackoff(ctx, fn, options, nil)
	}
}

// RetryOptions configures the retry behavior.
type RetryOptions struct {
	MaxRetries     int
	InitialDelayIn time.Duration
	BackoffFactor  float64
	OnRetry        OnRetryCallback

	// OnAuthRefresh is called when an operation fails with an authentication
	// error the caller may be able to resolve (e.g. an expired SSO session).
	// If it returns nil, the entire retry pass restarts with a fresh retry
	// budget; if it returns an error, the original auth error is returned
	// without retry. At most one refresh is attempted, since auth refresh is
	// a one-shot human-in-the-loop step and a second attempt would not fare
	// better.
	OnAuthRefresh OnAuthRefreshFunc
}

// OnRetryCallback is called before each retry attempt, after the retry
// delay is chosen but before it elapses. err is the failure that triggered
// the retry (nil if the failure was not a *ProviderError) and delay is how
// long the middleware will wait before the next attempt.
//
// A retry re-runs the entire step from scratch: the stream is recreated and
// the stream callbacks (OnTextStart, OnTextDelta, OnReasoningStart,
// OnReasoningDelta, OnToolInputStart, etc.) fire again from the beginning of
// the new response. Consumers that accumulate streamed content must reset
// that accumulated state here, otherwise the retried response is appended to
// the partial content from the failed attempt.
type OnRetryCallback = func(err *ProviderError, delay time.Duration)

// DefaultRetryOptions returns the default retry options.
func DefaultRetryOptions() RetryOptions {
	return RetryOptions{
		MaxRetries:     3,
		InitialDelayIn: 5000 * time.Millisecond,
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

	newErrors := append(allErrors, err)
	tryNumber := len(newErrors)

	if tryNumber > options.MaxRetries {
		return zero, &RetryError{newErrors}
	}

	var providerErr *ProviderError
	if isRetryableError(err) && tryNumber <= options.MaxRetries {
		delay := getRetryDelayInMs(err, options.InitialDelayIn)
		if options.OnRetry != nil {
			errors.As(err, &providerErr)
			options.OnRetry(providerErr, delay)
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

	return zero, &RetryError{newErrors}
}

// isAuthError reports whether the error is an authentication failure that a
// caller-supplied OnAuthRefresh hook may be able to resolve.
func isAuthError(err *ProviderError) bool {
	return err.StatusCode == http.StatusUnauthorized || err.AuthError
}

// isRetryableError reports whether the error should be retried.
// It checks for retryable ProviderError, network-level connection errors
// (DNS failures, TCP timeouts, connection refused), and HTTP/2 stream-
// level transport errors. The latter two categories may not be wrapped
// in ProviderError when they occur outside the provider's error handler.
func isRetryableError(err error) bool {
	var providerErr *ProviderError
	if errors.As(err, &providerErr) {
		return providerErr.IsRetryable()
	}
	if isAbortError(err) {
		return false
	}
	var netErr net.Error
	if errors.As(err, &netErr) {
		return true
	}
	return IsTransportError(err)
}

// isAbortError checks if the error is a context cancellation error.
func isAbortError(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}
