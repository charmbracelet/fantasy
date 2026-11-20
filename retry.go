package fantasy

import (
	"context"
	"errors"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// RetryFn is a function that returns a value and an error.
type RetryFn[T any] func() (T, error)

// RetryFunction is a function that retries another function.
type RetryFunction[T any] func(ctx context.Context, fn RetryFn[T]) (T, error)

const maxShortRetryHint = 60 * time.Second

// retryHint parses the provider's retry headers without applying backoff.
func retryHint(headers map[string]string) time.Duration {
	// Millisecond hints take precedence over Retry-After.
	if value := retryHeader(headers, "retry-after-ms"); value != "" {
		if ms, err := strconv.ParseFloat(value, 64); err == nil {
			if delay := time.Duration(ms * float64(time.Millisecond)); delay != 0 {
				return delay
			}
		}
	}
	if value := retryHeader(headers, "retry-after"); value != "" {
		if seconds, err := strconv.ParseFloat(value, 64); err == nil {
			return time.Duration(seconds * float64(time.Second))
		}
		if at, err := time.Parse(time.RFC1123, value); err == nil {
			return time.Until(at)
		}
	}
	return 0
}

func retryHeader(headers map[string]string, name string) string {
	if value, ok := headers[name]; ok {
		return value
	}
	for key, value := range headers {
		if strings.EqualFold(key, name) {
			return value
		}
	}
	return ""
}

// getRetryDelayInMs selects a delay for an error using the default retry policy.
func getRetryDelayInMs(err error, backoff time.Duration) time.Duration {
	if providerErr, ok := errors.AsType[*ProviderError](err); ok {
		if delay := retryHint(providerErr.ResponseHeaders); delay > 0 && (delay < maxShortRetryHint || delay < backoff) {
			return delay
		}
	}
	return backoff
}

func shortRetryHint(headers map[string]string) time.Duration {
	delay := retryHint(headers)
	if delay > 0 && delay < maxShortRetryHint {
		return delay
	}
	return 0
}

func retryDelay(err error, backoff time.Duration) (time.Duration, bool) {
	// Use the first RetryableError, just as isRetryableError does, so a
	// wrapper can still override the provider's policy.
	if decision, ok := errors.AsType[RetryableError](err); ok {
		if providerErr, ok := decision.(*ProviderError); ok && providerErr.RetryPolicy == RetryWithHint {
			delay := shortRetryHint(providerErr.ResponseHeaders)
			return delay, delay > 0
		}
	}
	if !isRetryableError(err) {
		return 0, false
	}
	return getRetryDelayInMs(err, backoff), true
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
		authErr, ok := errors.AsType[*ProviderError](err)
		if !ok || !isAuthError(authErr) {
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

	if delay, retry := retryDelay(err, options.InitialDelayIn); retry {
		if options.OnRetry != nil {
			providerErr, _ := errors.AsType[*ProviderError](err)
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

// RetryableError is an error that decides for itself whether its request is
// worth retrying. An error that wraps a context error is never retried.
type RetryableError interface {
	error
	IsRetryable() bool
}

// isRetryableError reports whether the error should be retried.
// A [RetryableError] in the chain, such as a ProviderError, decides first.
// Otherwise it checks for network-level connection errors (DNS failures,
// TCP timeouts, connection refused), and HTTP/2 stream-level transport
// errors. The latter two categories may not be wrapped in ProviderError
// when they occur outside the provider's error handler.
func isRetryableError(err error) bool {
	if retryableErr, ok := errors.AsType[RetryableError](err); ok {
		return retryableErr.IsRetryable()
	}
	if isAbortError(err) {
		return false
	}
	if _, ok := errors.AsType[net.Error](err); ok {
		return true
	}
	return IsTransportError(err)
}

// isAbortError checks if the error is a context cancellation error.
func isAbortError(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}
