package fantasy

import (
	"context"
	"errors"
	"net"
	"testing"
	"time"
)

func TestIsRetryableError(t *testing.T) {
	t.Parallel()

	t.Run("retryable ProviderError", func(t *testing.T) {
		t.Parallel()
		err := &ProviderError{
			StatusCode: 429,
			Message:    "rate limited",
		}
		if !isRetryableError(err) {
			t.Error("expected retryable ProviderError to be retryable")
		}
	})

	t.Run("non-retryable ProviderError", func(t *testing.T) {
		t.Parallel()
		err := &ProviderError{
			StatusCode: 400,
			Message:    "bad request",
		}
		if isRetryableError(err) {
			t.Error("expected non-retryable ProviderError to not be retryable")
		}
	})

	t.Run("net.OpError is retryable", func(t *testing.T) {
		t.Parallel()
		err := &net.OpError{Op: "dial", Net: "tcp", Err: errors.New("connection refused")}
		if !isRetryableError(err) {
			t.Error("expected net.OpError to be retryable")
		}
	})

	t.Run("context.Canceled is not retryable", func(t *testing.T) {
		t.Parallel()
		err := context.Canceled
		if isRetryableError(err) {
			t.Error("expected context.Canceled to not be retryable")
		}
	})

	t.Run("context.DeadlineExceeded is not retryable", func(t *testing.T) {
		t.Parallel()
		err := context.DeadlineExceeded
		if isRetryableError(err) {
			t.Error("expected context.DeadlineExceeded to not be retryable")
		}
	})

	t.Run("generic error is not retryable", func(t *testing.T) {
		t.Parallel()
		err := errors.New("something went wrong")
		if isRetryableError(err) {
			t.Error("expected generic error to not be retryable")
		}
	})
}

func TestRetryWithExponentialBackoff_ConnectionErrors(t *testing.T) {
	t.Parallel()

	t.Run("retries on net.OpError", func(t *testing.T) {
		t.Parallel()
		attempts := 0
		opts := RetryOptions{
			MaxRetries:     2,
			InitialDelayIn: 1 * time.Millisecond,
			BackoffFactor:  2.0,
		}

		retryFn := RetryWithExponentialBackoffRespectingRetryHeaders[int](opts)

		result, err := retryFn(context.Background(), func() (int, error) {
			attempts++
			if attempts < 3 {
				return 0, &net.OpError{Op: "dial", Net: "tcp", Err: errors.New("connection refused")}
			}
			return 42, nil
		})

		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		if result != 42 {
			t.Fatalf("expected result 42, got %d", result)
		}
		if attempts != 3 {
			t.Fatalf("expected 3 attempts, got %d", attempts)
		}
	})

	t.Run("does not retry on context.Canceled", func(t *testing.T) {
		t.Parallel()
		attempts := 0
		opts := RetryOptions{
			MaxRetries:     2,
			InitialDelayIn: 1 * time.Millisecond,
			BackoffFactor:  2.0,
		}

		retryFn := RetryWithExponentialBackoffRespectingRetryHeaders[int](opts)

		_, err := retryFn(context.Background(), func() (int, error) {
			attempts++
			return 0, context.Canceled
		})

		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if attempts != 1 {
			t.Fatalf("expected 1 attempt (no retry), got %d", attempts)
		}
	})

	t.Run("returns RetryError when max retries exceeded on connection error", func(t *testing.T) {
		t.Parallel()
		opts := RetryOptions{
			MaxRetries:     2,
			InitialDelayIn: 1 * time.Millisecond,
			BackoffFactor:  2.0,
		}

		retryFn := RetryWithExponentialBackoffRespectingRetryHeaders[int](opts)

		_, err := retryFn(context.Background(), func() (int, error) {
			return 0, &net.OpError{Op: "dial", Net: "tcp", Err: errors.New("connection refused")}
		})

		var retryErr *RetryError
		if !errors.As(err, &retryErr) {
			t.Fatalf("expected RetryError, got %T: %v", err, err)
		}
	})
}
