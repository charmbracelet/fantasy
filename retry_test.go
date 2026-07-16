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

	t.Run("HTTP/2 stream error is retryable", func(t *testing.T) {
		t.Parallel()
		err := errors.New("stream error: stream ID 27; INTERNAL_ERROR; received from peer")
		if !isRetryableError(err) {
			t.Error("expected HTTP/2 stream error to be retryable")
		}
	})

	t.Run("HTTP/2 connection error is retryable", func(t *testing.T) {
		t.Parallel()
		err := errors.New("connection error: INTERNAL_ERROR")
		if !isRetryableError(err) {
			t.Error("expected HTTP/2 connection error to be retryable")
		}
	})

	t.Run("ProviderError wrapping HTTP/2 stream error is retryable", func(t *testing.T) {
		t.Parallel()
		rawErr := errors.New("stream error: stream ID 5; REFUSED_STREAM")
		err := &ProviderError{
			Title:   "stream transport error",
			Message: "REFUSED_STREAM",
			Cause:   rawErr,
		}
		if !isRetryableError(err) {
			t.Error("expected ProviderError wrapping HTTP/2 stream error to be retryable")
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

	t.Run("retries on HTTP/2 stream error", func(t *testing.T) {
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
				return 0, errors.New("stream error: stream ID 27; INTERNAL_ERROR; received from peer")
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
}

func TestRetryWithAuthRefresh(t *testing.T) {
	t.Parallel()

	authErr := func() error { return &ProviderError{StatusCode: 401, Message: "unauthorized"} }

	t.Run("refreshes and retries transparently on a flagged non-401 auth error", func(t *testing.T) {
		t.Parallel()
		attempts := 0
		refreshed := 0
		opts := RetryOptions{
			MaxRetries:     2,
			InitialDelayIn: 1 * time.Millisecond,
			BackoffFactor:  2.0,
			OnAuthRefresh: func(_ context.Context, _ *ProviderError) error {
				refreshed++
				return nil
			},
		}

		retryFn := RetryWithExponentialBackoffRespectingRetryHeaders[int](opts)
		result, err := retryFn(context.Background(), func() (int, error) {
			attempts++
			if attempts == 1 {
				// No 401 status; classified as auth purely via the flag,
				// mirroring an expired AWS SSO session.
				return 0, &ProviderError{Message: "failed to refresh cached credentials", AuthError: true}
			}
			return 7, nil
		})
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		if result != 7 {
			t.Fatalf("expected result 7, got %d", result)
		}
		if refreshed != 1 {
			t.Fatalf("expected 1 refresh, got %d", refreshed)
		}
		if attempts != 2 {
			t.Fatalf("expected 2 attempts, got %d", attempts)
		}
	})

	t.Run("refreshes and retries transparently on 401", func(t *testing.T) {
		t.Parallel()
		attempts := 0
		refreshed := 0
		opts := RetryOptions{
			MaxRetries:     2,
			InitialDelayIn: 1 * time.Millisecond,
			BackoffFactor:  2.0,
			OnAuthRefresh: func(_ context.Context, _ *ProviderError) error {
				refreshed++
				return nil
			},
		}

		retryFn := RetryWithExponentialBackoffRespectingRetryHeaders[int](opts)
		result, err := retryFn(context.Background(), func() (int, error) {
			attempts++
			if attempts == 1 {
				return 0, authErr()
			}
			return 7, nil
		})
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		if result != 7 {
			t.Fatalf("expected result 7, got %d", result)
		}
		if refreshed != 1 {
			t.Fatalf("expected 1 refresh, got %d", refreshed)
		}
		if attempts != 2 {
			t.Fatalf("expected 2 attempts, got %d", attempts)
		}
	})

	t.Run("returns original error when refresh fails", func(t *testing.T) {
		t.Parallel()
		attempts := 0
		opts := RetryOptions{
			MaxRetries:     2,
			InitialDelayIn: 1 * time.Millisecond,
			BackoffFactor:  2.0,
			OnAuthRefresh: func(_ context.Context, _ *ProviderError) error {
				return errors.New("login canceled")
			},
		}

		retryFn := RetryWithExponentialBackoffRespectingRetryHeaders[int](opts)
		_, err := retryFn(context.Background(), func() (int, error) {
			attempts++
			return 0, authErr()
		})

		var providerErr *ProviderError
		if !errors.As(err, &providerErr) || providerErr.StatusCode != 401 {
			t.Fatalf("expected original 401 ProviderError, got %T: %v", err, err)
		}
		if attempts != 1 {
			t.Fatalf("expected 1 attempt (no retry after failed refresh), got %d", attempts)
		}
	})

	t.Run("attempts at most one refresh", func(t *testing.T) {
		t.Parallel()
		attempts := 0
		refreshed := 0
		opts := RetryOptions{
			MaxRetries:     2,
			InitialDelayIn: 1 * time.Millisecond,
			BackoffFactor:  2.0,
			OnAuthRefresh: func(_ context.Context, _ *ProviderError) error {
				refreshed++
				return nil
			},
		}

		retryFn := RetryWithExponentialBackoffRespectingRetryHeaders[int](opts)
		_, err := retryFn(context.Background(), func() (int, error) {
			attempts++
			return 0, authErr()
		})
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if refreshed != 1 {
			t.Fatalf("expected exactly 1 refresh, got %d", refreshed)
		}
		// 1 initial + 1 retry after refresh = 2 attempts.
		if attempts != 2 {
			t.Fatalf("expected 2 attempts, got %d", attempts)
		}
	})

	t.Run("does not refresh on non-auth errors", func(t *testing.T) {
		t.Parallel()
		refreshed := 0
		opts := RetryOptions{
			MaxRetries:     2,
			InitialDelayIn: 1 * time.Millisecond,
			BackoffFactor:  2.0,
			OnAuthRefresh: func(_ context.Context, _ *ProviderError) error {
				refreshed++
				return nil
			},
		}

		retryFn := RetryWithExponentialBackoffRespectingRetryHeaders[int](opts)
		_, _ = retryFn(context.Background(), func() (int, error) {
			return 0, &ProviderError{StatusCode: 429, Message: "rate limited"}
		})
		if refreshed != 0 {
			t.Fatalf("expected no refresh on 429, got %d", refreshed)
		}
	})

	t.Run("auth retry does not consume general retry budget", func(t *testing.T) {
		t.Parallel()
		attempts := 0
		opts := RetryOptions{
			MaxRetries:     2,
			InitialDelayIn: 1 * time.Millisecond,
			BackoffFactor:  2.0,
			OnAuthRefresh: func(_ context.Context, _ *ProviderError) error {
				return nil
			},
		}

		retryFn := RetryWithExponentialBackoffRespectingRetryHeaders[int](opts)
		result, err := retryFn(context.Background(), func() (int, error) {
			attempts++
			switch attempts {
			case 1:
				return 0, authErr() // resolved via refresh, not the retry budget
			case 2, 3:
				return 0, &ProviderError{StatusCode: 500, Message: "server error"}
			default:
				return 99, nil
			}
		})
		if err != nil {
			t.Fatalf("expected success after auth refresh + 2 transport retries, got %v", err)
		}
		if result != 99 {
			t.Fatalf("expected result 99, got %d", result)
		}
		if attempts != 4 {
			t.Fatalf("expected 4 attempts, got %d", attempts)
		}
	})
}

func TestIsAuthError(t *testing.T) {
	t.Parallel()

	if !isAuthError(&ProviderError{StatusCode: 401}) {
		t.Error("expected 401 to be an auth error")
	}
	if isAuthError(&ProviderError{StatusCode: 403}) {
		t.Error("expected 403 to not be an auth error")
	}
	if isAuthError(&ProviderError{StatusCode: 500}) {
		t.Error("expected 500 to not be an auth error")
	}
	if !isAuthError(&ProviderError{AuthError: true}) {
		t.Error("expected an AuthError-flagged error to be an auth error")
	}
}
