package anthropic

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"testing"

	"charm.land/fantasy"
	"github.com/anthropics/anthropic-sdk-go"
)

func TestToHeaderMap_LowercasesKeys(t *testing.T) {
	t.Parallel()

	in := http.Header{
		"Retry-After":    []string{"30"},
		"Retry-After-Ms": []string{"1500"},
	}

	out := toHeaderMap(in)

	if got := out["retry-after"]; got != "30" {
		t.Errorf(`out["retry-after"] = %q, want "30" (retry.go looks up headers by lowercase key)`, got)
	}
	if got := out["retry-after-ms"]; got != "1500" {
		t.Errorf(`out["retry-after-ms"] = %q, want "1500"`, got)
	}
}

func TestToProviderErr_WrapsUnexpectedEOF(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		err  error
	}{
		{"direct", io.ErrUnexpectedEOF},
		{"wrapped", fmt.Errorf("read stream: %w", io.ErrUnexpectedEOF)},
		{"double_wrapped", fmt.Errorf("anthropic: %w", fmt.Errorf("sse: %w", io.ErrUnexpectedEOF))},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			got := toProviderErr(tc.err)

			var providerErr *fantasy.ProviderError
			if !errors.As(got, &providerErr) {
				t.Fatalf("toProviderErr did not wrap %v as *fantasy.ProviderError (got %T)", tc.err, got)
			}
			if !errors.Is(providerErr.Cause, io.ErrUnexpectedEOF) {
				t.Errorf("ProviderError.Cause = %v, want chain containing io.ErrUnexpectedEOF", providerErr.Cause)
			}
			if !providerErr.IsRetryable() {
				t.Error("wrapped io.ErrUnexpectedEOF must be retryable so retry.go engages")
			}
		})
	}
}

func TestToProviderErr_PassesThroughUnrelatedErrors(t *testing.T) {
	t.Parallel()

	err := errors.New("something unrelated")
	got := toProviderErr(err)
	if got != err {
		t.Errorf("toProviderErr mutated unrelated error: got %v, want %v", got, err)
	}
}

func TestToProviderErr_PassesThroughPlainEOF(t *testing.T) {
	t.Parallel()

	// A clean io.EOF at the end of a stream is not a failure — the streaming
	// handler in anthropic.go treats it as a normal terminator and never
	// calls toProviderErr with io.EOF. But if it ever did, we should not
	// wrap it: io.EOF is not "retryable" in the ProviderError sense.
	got := toProviderErr(io.EOF)
	var providerErr *fantasy.ProviderError
	if errors.As(got, &providerErr) {
		t.Errorf("toProviderErr wrapped io.EOF as ProviderError; should pass through")
	}
}

func TestToProviderErr_FlagsExpiredBedrockCredentials(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		err  error
	}{
		{"direct", errors.New("failed to refresh cached credentials")},
		{"wrapped", fmt.Errorf("operation error Bedrock: %w", errors.New("failed to refresh cached credentials"))},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			var providerErr *fantasy.ProviderError
			if !errors.As(toProviderErr(tc.err), &providerErr) {
				t.Fatalf("toProviderErr did not wrap %v as *fantasy.ProviderError", tc.err)
			}
			if !providerErr.AuthError {
				t.Error("expected AuthError flag so OnAuthRefresh engages")
			}
		})
	}
}

// A mid-stream SSE error event rides inside a 200 response, so only the
// payload marks the failure as temporary.
func TestToProviderErr_RetriesMidStreamOverload(t *testing.T) {
	t.Parallel()

	err := errors.New(`received error while streaming: {"type":"error","error":{"details":null,"type":"overloaded_error","message":"Overloaded"}}`)

	var providerErr *fantasy.ProviderError
	if !errors.As(toProviderErr(err), &providerErr) {
		t.Fatalf("toProviderErr did not wrap %v as *fantasy.ProviderError", err)
	}
	if !providerErr.IsRetryable() {
		t.Error("a mid-stream overload must be retryable so the step is re-run")
	}
	if !providerErr.TransientError {
		t.Error("TransientError must be set for a transient stream error")
	}
	if providerErr.StatusCode != 0 {
		t.Errorf("StatusCode = %d, want 0 (no HTTP status was returned)", providerErr.StatusCode)
	}
	if providerErr.Title != "provider overloaded" {
		t.Errorf("Title = %q, want %q", providerErr.Title, "provider overloaded")
	}
	if providerErr.Message != "Overloaded" {
		t.Errorf("Message = %q, want %q", providerErr.Message, "Overloaded")
	}
	if !errors.Is(providerErr.Cause, err) {
		t.Error("Cause chain must include the original error")
	}
}

func TestToProviderErr_StreamErrorTransientTypes(t *testing.T) {
	t.Parallel()

	for _, errType := range []string{"overloaded_error", "api_error", "server_error", "internal_error", "rate_limit_error"} {
		t.Run(errType, func(t *testing.T) {
			t.Parallel()

			err := errors.New(`received error while streaming: {"type":"error","error":{"type":"` + errType + `","message":"transient"}}`)

			var providerErr *fantasy.ProviderError
			if !errors.As(toProviderErr(err), &providerErr) {
				t.Fatalf("toProviderErr did not wrap %v as *fantasy.ProviderError", err)
			}
			if !providerErr.IsRetryable() {
				t.Errorf("%s stream failure must be retryable", errType)
			}
		})
	}
}

func TestToProviderErr_PermanentStreamErrorNotRetried(t *testing.T) {
	t.Parallel()

	err := errors.New(`received error while streaming: {"type":"error","error":{"type":"invalid_request_error","message":"bad tool schema"}}`)

	var providerErr *fantasy.ProviderError
	if !errors.As(toProviderErr(err), &providerErr) {
		t.Fatalf("toProviderErr did not wrap %v as *fantasy.ProviderError", err)
	}
	if providerErr.IsRetryable() {
		t.Error("a permanent stream error type must not be retryable")
	}
	if providerErr.Message != "bad tool schema" {
		t.Errorf("Message = %q, want %q", providerErr.Message, "bad tool schema")
	}
}

func TestToProviderErr_StreamErrorWrappedByOuterError(t *testing.T) {
	t.Parallel()

	inner := errors.New(`received error while streaming: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`)
	err := fmt.Errorf("anthropic: %w", inner)

	var providerErr *fantasy.ProviderError
	if !errors.As(toProviderErr(err), &providerErr) {
		t.Fatalf("toProviderErr did not wrap %v as *fantasy.ProviderError", err)
	}
	if !providerErr.IsRetryable() {
		t.Error("a wrapped mid-stream overload must still be retryable")
	}
}

// The same overload, arriving the way Anthropic actually sends it.
//
// Anthropic accepts the request and answers 200, opens the stream, and only
// then sends an overloaded_error event down it. The SDK turns that event into
// a typed error carrying the status of the response it arrived in, so the
// status says the exchange succeeded while the body says the provider could
// not serve it. Titling from the status put "ok" above the error.
//
// Driving a real stream rather than hand-building the SDK error is the point:
// the status on a mid-stream event is the SDK's behaviour, not ours, and a
// hand-built error would assume the very thing worth checking.
func TestToProviderErr_TitlesMidStreamOverloadFromThePayload(t *testing.T) {
	t.Parallel()

	server, _ := newAnthropicStreamingServer([]string{
		anthropicSSEEvent("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}`),
		anthropicSSEEvent("error", `{"type":"error","error":{"details":null,"type":"overloaded_error","message":"Overloaded"}}`),
	})
	defer server.Close()

	provider, err := New(WithAPIKey("test-api-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	model, err := provider.LanguageModel(context.Background(), "claude-sonnet-4-20250514")
	if err != nil {
		t.Fatalf("LanguageModel: %v", err)
	}
	stream, err := model.Stream(context.Background(), fantasy.Call{Prompt: testPrompt()})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	var streamErr error
	for _, part := range collectAnthropicStreamParts(stream) {
		if part.Type == fantasy.StreamPartTypeError {
			streamErr = part.Error
		}
	}
	if streamErr == nil {
		t.Fatal("the stream reported no error, so the overload never surfaced")
	}

	var providerErr *fantasy.ProviderError
	if !errors.As(streamErr, &providerErr) {
		t.Fatalf("stream error is %T, want *fantasy.ProviderError", streamErr)
	}

	// The premise: the SDK really does carry the response's status here.
	if providerErr.StatusCode != http.StatusOK {
		t.Fatalf("StatusCode = %d, want 200; a mid-stream event is meant to "+
			"arrive inside a successful response", providerErr.StatusCode)
	}
	if providerErr.Title != "provider overloaded" {
		t.Errorf("Title = %q, want %q", providerErr.Title, "provider overloaded")
	}
	if !providerErr.TransientError {
		t.Error("TransientError must stay set so the step is retried")
	}
	if !providerErr.IsRetryable() {
		t.Error("a mid-stream overload must stay retryable")
	}
}

// The status names the failure where it can, since "too many requests" beats
// anything derived from a body. Where it cannot, the payload's type is the
// next best thing and a generic title is the last resort.
func TestAPIErrorTitle(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name    string
		status  int
		errType string
		want    string
	}{
		{"a named failure status wins", http.StatusTooManyRequests, "rate_limit_error", "too many requests"},
		{"even over a type that disagrees", http.StatusUnauthorized, "overloaded_error", "unauthorized"},
		{"a named status needs no type", http.StatusInternalServerError, "", "internal server error"},

		// Anthropic answers an overload with 529, which the HTTP registry
		// does not name, so without the payload this reads "provider request
		// failed" while the error says overloaded_error two fields away.
		{"an unnamed status defers to the type", 529, "overloaded_error", "provider overloaded"},
		{"an unnamed status with no type is generic", 529, "", "provider request failed"},

		// A 2xx carried a mid-stream failure: its status describes the
		// exchange, so only the payload can name the error.
		{"a 2xx never titles from the status", http.StatusOK, "overloaded_error", "provider overloaded"},
		{"a 2xx with no type says so", http.StatusOK, "", "provider stream error"},

		{"no HTTP exchange happened", 0, "", "provider request failed"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			if got := apiErrorTitle(apiErrorWithType(t, tc.status, tc.errType)); got != tc.want {
				t.Errorf("apiErrorTitle(status %d, type %q) = %q, want %q",
					tc.status, tc.errType, got, tc.want)
			}
		})
	}
}

// apiErrorWithType builds an SDK error carrying errType. The field behind
// Type() is unexported, so it has to arrive the way the SDK fills it: by
// decoding an error envelope.
func apiErrorWithType(t *testing.T, status int, errType string) *anthropic.Error {
	t.Helper()

	apiErr := &anthropic.Error{}
	if errType != "" {
		envelope := fmt.Sprintf(`{"type":"error","error":{"type":%q,"message":"x"}}`, errType)
		if err := apiErr.UnmarshalJSON([]byte(envelope)); err != nil {
			t.Fatalf("UnmarshalJSON(%s): %v", envelope, err)
		}
		if got := string(apiErr.Type()); got != errType {
			t.Fatalf("the SDK did not take the type: Type() = %q, want %q", got, errType)
		}
	}
	apiErr.StatusCode = status
	return apiErr
}

// ErrorType carries the provider's own name for the failure so callers do not
// have to read it back out of Message, which is formatted for people.
func TestToProviderErr_CarriesTheErrorType(t *testing.T) {
	t.Parallel()

	t.Run("typed SDK error", func(t *testing.T) {
		t.Parallel()

		server, _ := newAnthropicStreamingServer([]string{
			anthropicSSEEvent("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}`),
			anthropicSSEEvent("error", `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`),
		})
		defer server.Close()

		providerErr := streamProviderError(t, server.URL)
		if providerErr.ErrorType != "overloaded_error" {
			t.Errorf("ErrorType = %q, want %q", providerErr.ErrorType, "overloaded_error")
		}
	})

	t.Run("untyped stream error", func(t *testing.T) {
		t.Parallel()

		err := errors.New(streamErrorPrefix + ` {"type":"error","error":{"type":"api_error","message":"Internal"}}`)

		var providerErr *fantasy.ProviderError
		if !errors.As(toProviderErr(err), &providerErr) {
			t.Fatalf("toProviderErr did not wrap %v as *fantasy.ProviderError", err)
		}
		if providerErr.ErrorType != "api_error" {
			t.Errorf("ErrorType = %q, want %q", providerErr.ErrorType, "api_error")
		}
	})
}

// streamProviderError drives one stream against url and returns the
// ProviderError it reported.
func streamProviderError(t *testing.T, url string) *fantasy.ProviderError {
	t.Helper()

	provider, err := New(WithAPIKey("test-api-key"), WithBaseURL(url))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	model, err := provider.LanguageModel(context.Background(), "claude-sonnet-4-20250514")
	if err != nil {
		t.Fatalf("LanguageModel: %v", err)
	}
	stream, err := model.Stream(context.Background(), fantasy.Call{Prompt: testPrompt()})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	var streamErr error
	for _, part := range collectAnthropicStreamParts(stream) {
		if part.Type == fantasy.StreamPartTypeError {
			streamErr = part.Error
		}
	}
	if streamErr == nil {
		t.Fatal("the stream reported no error")
	}
	var providerErr *fantasy.ProviderError
	if !errors.As(streamErr, &providerErr) {
		t.Fatalf("stream error is %T, want *fantasy.ProviderError", streamErr)
	}
	return providerErr
}
