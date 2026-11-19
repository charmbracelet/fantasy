package openai

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"charm.land/fantasy"
	"github.com/charmbracelet/openai-go"
	"github.com/charmbracelet/openai-go/packages/ssestream"
)

// ResponsesError keeps the provider error fields for errors.As.
// The embedded ProviderError contains HTTP details and is not serialized.
type ResponsesError struct {
	Code                   string `json:"code"`
	Type                   string `json:"type"`
	*fantasy.ProviderError `json:"-"`
}

// Error returns the message and code, without request data.
func (e *ResponsesError) Error() string {
	if e.Code == "" {
		return e.ProviderError.Error()
	}
	return fmt.Sprintf("%s (code: %s)", e.ProviderError.Error(), e.Code)
}

// Unwrap lets callers inspect the provider and SDK errors.
func (e *ResponsesError) Unwrap() error { return e.ProviderError }

func responsesStreamFailureError(title, raw string, response *http.Response) *ResponsesError {
	var payload struct {
		Code       string          `json:"code"`
		Type       string          `json:"type"`
		Message    string          `json:"message"`
		StatusCode int             `json:"status_code"`
		Error      json.RawMessage `json:"error"`
	}
	_ = json.Unmarshal([]byte(raw), &payload)
	if len(payload.Error) > 0 {
		nested := payload.Error
		_ = json.Unmarshal(nested, &payload)
	}
	status := payload.StatusCode
	if response != nil {
		status = response.StatusCode
	}
	provider := &fantasy.ProviderError{Title: title, Message: payload.Message, StatusCode: status}
	parseContextTooLargeError(payload.Message, provider)
	return &ResponsesError{Code: payload.Code, Type: payload.Type, ProviderError: provider}
}

func toResponsesProviderErr(err error, response *http.Response) error {
	converted := toProviderErr(err)
	var provider *fantasy.ProviderError
	if !errors.As(converted, &provider) {
		return converted
	}
	var api *openai.Error
	if errors.As(err, &api) {
		return &ResponsesError{Code: api.Code, Type: api.Type, ProviderError: provider}
	}
	var stream *ssestream.StreamError
	if errors.As(err, &stream) {
		typed := responsesStreamFailureError("stream error", string(stream.Event.Data), response)
		provider.StatusCode = typed.StatusCode
		typed.ProviderError = provider
		return typed
	}
	return converted
}
