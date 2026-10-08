package openai

import (
	"bytes"
	"cmp"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"

	"github.com/openai/openai-go/v3/option"
)

// maxErrorBodySize bounds how much of an error response body the
// normalization middleware buffers.
const maxErrorBodySize = 1 << 20 // 1 MiB

// normalizeErrorBodyMiddleware rewrites non-standard error response bodies
// (status >= 400) into the OpenAI error envelope before the SDK parses them.
// The openai-go SDK unmarshals the body's "error" member into a struct and
// returns the parse error instead of the API error when the member is not an
// object, which some OpenAI-compatible providers do (e.g.
// `{"error": "rate limit reached"}` or a plain text body). In that case the
// HTTP status code and the provider's actual message are lost. Normalizing
// the body up front lets the SDK surface a typed error, which toProviderErr
// then converts into a *fantasy.ProviderError carrying the real status and
// message.
func normalizeErrorBodyMiddleware(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
	res, err := next(req)
	if err != nil || res == nil || res.Body == nil || res.StatusCode < 400 {
		return res, err
	}

	body, readErr := io.ReadAll(io.LimitReader(res.Body, maxErrorBodySize))
	_ = res.Body.Close()
	if readErr != nil {
		// Propagate the read error so a truncated error body is classified
		// as a retryable transport failure instead of a JSON parse error.
		return res, readErr
	}

	body = normalizeErrorBody(body, res.StatusCode)
	res.ContentLength = int64(len(body))
	if res.Header == nil {
		// Custom transports may return a response without a header map.
		res.Header = make(http.Header)
	}
	res.Header.Set("Content-Length", strconv.Itoa(len(body)))
	res.Body = io.NopCloser(bytes.NewReader(body))
	return res, nil
}

// normalizeErrorBody returns body unchanged when its "error" member is an
// object the SDK can parse. Otherwise it synthesizes an error envelope
// carrying the provider's message.
func normalizeErrorBody(body []byte, statusCode int) []byte {
	trimmed := bytes.TrimSpace(body)

	var envelope struct {
		Error json.RawMessage `json:"error"`
	}
	if err := json.Unmarshal(trimmed, &envelope); err == nil && len(envelope.Error) > 0 && string(envelope.Error) != "null" {
		var errObj struct {
			Code    *string `json:"code"`
			Message *string `json:"message"`
			Param   *string `json:"param"`
			Type    *string `json:"type"`
		}
		if err := json.Unmarshal(envelope.Error, &errObj); err == nil {
			return body
		}
	}

	message := extractErrorMessage(trimmed)
	if message == "" {
		message = cmp.Or(http.StatusText(statusCode), fmt.Sprintf("HTTP status %d", statusCode))
	}

	rewritten, err := json.Marshal(map[string]any{
		"error": map[string]string{"message": message},
	})
	if err != nil {
		return body
	}
	return rewritten
}

// extractErrorMessage pulls the most informative message out of a
// non-standard error body.
func extractErrorMessage(trimmed []byte) string {
	var envelope struct {
		Error   json.RawMessage `json:"error"`
		Message *string         `json:"message"`
	}
	_ = json.Unmarshal(trimmed, &envelope)

	var errString string
	if err := json.Unmarshal(envelope.Error, &errString); err == nil && errString != "" {
		return errString
	}

	var errObj struct {
		Message *string `json:"message"`
	}
	if err := json.Unmarshal(envelope.Error, &errObj); err == nil && errObj.Message != nil && *errObj.Message != "" {
		return *errObj.Message
	}

	if envelope.Message != nil && *envelope.Message != "" {
		return *envelope.Message
	}

	var bodyString string
	if err := json.Unmarshal(trimmed, &bodyString); err == nil {
		return bodyString
	}

	return string(trimmed)
}
