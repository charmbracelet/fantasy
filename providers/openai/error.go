package openai

import (
	"cmp"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"strings"

	"charm.land/fantasy"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/ssestream"
)

var (
	openaiContextPattern  = regexp.MustCompile(`maximum context length (?:is|of) (\d+) tokens.*?(?:resulted in|requested) ~?(\d+) tokens`)
	alibabaContextPattern = regexp.MustCompile(`Range of input length should be \[\d+,\s*(\d+)\]`)
	vercelContextPattern  = regexp.MustCompile(`Input too long:\s*(\d+)\s*input tokens,\s*limit is\s*(\d+)`)
)

func toProviderErr(err error) error {
	var apiErr *openai.Error
	if errors.As(err, &apiErr) {
		message := toProviderErrMessage(apiErr)
		providerErr := &fantasy.ProviderError{
			Title:           cmp.Or(fantasy.ErrorTitleForStatusCode(apiErr.StatusCode), "provider request failed"),
			Message:         message,
			Cause:           apiErr,
			URL:             apiErr.Request.URL.String(),
			StatusCode:      apiErr.StatusCode,
			RequestBody:     apiErr.DumpRequest(true),
			ResponseHeaders: toHeaderMap(apiErr.Response.Header),
			ResponseBody:    apiErr.DumpResponse(true),
		}

		parseContextTooLargeError(message, providerErr)

		return providerErr
	}

	// Mid-stream SSE error events surface as *ssestream.StreamError, not
	// *openai.Error, so they need their own classification path.
	var streamErr *ssestream.StreamError
	if errors.As(err, &streamErr) {
		return toProviderErrFromStreamError(streamErr)
	}

	// Wrap transient transport failures so `.IsRetryable()` works.
	return fantasy.WrapTransportError(err)
}

// streamErrorEnvelope mirrors the OpenAI-standard error envelope
// (`{"error": {"code", "message", "type"}}`) that arrives as an in-band
// SSE event on stream failure.
type streamErrorEnvelope struct {
	Error struct {
		Code    string `json:"code"`
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

func toProviderErrFromStreamError(streamErr *ssestream.StreamError) *fantasy.ProviderError {
	var envelope streamErrorEnvelope
	_ = json.Unmarshal(streamErr.Event.Data, &envelope) // best-effort; falls back to the raw message on parse failure.

	errType := cmp.Or(envelope.Error.Type, envelope.Error.Code)

	return &fantasy.ProviderError{
		Title:          "stream error",
		Message:        cmp.Or(envelope.Error.Message, streamErr.Message),
		Cause:          streamErr,
		ResponseBody:   streamErr.Event.Data,
		TransientError: fantasy.TransientStreamErrorTypes[errType],
	}
}

func parseContextTooLargeError(message string, providerErr *fantasy.ProviderError) {
	if matches := openaiContextPattern.FindStringSubmatch(message); matches != nil {
		providerErr.ContextTooLargeErr = true
		providerErr.ContextMaxTokens, _ = strconv.Atoi(matches[1])
		providerErr.ContextUsedTokens, _ = strconv.Atoi(matches[2])
		return
	}
	if matches := alibabaContextPattern.FindStringSubmatch(message); matches != nil {
		providerErr.ContextTooLargeErr = true
		providerErr.ContextMaxTokens, _ = strconv.Atoi(matches[1])
		return
	}
	if matches := vercelContextPattern.FindStringSubmatch(message); matches != nil {
		providerErr.ContextTooLargeErr = true
		providerErr.ContextUsedTokens, _ = strconv.Atoi(matches[1])
		providerErr.ContextMaxTokens, _ = strconv.Atoi(matches[2])
	}
}

func toProviderErrMessage(apiErr *openai.Error) string {
	if apiErr.Message != "" {
		return apiErr.Message
	}

	// For some OpenAI-compatible providers, the SDK is not always able to parse
	// the error message correctly.
	// Fallback to returning the raw response body in such cases.
	data, _ := io.ReadAll(apiErr.Response.Body)
	return string(data)
}

func toHeaderMap(in http.Header) (out map[string]string) {
	out = make(map[string]string, len(in))
	for k, v := range in {
		if l := len(v); l > 0 {
			out[k] = v[l-1]
			in[strings.ToLower(k)] = v
		}
	}
	return out
}
