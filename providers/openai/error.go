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

	// The SDK only wraps errors on the initial HTTP response as *openai.Error
	// ("Other errors are not wrapped by this SDK" -- its own doc comment). An
	// in-band SSE error event that arrives mid-stream, after the response
	// already returned 200 OK, surfaces as a *ssestream.StreamError instead
	// and would otherwise never reach ProviderError.IsRetryable(), so a
	// transient upstream failure (e.g. "type":"server_error") silently skips
	// retry entirely instead of just being classified non-retryable.
	var streamErr *ssestream.StreamError
	if errors.As(err, &streamErr) {
		return toProviderErrFromStreamError(streamErr)
	}

	// Wrap transient transport failures so `.IsRetryable()` works.
	return fantasy.WrapTransportError(err)
}

// streamErrorEnvelope mirrors the OpenAI-standard error envelope
// (`{"error": {"code", "message", "param", "type"}}`) that arrives as an
// in-band SSE event on stream failure.
type streamErrorEnvelope struct {
	Error struct {
		Code    string `json:"code"`
		Message string `json:"message"`
		Param   string `json:"param"`
		Type    string `json:"type"`
	} `json:"error"`
}

// retryableStreamErrorTypes are the "type"/"code" values documented as
// transient, provider-side failures -- the in-stream equivalent of an HTTP
// 5xx on the initial request.
var retryableStreamErrorTypes = map[string]bool{
	"server_error": true,
}

func toProviderErrFromStreamError(streamErr *ssestream.StreamError) *fantasy.ProviderError {
	var envelope streamErrorEnvelope
	_ = json.Unmarshal(streamErr.Event.Data, &envelope) // best-effort; falls back to the raw message on parse failure.

	providerErr := &fantasy.ProviderError{
		Title:   "stream error",
		Message: cmp.Or(envelope.Error.Message, streamErr.Message),
		Cause:   streamErr,
	}

	if retryableStreamErrorTypes[envelope.Error.Type] || retryableStreamErrorTypes[envelope.Error.Code] {
		// Synthesize a 5xx so the existing IsRetryable() status-code check
		// covers this without needing its own special case.
		providerErr.StatusCode = http.StatusInternalServerError
	}

	return providerErr
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
