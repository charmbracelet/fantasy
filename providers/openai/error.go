package openai

import (
	"cmp"
	"errors"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"strings"

	"charm.land/fantasy"
	"github.com/openai/openai-go/v2"
)

var openaiContextPattern = regexp.MustCompile(`maximum context length is (\d+) tokens.*?(?:resulted in|requested) (\d+) tokens`)

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
	return err
}

func parseContextTooLargeError(message string, providerErr *fantasy.ProviderError) {
	matches := openaiContextPattern.FindStringSubmatch(message)
	if matches == nil {
		return
	}
	providerErr.ContextTooLargeErr = true
	providerErr.ContextMaxTokens, _ = strconv.Atoi(matches[1])
	providerErr.ContextUsedTokens, _ = strconv.Atoi(matches[2])
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
