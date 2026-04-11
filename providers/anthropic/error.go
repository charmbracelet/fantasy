package anthropic

import (
	"cmp"
	"errors"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"strings"

	"charm.land/fantasy"
	"github.com/charmbracelet/anthropic-sdk-go"
)

var anthropicContextPattern = regexp.MustCompile(`prompt is too long:\s*(\d+)\s*tokens?\s*>\s*(\d+)\s*maximum`)

func toProviderErr(err error) error {
	var apiErr *anthropic.Error
	if errors.As(err, &apiErr) {
		providerErr := &fantasy.ProviderError{
			Title:           cmp.Or(fantasy.ErrorTitleForStatusCode(apiErr.StatusCode), "provider request failed"),
			Message:         apiErr.Error(),
			Cause:           apiErr,
			URL:             apiErr.Request.URL.String(),
			StatusCode:      apiErr.StatusCode,
			RequestBody:     apiErr.DumpRequest(true),
			ResponseHeaders: toHeaderMap(apiErr.Response.Header),
			ResponseBody:    apiErr.DumpResponse(true),
		}

		parseContextTooLargeError(apiErr.Error(), providerErr)

		return providerErr
	}
	// Transient transport failures from the streaming decoder (most commonly
	// io.ErrUnexpectedEOF from a mid-stream SSE disconnect) arrive here as
	// plain errors that are not *anthropic.Error. Wrap them as ProviderError
	// so that the retry loop in retry.go engages — ProviderError.IsRetryable
	// already classifies io.ErrUnexpectedEOF as retryable, but that check is
	// only reached if the error is a *ProviderError to begin with.
	if errors.Is(err, io.ErrUnexpectedEOF) {
		return &fantasy.ProviderError{
			Title:   "stream transport error",
			Message: err.Error(),
			Cause:   err,
		}
	}
	return err
}

func parseContextTooLargeError(message string, providerErr *fantasy.ProviderError) {
	matches := anthropicContextPattern.FindStringSubmatch(message)
	if matches == nil {
		return
	}

	providerErr.ContextTooLargeErr = true
	providerErr.ContextUsedTokens, _ = strconv.Atoi(matches[1])
	providerErr.ContextMaxTokens, _ = strconv.Atoi(matches[2])
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
