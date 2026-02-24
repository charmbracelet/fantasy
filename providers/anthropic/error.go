package anthropic

import (
	"cmp"
	"errors"
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
