package google

import (
	"cmp"
	"errors"
	"regexp"
	"strconv"
	"strings"
	"time"

	"charm.land/fantasy"
	"google.golang.org/genai"
)

var googleContextPattern = regexp.MustCompile(`input token count.*?(\d+).*?exceeds.*?maximum.*?(\d+)`)

func toProviderErr(err error) error {
	var apiErr genai.APIError
	if !errors.As(err, &apiErr) {
		// Wrap transient transport failures so `.IsRetryable()` works.
		return fantasy.WrapTransportError(err)
	}

	providerErr := &fantasy.ProviderError{
		Message:         apiErr.Message,
		Title:           cmp.Or(fantasy.ErrorTitleForStatusCode(apiErr.Code), "provider request failed"),
		Cause:           err,
		StatusCode:      apiErr.Code,
		ResponseBody:    []byte(apiErr.Message),
		ResponseHeaders: retryHeadersFromDetails(apiErr.Details),
	}

	parseContextTooLargeError(apiErr.Message, providerErr)

	return providerErr
}

// retryHeadersFromDetails looks for a google.rpc.RetryInfo entry in a
// genai.APIError's Details — the structured hint Gemini actually uses to
// report how long to wait on RESOURCE_EXHAUSTED (HTTP 429) — and, if found,
// synthesizes a lowercase "retry-after" header from its retryDelay so
// retry.go's getRetryDelayInMs picks it up the same way it does for the
// other providers' real Retry-After headers.
func retryHeadersFromDetails(details []map[string]any) map[string]string {
	for _, detail := range details {
		typ, _ := detail["@type"].(string)
		if !strings.Contains(typ, "RetryInfo") {
			continue
		}
		raw, _ := detail["retryDelay"].(string)
		if raw == "" {
			continue
		}
		if d, err := time.ParseDuration(raw); err == nil && d > 0 {
			return map[string]string{
				"retry-after": strconv.FormatFloat(d.Seconds(), 'f', -1, 64),
			}
		}
	}
	return nil
}

func parseContextTooLargeError(message string, providerErr *fantasy.ProviderError) {
	matches := googleContextPattern.FindStringSubmatch(message)
	if matches == nil {
		return
	}
	providerErr.ContextTooLargeErr = true
	providerErr.ContextUsedTokens, _ = strconv.Atoi(matches[1])
	providerErr.ContextMaxTokens, _ = strconv.Atoi(matches[2])
}
