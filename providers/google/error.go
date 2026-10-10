package google

import (
	"cmp"
	"errors"
	"regexp"
	"strconv"
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
		ResponseHeaders: retryHeadersFromDetails(apiErr.Details),
		ResponseBody:    []byte(apiErr.Message),
		ErrorType:       apiErr.Status,
	}

	parseContextTooLargeError(apiErr.Message, providerErr)

	return providerErr
}

// retryInfoType is the type URL of the google.rpc.RetryInfo error detail.
const retryInfoType = "type.googleapis.com/google.rpc.RetryInfo"

// retryHeadersFromDetails turns a RetryInfo detail into a retry-after header.
// Gemini answers a 429 without a Retry-After header and puts the wait it wants
// in the error's details instead, so the retry loop, which only reads headers,
// would otherwise fall back to its own backoff and retry too soon.
func retryHeadersFromDetails(details []map[string]any) map[string]string {
	for _, detail := range details {
		if detail["@type"] != retryInfoType {
			continue
		}
		raw, _ := detail["retryDelay"].(string)
		if delay, err := time.ParseDuration(raw); err == nil && delay > 0 {
			return map[string]string{
				"retry-after": strconv.FormatFloat(delay.Seconds(), 'f', -1, 64),
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
