package google

import (
	"cmp"
	"errors"
	"io"
	"regexp"
	"strconv"

	"charm.land/fantasy"
	"google.golang.org/genai"
)

var googleContextPattern = regexp.MustCompile(`input token count.*?(\d+).*?exceeds.*?maximum.*?(\d+)`)

func toProviderErr(err error) error {
	var apiErr genai.APIError
	if !errors.As(err, &apiErr) {
		// Transient transport failures from the streaming decoder (most
		// commonly io.ErrUnexpectedEOF from a mid-stream SSE disconnect)
		// arrive here as plain errors that are not genai.APIError. Wrap
		// them as ProviderError so that the retry loop in retry.go
		// engages — ProviderError.IsRetryable already classifies
		// io.ErrUnexpectedEOF as retryable, but that check is only
		// reached if the error is a *ProviderError to begin with.
		if errors.Is(err, io.ErrUnexpectedEOF) {
			return &fantasy.ProviderError{
				Title:   "stream transport error",
				Message: err.Error(),
				Cause:   err,
			}
		}
		return err
	}

	providerErr := &fantasy.ProviderError{
		Message:      apiErr.Message,
		Title:        cmp.Or(fantasy.ErrorTitleForStatusCode(apiErr.Code), "provider request failed"),
		Cause:        err,
		StatusCode:   apiErr.Code,
		ResponseBody: []byte(apiErr.Message),
	}

	parseContextTooLargeError(apiErr.Message, providerErr)

	return providerErr
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
