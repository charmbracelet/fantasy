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

// awsCredentialErrorFragment identifies an expired AWS credential-chain
// failure. Bedrock runs through this provider, so when its SSO/role
// credentials need refreshing the AWS SDK surfaces this message locally
// rather than as an HTTP 401. Direct Anthropic API calls never produce it.
const awsCredentialErrorFragment = "failed to refresh cached credentials"

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
	// Expired Bedrock (AWS) credentials surface from the local credential
	// chain, not as a 401. Flag them so OnAuthRefresh can engage.
	if strings.Contains(err.Error(), awsCredentialErrorFragment) {
		return &fantasy.ProviderError{
			Title:     "authentication error",
			Message:   err.Error(),
			Cause:     err,
			AuthError: true,
		}
	}
	// Wrap transient transport failures so `.IsRetryable()` works.
	return fantasy.WrapTransportError(err)
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
