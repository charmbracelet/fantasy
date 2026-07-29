package anthropic

import (
	"cmp"
	"encoding/json"
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
const awsCredentialErrorFragment = "failed to refresh cached credentials" //nolint:gosec // false positive: error message fragment, not a credential

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
	// Wrap transient failures so `.IsRetryable()` works: mid-stream error
	// events first, then transport-level failures.
	if wrapped := wrapStreamError(err); wrapped != nil {
		return wrapped
	}
	return fantasy.WrapTransportError(err)
}

// streamErrorPrefix is the message prefix the Anthropic SDK uses for a
// mid-stream `error` event. The SDK reports it via fmt.Errorf with no
// exported type, so we match the prefix and parse the payload ourselves.
const streamErrorPrefix = "received error while streaming:"

// streamErrorEnvelope is the payload of an SSE `error` event. Anthropic
// nests the useful detail under "error"; the outer "type" is always the
// literal "error" and carries no classification value.
type streamErrorEnvelope struct {
	Error *struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// wrapStreamError classifies a mid-stream SSE error event as a
// ProviderError, marking it transient when the payload names a temporary
// server-side condition. Returns nil if err is not a stream error event.
func wrapStreamError(err error) *fantasy.ProviderError {
	_, payload, ok := strings.Cut(err.Error(), streamErrorPrefix)
	if !ok {
		return nil
	}
	payload = strings.TrimSpace(payload)

	var envelope streamErrorEnvelope
	errType, message := "", ""
	if json.Unmarshal([]byte(payload), &envelope) == nil && envelope.Error != nil {
		errType = envelope.Error.Type
		message = envelope.Error.Message
	}
	if errType == "" && message == "" {
		// Nothing recognizable: let it fall through to transport handling.
		return nil
	}

	return &fantasy.ProviderError{
		Title:          streamErrorTitle(errType),
		Message:        cmp.Or(message, payload),
		Cause:          err,
		ResponseBody:   []byte(payload),
		TransientError: fantasy.TransientStreamErrorTypes[errType],
	}
}

func streamErrorTitle(errType string) string {
	switch errType {
	case "overloaded_error":
		return "provider overloaded"
	case "rate_limit_error":
		return "rate limit exceeded"
	case "":
		return "provider stream error"
	default:
		return strings.ReplaceAll(errType, "_", " ")
	}
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
