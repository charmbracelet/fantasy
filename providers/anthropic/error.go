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
	"github.com/anthropics/anthropic-sdk-go"
)

var anthropicContextPattern = regexp.MustCompile(`prompt is too long:\s*(\d+)\s*tokens?\s*>\s*(\d+)\s*maximum`)

// awsCredentialErrorFragment identifies an expired AWS credential-chain
// failure. Bedrock runs through this provider, so when its SSO/role
// credentials need refreshing the AWS SDK surfaces this message locally
// rather than as an HTTP 401. Direct Anthropic API calls never produce it.
const awsCredentialErrorFragment = "failed to refresh cached credentials" //nolint:gosec // false positive: error message fragment, not a credential

func toProviderErr(err error) error {
	if apiErr, ok := errors.AsType[*anthropic.Error](err); ok {
		var envelope errorEnvelope
		_ = json.Unmarshal([]byte(apiErr.RawJSON()), &envelope) // A foreign body keeps the SDK's message.
		providerErr := &fantasy.ProviderError{
			Title:           apiErrorTitle(apiErr),
			Message:         cmp.Or(envelope.Error.Message, apiErr.Error()),
			Cause:           apiErr,
			URL:             apiErr.Request.URL.String(),
			StatusCode:      apiErr.StatusCode,
			RequestBody:     apiErr.DumpRequest(true),
			ResponseHeaders: toHeaderMap(apiErr.Response.Header),
			ResponseBody:    apiErr.DumpResponse(true),
			ErrorType:       string(apiErr.Type()),
			RequestID:       cmp.Or(apiErr.RequestID, envelope.RequestID),
			TransientError:  fantasy.TransientStreamErrorTypes[string(apiErr.Type())],
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

// apiErrorTitle names what the error describes rather than the exchange that
// carried it, preferring the status, then the payload's type, then a generic
// title. A mid-stream SSE event rides inside a response that already went out
// as a 2xx, so titling from that status puts "ok" above a failure, and only
// the payload can name that one.
func apiErrorTitle(apiErr *anthropic.Error) string {
	errType := string(apiErr.Type())
	if apiErr.StatusCode >= http.StatusOK && apiErr.StatusCode < http.StatusMultipleChoices {
		return streamErrorTitle(errType)
	}
	// Anthropic answers an overload with 529, which the HTTP registry does
	// not name, so a status-derived title comes out empty and the payload's
	// type is the only thing left that says what happened.
	statusTitle := fantasy.ErrorTitleForStatusCode(apiErr.StatusCode)
	if statusTitle == "" && errType != "" {
		return streamErrorTitle(errType)
	}
	return cmp.Or(statusTitle, "provider request failed")
}

// streamErrorPrefix is the message prefix the Anthropic SDK uses for a
// mid-stream `error` event. The SDK reports it via fmt.Errorf with no
// exported type, so we match the prefix and parse the payload ourselves.
const streamErrorPrefix = "received error while streaming:"

// errorEnvelope is the body of an Anthropic error, whether an HTTP response
// or an SSE `error` event. The outer "type" is always the literal "error".
type errorEnvelope struct {
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
	RequestID string `json:"request_id"`
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

	var envelope errorEnvelope
	_ = json.Unmarshal([]byte(payload), &envelope) // A payload that does not parse names nothing.
	if envelope.Error.Type == "" && envelope.Error.Message == "" {
		// Nothing recognizable: let it fall through to transport handling.
		return nil
	}

	errType := envelope.Error.Type
	return &fantasy.ProviderError{
		Title:          streamErrorTitle(errType),
		Message:        cmp.Or(envelope.Error.Message, payload),
		Cause:          err,
		ResponseBody:   []byte(payload),
		ErrorType:      errType,
		RequestID:      envelope.RequestID,
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
	out = make(map[string]string, len(in)*2)
	for k, v := range in {
		if l := len(v); l > 0 {
			out[k] = v[l-1]
			out[strings.ToLower(k)] = v[l-1]
		}
	}
	return out
}
