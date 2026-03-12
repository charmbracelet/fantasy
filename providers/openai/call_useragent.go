package openai

import (
	"charm.land/fantasy"
	"charm.land/fantasy/providers/internal/httpheaders"
	"github.com/openai/openai-go/v2/option"
)

// callUARequestOptions returns per-request options that override the
// client-level User-Agent header when the Call carries agent-level UA
// settings.
//
// When noDefaultUA is true the SDK's own User-Agent is preserved and no
// override is applied (needed for providers like OpenRouter, which reject
// User-Agent headers they don't expect).
func callUARequestOptions(call fantasy.Call, noDefaultUA bool) []option.RequestOption {
	if noDefaultUA {
		return nil
	}
	if ua, ok := httpheaders.CallUserAgent(call.UserAgent); ok {
		return []option.RequestOption{option.WithHeader("User-Agent", ua)}
	}
	return nil
}

// objectCallUARequestOptions returns per-request options that override the
// client-level User-Agent header when the ObjectCall carries agent-level UA
// settings.
func objectCallUARequestOptions(call fantasy.ObjectCall, noDefaultUA bool) []option.RequestOption {
	if noDefaultUA {
		return nil
	}
	if ua, ok := httpheaders.CallUserAgent(call.UserAgent); ok {
		return []option.RequestOption{option.WithHeader("User-Agent", ua)}
	}
	return nil
}
