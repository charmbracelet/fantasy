package openai

import (
	"charm.land/fantasy"
	"charm.land/fantasy/providers/internal/httpheaders"
	"github.com/openai/openai-go/v2/option"
)

// callUARequestOptions returns per-request options that override the
// client-level User-Agent header when the Call carries agent-level UA settings.
func callUARequestOptions(call fantasy.Call) []option.RequestOption {
	if ua, ok := httpheaders.CallUserAgent(fantasy.Version, call.UserAgent, call.ModelSegment); ok {
		return []option.RequestOption{option.WithHeader("User-Agent", ua)}
	}
	return nil
}

// objectCallUARequestOptions returns per-request options that override the
// client-level User-Agent header when the ObjectCall carries agent-level UA settings.
func objectCallUARequestOptions(call fantasy.ObjectCall) []option.RequestOption {
	if ua, ok := httpheaders.CallUserAgent(fantasy.Version, call.UserAgent, call.ModelSegment); ok {
		return []option.RequestOption{option.WithHeader("User-Agent", ua)}
	}
	return nil
}
