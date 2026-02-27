package anthropic

import (
	"charm.land/fantasy"
	"charm.land/fantasy/providers/internal/httpheaders"
	"github.com/charmbracelet/anthropic-sdk-go/option"
)

func callUARequestOptions(call fantasy.Call) []option.RequestOption {
	if ua, ok := httpheaders.CallUserAgent(call.UserAgent); ok {
		return []option.RequestOption{option.WithHeader("User-Agent", ua)}
	}
	return nil
}
