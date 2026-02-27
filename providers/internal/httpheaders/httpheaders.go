// Package httpheaders provides shared User-Agent resolution for all HTTP-based providers.
package httpheaders

import (
	"strings"
	"unicode"
)

const maxAgentLength = 64

// DefaultUserAgent returns the default User-Agent string for the SDK.
// If agent is non-empty, the result is "Charm Fantasy/<version> (<agent>)".
// Otherwise, the result is "Charm Fantasy/<version>".
func DefaultUserAgent(version, agent string) string {
	const sdk = "Charm Fantasy/"
	agent = sanitizeAgent(agent)
	if agent == "" {
		return sdk + version
	}
	return sdk + version + " (" + agent + ")"
}

// ResolveHeaders returns a new header map with User-Agent resolved according to precedence:
//  1. explicitUA (highest — set via WithUserAgent)
//  2. existing User-Agent key in headers (case-insensitive — set via WithHeaders)
//  3. defaultUA (lowest — generated default)
//
// The input map is never mutated.
func ResolveHeaders(headers map[string]string, explicitUA, defaultUA string) map[string]string {
	out := make(map[string]string, len(headers)+1)
	var uaKeys []string

	for k, v := range headers {
		out[k] = v
		if strings.EqualFold(k, "User-Agent") {
			uaKeys = append(uaKeys, k)
		}
	}

	switch {
	case explicitUA != "":
		for _, k := range uaKeys {
			delete(out, k)
		}
		out["User-Agent"] = explicitUA
	case len(uaKeys) > 0:
		// keep the header-map value as-is
	default:
		out["User-Agent"] = defaultUA
	}

	return out
}

func sanitizeAgent(s string) string {
	s = strings.TrimSpace(s)
	var b strings.Builder
	b.Grow(len(s))
	count := 0
	for _, r := range s {
		if r < 0x20 || r == '(' || r == ')' {
			continue
		}
		if count >= maxAgentLength {
			break
		}
		b.WriteRune(r)
		count++
	}
	return strings.TrimRightFunc(b.String(), unicode.IsSpace)
}
