// Package httpheaders provides shared User-Agent resolution for all HTTP-based providers.
package httpheaders

import "strings"

// DefaultUserAgent returns the default User-Agent string for the SDK.
// The result is "Charm Fantasy/<version>".
func DefaultUserAgent(version string) string {
	return "Charm Fantasy/" + version
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

// CallUserAgent resolves the User-Agent for a single API call. It returns the
// resolved UA string and true if a per-call override should be applied, or
// empty string and false if the client-level UA should be used as-is.
func CallUserAgent(callUA string) (string, bool) {
	if callUA != "" {
		return callUA, true
	}
	return "", false
}
