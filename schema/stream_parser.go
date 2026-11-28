package schema

import (
	"encoding/json"
	"errors"
	"strings"

	"github.com/gatewaytofredom/pjson"
)

var errIncompleteJSON = errors.New("incomplete json")

// StreamParser incrementally parses streamed JSON chunks using pjson in O(N)
// total time, falling back to ParsePartialJSON (jsonrepair) if the stream
// contains malformed non-standard JSON syntax (e.g. unquoted keys or single
// quotes).
type StreamParser struct {
	parser   *pjson.Parser
	raw      strings.Builder
	buf      []byte
	fallback bool
}

// NewStreamParser creates a new incremental partial JSON parser for streaming
// structured outputs.
func NewStreamParser() *StreamParser {
	return &StreamParser{
		parser: pjson.New(
			pjson.WithMarkdownFence(),
			pjson.WithCompleteLiterals(),
			pjson.WithTrailingCommas(),
		),
	}
}

// Raw returns the raw text accumulated so far.
func (s *StreamParser) Raw() string {
	return s.raw.String()
}

// Feed appends delta to the stream and attempts to parse the current partial
// JSON state.
func (s *StreamParser) Feed(delta string) (any, ParseState, error) {
	if delta != "" {
		s.raw.WriteString(delta)
	}
	if s.raw.Len() == 0 {
		return nil, ParseStateUndefined, nil
	}

	if !s.fallback {
		if _, err := s.parser.WriteString(delta); err == nil {
			s.buf = s.parser.CompleteJSONAppend(s.buf[:0])
			if len(s.buf) == 0 {
				return nil, ParseStateFailed, errIncompleteJSON
			}
			var result any
			if err := json.Unmarshal(s.buf, &result); err == nil {
				if s.parser.Done() && len(s.buf) == len(strings.TrimSpace(s.raw.String())) {
					return result, ParseStateSuccessful, nil
				}
				return result, ParseStateRepaired, nil
			}
			// Valid pjson snapshot failed json.Unmarshal (should not happen);
			// fall back to full repair.
		}
		s.fallback = true
	}

	return ParsePartialJSON(s.raw.String())
}
