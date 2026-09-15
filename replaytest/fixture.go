// Package replaytest provides a fixture replay harness for provider tests:
// it plays a recorded upstream response through a real fantasy provider and
// compares the resulting StreamPart sequence against golden files.
//
// Fixtures live under providertests/testdata/shapes/<shape>/<case>/ and are
// shape-named: the directory names describe what the upstream chunks look
// like, never which provider produced them. See the package README for the
// full fixture layout, the golden format, and the -update flag.
package replaytest

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Fixture is a recorded upstream response together with the request that
// produced it.
type Fixture struct {
	// Dir is the directory the fixture was loaded from.
	Dir string
	// Request is the content of request.json: the upstream request the
	// provider sends for this fixture, for documentation and future
	// request-side assertions.
	Request json.RawMessage
	// Events holds the raw SSE events of response.sse, split on
	// blank-line boundaries. Each event is kept verbatim, including its
	// "data:" prefixes. Empty when the fixture is non-streaming.
	Events []string
	// Body holds the raw non-streaming response of response.json. Empty
	// when the fixture is streaming.
	Body []byte

	// server is the replay server bound to this fixture by Serve, so
	// RunAgentStep can record the requests the model under test sends.
	server *Server
}

// Load reads a fixture directory. It requires request.json and exactly one
// of response.sse (streaming) or response.json (non-streaming).
func Load(dir string) (*Fixture, error) {
	request, err := os.ReadFile(filepath.Join(dir, "request.json"))
	if err != nil {
		return nil, fmt.Errorf("replaytest: read request.json: %w", err)
	}

	sse, sseErr := os.ReadFile(filepath.Join(dir, "response.sse"))
	if sseErr == nil {
		if _, bodyErr := os.Stat(filepath.Join(dir, "response.json")); bodyErr == nil {
			return nil, fmt.Errorf("replaytest: %s has both response.sse and response.json", dir)
		}
		return &Fixture{
			Dir:     dir,
			Request: request,
			Events:  SplitSSE(string(sse)),
		}, nil
	}
	if !os.IsNotExist(sseErr) {
		return nil, fmt.Errorf("replaytest: read response.sse: %w", sseErr)
	}

	body, err := os.ReadFile(filepath.Join(dir, "response.json"))
	if err != nil {
		return nil, fmt.Errorf("replaytest: %s needs response.sse or response.json: %w", dir, err)
	}
	return &Fixture{
		Dir:     dir,
		Request: request,
		Body:    body,
	}, nil
}

// SplitSSE splits raw SSE text into events on blank-line boundaries. Each
// event is kept verbatim, including its field prefixes. CRLF line endings
// are normalized to LF first, so fixtures checked out by git with CRLF on
// Windows still split into the same events.
func SplitSSE(s string) []string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	blocks := strings.Split(s, "\n\n")
	events := make([]string, 0, len(blocks))
	for _, block := range blocks {
		block = strings.TrimSuffix(block, "\n")
		if strings.TrimSpace(block) == "" {
			continue
		}
		events = append(events, block)
	}
	return events
}
