// Package prompttest provides a shared corpus of prompts and a golden-file
// harness for the chat-completions prompt builders.
//
// Four providers (openai, openaicompat, openrouter, vercel) convert a
// fantasy.Prompt into the same OpenAI SDK message type, and each has its own
// copy of the conversion. One corpus runs through all four and records every
// converter's output for a case in a single file, testdata/<case>.json, keyed
// by provider. The file is the behaviour matrix: a difference between
// providers is a difference between adjacent keys, visible without running a
// diff across four directories.
//
// Reviewing a change to any converter means reading the golden diff rather
// than the control flow, and a behaviour that only one provider has becomes
// visible instead of buried. Regenerate with:
//
//	go test ./providers/internal/prompttest -update
package prompttest

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openai"
)

var update = flag.Bool("update", false, "rewrite prompt golden files")

// Case is one prompt fixture. Model is passed through to the converter because
// several providers branch on the model ID.
type Case struct {
	Name   string
	Model  string
	Prompt fantasy.Prompt
}

// Result is one converter's output for one case. Messages is stored as raw
// JSON so the harness records exactly what the SDK would put on the wire,
// including the extra fields providers attach for cache control and reasoning
// details.
type Result struct {
	Messages json.RawMessage `json:"messages"`
	Warnings []string        `json:"warnings"`
}

// Converter is one named prompt builder under test.
type Converter struct {
	Name    string
	Convert openai.LanguageModelToPromptFunc
}

// Golden runs every case through every converter and compares the recorded
// matrix against testdata, or rewrites it under -update.
//
// Warnings are recorded alongside the messages because dropping content
// silently and dropping it with a warning are very different behaviours, and
// only the record makes that difference visible.
func Golden(t *testing.T, converters []Converter) {
	t.Helper()

	cases := Cases()
	seen := make(map[string]bool, len(cases))
	for _, tc := range cases {
		if seen[tc.Name] {
			t.Fatalf("duplicate case name %q: both would write the same golden file", tc.Name)
		}
		seen[tc.Name] = true
	}

	for _, tc := range cases {
		t.Run(tc.Name, func(t *testing.T) {
			t.Parallel()
			got := record(t, tc, converters)
			path := filepath.Join(goldenDir, tc.Name+".json")
			if *update {
				if err := os.MkdirAll(goldenDir, 0o750); err != nil {
					t.Fatalf("create golden dir: %v", err)
				}
				if err := os.WriteFile(path, got, 0o600); err != nil {
					t.Fatalf("write golden: %v", err)
				}
				return
			}
			want, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("read golden (run with -update to create): %v", err)
			}
			// record writes LF, so a CRLF checkout would mismatch every line
			// while the diff rendered identically.
			want = []byte(strings.ReplaceAll(string(want), "\r\n", "\n"))
			if diff := lineDiff(string(want), string(got)); diff != "" {
				t.Errorf("golden mismatch for %s:\n%s", tc.Name, diff)
			}
		})
	}

	t.Run("no_stale_goldens", func(t *testing.T) {
		if *update {
			t.Skip("rewriting goldens")
		}
		entries, err := os.ReadDir(goldenDir)
		if err != nil {
			t.Fatalf("read golden dir: %v", err)
		}
		for _, e := range entries {
			if e.IsDir() || filepath.Ext(e.Name()) != ".json" {
				continue
			}
			name := strings.TrimSuffix(e.Name(), ".json")
			if !seen[name] {
				t.Errorf("golden %s has no matching case; delete it or restore the case", e.Name())
			}
		}
	})
}

const goldenDir = "testdata"

// record converts one case with every converter and marshals the matrix.
//
// Converters producing identical output share one entry, keyed by their
// comma-joined names, so agreement collapses to a single block and any file
// with more than one key names exactly who diverges.
func record(t *testing.T, tc Case, converters []Converter) []byte {
	t.Helper()

	type group struct {
		result Result
		names  []string
	}
	groups := map[string]*group{}
	for _, c := range converters {
		messages, warnings := c.Convert(tc.Prompt, c.Name, tc.Model)
		raw, err := json.Marshal(messages)
		if err != nil {
			t.Fatalf("marshal %s messages: %v", c.Name, err)
		}
		texts := make([]string, 0, len(warnings))
		for _, w := range warnings {
			texts = append(texts, string(w.Type)+": "+w.Message)
		}
		result := Result{Messages: raw, Warnings: texts}
		key, err := json.Marshal(result)
		if err != nil {
			t.Fatalf("key %s result: %v", c.Name, err)
		}
		g, ok := groups[string(key)]
		if !ok {
			g = &group{result: result}
			groups[string(key)] = g
		}
		g.names = append(g.names, c.Name)
	}

	matrix := make(map[string]Result, len(groups))
	for _, g := range groups {
		matrix[strings.Join(g.names, ",")] = g.result
	}
	got, err := json.MarshalIndent(matrix, "", "  ")
	if err != nil {
		t.Fatalf("marshal matrix: %v", err)
	}
	return append(got, '\n')
}

// lineDiff reports the differing lines between want and got, or "" when they
// match. Golden files here run to hundreds of base64-heavy lines, so printing
// both in full buries the one field that moved.
func lineDiff(want, got string) string {
	if want == got {
		return ""
	}
	wantLines, gotLines := strings.Split(want, "\n"), strings.Split(got, "\n")
	var b strings.Builder
	for i := 0; i < max(len(wantLines), len(gotLines)); i++ {
		w, g := at(wantLines, i), at(gotLines, i)
		if w == g {
			continue
		}
		fmt.Fprintf(&b, "  line %d:\n    want: %s\n    got:  %s\n", i+1, w, g)
	}
	return b.String()
}

func at(lines []string, i int) string {
	if i < len(lines) {
		return lines[i]
	}
	return "<missing>"
}
