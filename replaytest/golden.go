package replaytest

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

var update = flag.Bool("update", false, "rewrite golden files with the observed output")

// AssertGolden marshals records with two-space indentation and stable key
// order, then compares the bytes with the golden file at path. Under
// -update the file is written and the assertion passes. On mismatch the
// test fails with a unified diff.
//
// Both sides are normalized to LF line endings before comparing, so goldens
// stay stable when git checks the files out with CRLF on Windows.
//
// The stable output is achieved by marshaling struct fields in declaration
// order and map keys sorted; maps must not be relied on for ordering
// anywhere records are built.
func AssertGolden(t testing.TB, path string, records any) {
	t.Helper()
	data, err := json.MarshalIndent(records, "", "  ")
	if err != nil {
		t.Fatalf("replaytest: marshal golden records: %v", err)
	}
	data = append(data, '\n')

	if *update {
		if err := os.MkdirAll(filepath.Dir(path), 0o750); err != nil {
			t.Fatalf("replaytest: create golden directory: %v", err)
		}
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatalf("replaytest: write golden %s: %v", path, err)
		}
		return
	}

	existing, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			t.Fatalf("replaytest: golden file %s does not exist; run with -update to create it", path)
		}
		t.Fatalf("replaytest: read golden %s: %v", path, err)
	}
	if !bytes.Equal(normalizeEOL(existing), normalizeEOL(data)) {
		t.Fatalf("replaytest: golden mismatch for %s:\n%s", path,
			UnifiedDiff(string(normalizeEOL(existing)), string(normalizeEOL(data))))
	}
}

// normalizeEOL rewrites CRLF line endings to LF so comparisons are stable
// regardless of how git checked the files out.
func normalizeEOL(data []byte) []byte {
	return bytes.ReplaceAll(data, []byte("\r\n"), []byte("\n"))
}

// UnifiedDiff renders a unified diff (three lines of context) between two
// newline-terminated texts.
func UnifiedDiff(before, after string) string {
	oldLines := splitLines(before)
	newLines := splitLines(after)

	ops := diffOps(oldLines, newLines)
	var out strings.Builder
	for _, hunk := range groupHunks(ops, 3) {
		fmt.Fprintf(&out, "@@ -%d,%d +%d,%d @@\n", hunk.oldStart, hunk.oldLines, hunk.newStart, hunk.newLines)
		for _, op := range hunk.ops {
			switch op.kind {
			case opEqual:
				fmt.Fprintf(&out, " %s\n", op.text)
			case opDelete:
				fmt.Fprintf(&out, "-%s\n", op.text)
			case opInsert:
				fmt.Fprintf(&out, "+%s\n", op.text)
			}
		}
	}
	return out.String()
}

func splitLines(s string) []string {
	s = strings.TrimSuffix(s, "\n")
	if s == "" {
		return nil
	}
	return strings.Split(s, "\n")
}

type diffOpKind int

const (
	opEqual diffOpKind = iota
	opDelete
	opInsert
)

type diffOp struct {
	kind diffOpKind
	text string
}

// diffOps walks the longest common subsequence of the two line slices and
// emits the edits that turn old into new.
func diffOps(oldLines, newLines []string) []diffOp {
	n, m := len(oldLines), len(newLines)
	lcs := make([][]int, n+1)
	for i := range lcs {
		lcs[i] = make([]int, m+1)
	}
	for i := n - 1; i >= 0; i-- {
		for j := m - 1; j >= 0; j-- {
			if oldLines[i] == newLines[j] {
				lcs[i][j] = lcs[i+1][j+1] + 1
			} else if lcs[i+1][j] >= lcs[i][j+1] {
				lcs[i][j] = lcs[i+1][j]
			} else {
				lcs[i][j] = lcs[i][j+1]
			}
		}
	}

	ops := make([]diffOp, 0, n+m)
	i, j := 0, 0
	for i < n && j < m {
		switch {
		case oldLines[i] == newLines[j]:
			ops = append(ops, diffOp{kind: opEqual, text: oldLines[i]})
			i++
			j++
		case lcs[i+1][j] >= lcs[i][j+1]:
			ops = append(ops, diffOp{kind: opDelete, text: oldLines[i]})
			i++
		default:
			ops = append(ops, diffOp{kind: opInsert, text: newLines[j]})
			j++
		}
	}
	for ; i < n; i++ {
		ops = append(ops, diffOp{kind: opDelete, text: oldLines[i]})
	}
	for ; j < m; j++ {
		ops = append(ops, diffOp{kind: opInsert, text: newLines[j]})
	}
	return ops
}

type diffHunk struct {
	ops      []diffOp
	oldStart int
	oldLines int
	newStart int
	newLines int
}

// groupHunks merges runs of edits that are closer than twice the context
// size and pads the surrounding context.
func groupHunks(ops []diffOp, context int) []diffHunk {
	changed := make([]int, 0, len(ops))
	for i, op := range ops {
		if op.kind != opEqual {
			changed = append(changed, i)
		}
	}
	if len(changed) == 0 {
		return nil
	}

	groups := [][]int{{changed[0]}}
	for _, index := range changed[1:] {
		last := groups[len(groups)-1]
		if index-last[len(last)-1] <= 2*context {
			groups[len(groups)-1] = append(last, index)
			continue
		}
		groups = append(groups, []int{index})
	}

	hunks := make([]diffHunk, 0, len(groups))
	for _, group := range groups {
		start := max(group[0]-context, 0)
		end := min(group[len(group)-1]+context+1, len(ops))
		hunk := diffHunk{ops: ops[start:end]}
		oldSeen, newSeen := false, false
		for _, op := range hunk.ops {
			switch op.kind {
			case opEqual:
				hunk.oldLines++
				hunk.newLines++
			case opDelete:
				oldSeen = true
				hunk.oldLines++
			case opInsert:
				newSeen = true
				hunk.newLines++
			}
		}
		for _, op := range ops[:start] {
			switch op.kind {
			case opEqual, opDelete:
				hunk.oldStart++
			case opInsert:
				hunk.newStart++
			}
		}
		for _, op := range ops[end:] {
			switch op.kind {
			case opEqual, opDelete:
				hunk.oldStart++
			case opInsert:
				hunk.newStart++
			}
		}
		if !oldSeen {
			hunk.oldLines = 0
		}
		if !newSeen {
			hunk.newLines = 0
		}
		hunk.oldStart++
		hunk.newStart++
		hunks = append(hunks, hunk)
	}
	return hunks
}
