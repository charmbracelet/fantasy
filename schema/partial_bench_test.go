package schema

import (
	"fmt"
	"strings"
	"testing"
)

func buildBenchJSON(targetBytes int) string {
	var b strings.Builder
	b.WriteString(`{"title":"Streaming Benchmark","items":[`)
	i := 0
	for b.Len() < targetBytes {
		if i > 0 {
			b.WriteByte(',')
		}
		fmt.Fprintf(&b, `{"id":%d,"name":"item-%d","active":true,"score":%d.5,"desc":"sample payload text for realistic streaming JSON parsing"}`, i, i, i)
		i++
	}
	b.WriteString(`]}`)
	return b.String()
}

func BenchmarkParsePartialJSON(b *testing.B) {
	for _, size := range []struct {
		name  string
		bytes int
	}{
		{"1k", 1024},
		{"8k", 8192},
		{"32k", 32768},
	} {
		full := buildBenchJSON(size.bytes)
		// Truncate at ~75% to exercise the partial-repair path.
		partial := full[:len(full)*3/4]
		b.Run(size.name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(partial)))
			for b.Loop() {
				obj, state, err := ParsePartialJSON(partial)
				if err != nil || state != ParseStateRepaired || obj == nil {
					b.Fatalf("unexpected result: state=%v err=%v", state, err)
				}
			}
		})
	}
}

func BenchmarkStreamParsePartialJSON(b *testing.B) {
	const chunkSize = 8
	for _, size := range []struct {
		name  string
		bytes int
	}{
		{"1k", 1024},
		{"8k", 8192},
	} {
		full := buildBenchJSON(size.bytes)
		chunks := make([]string, 0, (len(full)+chunkSize-1)/chunkSize)
		for i := 0; i < len(full); i += chunkSize {
			end := min(i+chunkSize, len(full))
			chunks = append(chunks, full[i:end])
		}
		b.Run(size.name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(full)))
			for b.Loop() {
				p := NewStreamParser()
				var last any
				for _, ch := range chunks {
					obj, state, _ := p.Feed(ch)
					if state == ParseStateSuccessful || state == ParseStateRepaired {
						last = obj
					}
				}
				if last == nil {
					b.Fatal("expected parsed object")
				}
			}
		})
	}
}
