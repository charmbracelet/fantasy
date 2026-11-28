package object

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/schema"
)

type benchItem struct {
	ID     int     `json:"id"`
	Name   string  `json:"name"`
	Active bool    `json:"active"`
	Score  float64 `json:"score"`
	Desc   string  `json:"desc"`
}

type benchPayload struct {
	Title string      `json:"title"`
	Items []benchItem `json:"items"`
}

func buildBenchPayloadJSON(targetBytes int) string {
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

func BenchmarkStreamWithTool(b *testing.B) {
	const chunkSize = 8
	s := schema.Generate(reflect.TypeFor[benchPayload]())

	for _, size := range []struct {
		name  string
		bytes int
	}{
		{"1k", 1024},
		{"8k", 8192},
	} {
		full := buildBenchPayloadJSON(size.bytes)
		parts := make([]fantasy.StreamPart, 0, (len(full)+chunkSize-1)/chunkSize+1)
		for i := 0; i < len(full); i += chunkSize {
			end := min(i+chunkSize, len(full))
			parts = append(parts, fantasy.StreamPart{
				Type:  fantasy.StreamPartTypeToolInputDelta,
				Delta: full[i:end],
			})
		}
		parts = append(parts, fantasy.StreamPart{
			Type:         fantasy.StreamPartTypeFinish,
			FinishReason: fantasy.FinishReasonStop,
		})
		model := &warningsStreamModel{parts: parts}
		call := fantasy.ObjectCall{
			Prompt: fantasy.Prompt{fantasy.NewUserMessage("generate")},
			Schema: s,
		}

		b.Run(size.name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(full)))
			for b.Loop() {
				stream, err := StreamWithTool(b.Context(), model, call)
				if err != nil {
					b.Fatal(err)
				}
				var count int
				for part := range stream {
					if part.Type == fantasy.ObjectStreamPartTypeObject {
						count++
					}
					if part.Type == fantasy.ObjectStreamPartTypeError {
						b.Fatal(part.Error)
					}
				}
				if count == 0 {
					b.Fatal("expected at least one object")
				}
			}
		})
	}
}
