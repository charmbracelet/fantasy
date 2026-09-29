package jsonrepair

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestRepairJSONStandardEscapes(t *testing.T) {
	cases := []string{
		`{"text":"page\fbreak"}`,
		`{"text":"https:\/\/example.com\/path"}`,
		`{"\f":"value"}`,
		`["\f", "\/", "\b", "\t", "\n", "\r"]`,
	}
	for _, input := range cases {
		t.Run(input, func(t *testing.T) {
			var want, got any
			if err := json.Unmarshal([]byte(input), &want); err != nil {
				t.Fatal(err)
			}
			repaired, err := RepairJSON(input)
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal([]byte(repaired), &got); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("RepairJSON(%q) = %q; decoded got %#v, want %#v", input, repaired, got, want)
			}
		})
	}
}
