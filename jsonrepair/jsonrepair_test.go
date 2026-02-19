package jsonrepair

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestRepairJSON(t *testing.T) {
	cases := []struct {
		name  string
		input string
		opts  []Option
		want  string
	}{
		{
			name:  "valid_object",
			input: "{\"name\": \"John\", \"age\": 30, \"city\": \"New York\"}",
			want:  "{\"name\": \"John\", \"age\": 30, \"city\": \"New York\"}",
		},
		{
			name:  "array_spacing",
			input: "{\"employees\":[\"John\", \"Anna\", \"Peter\"]} ",
			want:  "{\"employees\": [\"John\", \"Anna\", \"Peter\"]}",
		},
		{
			name:  "colon_in_string",
			input: "{\"key\": \"value:value\"}",
			want:  "{\"key\": \"value:value\"}",
		},
		{
			name:  "trailing_comma_in_string",
			input: "{\"text\": \"The quick brown fox,\"}",
			want:  "{\"text\": \"The quick brown fox,\"}",
		},
		{
			name:  "apostrophe_in_string",
			input: "{\"text\": \"The quick brown fox won't jump\"}",
			want:  "{\"text\": \"The quick brown fox won't jump\"}",
		},
		{
			name:  "missing_brace",
			input: "{\"key\": \"\"",
			want:  "{\"key\": \"\"}",
		},
		{
			name:  "nested_object",
			input: "{\"key1\": {\"key2\": [1, 2, 3]}}",
			want:  "{\"key1\": {\"key2\": [1, 2, 3]}}",
		},
		{
			name:  "large_integer",
			input: "{\"key\": 12345678901234567890}",
			want:  "{\"key\": 12345678901234567890}",
		},
		{
			name:  "unicode_escape",
			input: "{\"key\": \"value☺\"}",
			want:  "{\"key\": \"value\\u263a\"}",
		},
		{
			name:  "escaped_newline",
			input: "{\"key\": \"value\\nvalue\"}",
			want:  "{\"key\": \"value\\nvalue\"}",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := RepairJSON(tc.input, tc.opts...)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestRepairJSONMultipleTopLevel(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "array_then_object",
			input: "[]{}",
			want:  "[]",
		},
		{
			name:  "array_then_object_with_value",
			input: "[]{\"key\":\"value\"}",
			want:  "{\"key\": \"value\"}",
		},
		{
			name:  "object_then_array",
			input: "{\"key\":\"value\"}[1,2,3,True]",
			want:  "[{\"key\": \"value\"}, [1, 2, 3, true]]",
		},
		{
			name:  "embedded_code_blocks",
			input: "lorem ```json {\"key\":\"value\"} ``` ipsum ```json [1,2,3,True] ``` 42",
			want:  "[{\"key\": \"value\"}, [1, 2, 3, true]]",
		},
		{
			name:  "array_followed_by_array",
			input: "[{\"key\":\"value\"}][{\"key\":\"value_after\"}]",
			want:  "[{\"key\": \"value_after\"}]",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := RepairJSON(tc.input)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestRepairJSONEnsureASCII(t *testing.T) {
	got, err := RepairJSON("{'test_中国人_ascii':'统一码'}", WithEnsureASCII(false))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := "{\"test_中国人_ascii\": \"统一码\"}"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestRepairJSONStreamStable(t *testing.T) {
	cases := []struct {
		name  string
		input string
		opts  []Option
		want  string
	}{
		{
			name:  "default_trailing_backslash",
			input: "{\"key\": \"val\\",
			want:  "{\"key\": \"val\\\\\"}",
		},
		{
			name:  "default_trailing_newline",
			input: "{\"key\": \"val\\n",
			want:  "{\"key\": \"val\"}",
		},
		{
			name:  "default_split_object",
			input: "{\"key\": \"val\\n123,`key2:value2",
			want:  "{\"key\": \"val\\n123\", \"key2\": \"value2\"}",
		},
		{
			name:  "stable_trailing_backslash",
			input: "{\"key\": \"val\\",
			opts:  []Option{WithStreamStable()},
			want:  "{\"key\": \"val\"}",
		},
		{
			name:  "stable_trailing_newline",
			input: "{\"key\": \"val\\n",
			opts:  []Option{WithStreamStable()},
			want:  "{\"key\": \"val\\n\"}",
		},
		{
			name:  "stable_split_object",
			input: "{\"key\": \"val\\n123,`key2:value2",
			opts:  []Option{WithStreamStable()},
			want:  "{\"key\": \"val\\n123,`key2:value2\"}",
		},
		{
			name:  "stable_complete_stream",
			input: "{\"key\": \"val\\n123,`key2:value2`\"}",
			opts:  []Option{WithStreamStable()},
			want:  "{\"key\": \"val\\n123,`key2:value2`\"}",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := RepairJSON(tc.input, tc.opts...)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestLoads(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  any
	}{
		{
			name:  "empty_array",
			input: "[]",
			want:  []any{},
		},
		{
			name:  "empty_object",
			input: "{}",
			want:  map[string]any{},
		},
		{
			name:  "bools_nulls",
			input: "{\"key\": true, \"key2\": false, \"key3\": null}",
			want: map[string]any{
				"key":  true,
				"key2": false,
				"key3": nil,
			},
		},
		{
			name:  "simple_object",
			input: "{\"name\": \"John\", \"age\": 30, \"city\": \"New York\"}",
			want: map[string]any{
				"name": "John",
				"age":  json.Number("30"),
				"city": "New York",
			},
		},
		{
			name:  "array_numbers",
			input: "[1, 2, 3, 4]",
			want: []any{
				json.Number("1"),
				json.Number("2"),
				json.Number("3"),
				json.Number("4"),
			},
		},
		{
			name:  "string_array",
			input: "{\"employees\":[\"John\", \"Anna\", \"Peter\"]} ",
			want: map[string]any{
				"employees": []any{"John", "Anna", "Peter"},
			},
		},
		{
			name:  "string_quotes_repaired",
			input: "[{\"foo\": \"foo bar \"foobar\" foo bar baz.\", \"tag\": \"#foo-bar-foobar\"}]",
			want: []any{
				map[string]any{
					"foo": "foo bar \"foobar\" foo bar baz.",
					"tag": "#foo-bar-foobar",
				},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := Loads(tc.input)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got %#v want %#v", got, tc.want)
			}
		})
	}
}

func TestRepairJSONSkipJSONLoads(t *testing.T) {
	cases := []struct {
		name  string
		input string
		opts  []Option
		want  string
	}{
		{
			name:  "valid_json",
			input: "{\"key\": true, \"key2\": false, \"key3\": null}",
			opts:  []Option{WithSkipJSONLoads()},
			want:  "{\"key\": true, \"key2\": false, \"key3\": null}",
		},
		{
			name:  "missing_value",
			input: "{\"key\": true, \"key2\": false, \"key3\": }",
			opts:  []Option{WithSkipJSONLoads()},
			want:  "{\"key\": true, \"key2\": false, \"key3\": \"\"}",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := RepairJSON(tc.input, tc.opts...)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}

	got, err := Loads("{\"key\": true, \"key2\": false, \"key3\": }", WithSkipJSONLoads())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := map[string]any{
		"key":  true,
		"key2": false,
		"key3": "",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %#v want %#v", got, want)
	}
}

func TestRepairJSONWithLog(t *testing.T) {
	cases := []struct {
		name      string
		input     string
		wantValue any
		wantLog   []LogEntry
	}{
		{
			name:      "valid_json",
			input:     "{}",
			wantValue: map[string]any{},
			wantLog:   []LogEntry{},
		},
		{
			name:  "missing_quote",
			input: "{\"key\": \"value}",
			wantValue: map[string]any{
				"key": "value",
			},
			wantLog: []LogEntry{
				{
					Context: "y\": \"value}",
					Text:    "While parsing a string missing the left delimiter in object value context, we found a , or } and we couldn't determine that a right delimiter was present. Stopping here",
				},
				{
					Context: "y\": \"value}",
					Text:    "While parsing a string, we missed the closing quote, ignoring",
				},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gotValue, gotLog, err := RepairJSONWithLog(tc.input)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(gotValue, tc.wantValue) {
				t.Fatalf("got %#v want %#v", gotValue, tc.wantValue)
			}
			if !reflect.DeepEqual(gotLog, tc.wantLog) {
				t.Fatalf("got %#v want %#v", gotLog, tc.wantLog)
			}
		})
	}
}

func TestRepairJSONStrict(t *testing.T) {
	cases := []struct {
		name    string
		input   string
		opts    []Option
		wantErr string
	}{
		{
			name:    "multiple_top_level",
			input:   "{\"key\":\"value\"}[\"value\"]",
			opts:    []Option{WithStrict()},
			wantErr: "multiple top-level JSON elements",
		},
		{
			name:    "duplicate_keys_in_array",
			input:   "[{\"key\": \"first\", \"key\": \"second\"}]",
			opts:    []Option{WithStrict(), WithSkipJSONLoads()},
			wantErr: "duplicate key found",
		},
		{
			name:    "empty_key",
			input:   "{\"\" : \"value\"}",
			opts:    []Option{WithStrict(), WithSkipJSONLoads()},
			wantErr: "empty key found",
		},
		{
			name:    "missing_colon",
			input:   "{\"missing\" \"colon\"}",
			opts:    []Option{WithStrict()},
			wantErr: "missing ':' after key",
		},
		{
			name:    "empty_value",
			input:   "{\"key\": , \"key2\": \"value2\"}",
			opts:    []Option{WithStrict(), WithSkipJSONLoads()},
			wantErr: "parsed value is empty",
		},
		{
			name:    "empty_object_with_extra",
			input:   "{\"dangling\"}",
			opts:    []Option{WithStrict()},
			wantErr: "parsed object is empty",
		},
		{
			name:    "immediate_doubled_quotes",
			input:   "{\"key\": \"\"\"\"}",
			opts:    []Option{WithStrict()},
			wantErr: "doubled quotes followed by another quote",
		},
		{
			name:    "doubled_quotes_followed_by_string",
			input:   "{\"key\": \"\" \"value\"}",
			opts:    []Option{WithStrict()},
			wantErr: "doubled quotes followed by another quote while parsing a string",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := RepairJSON(tc.input, tc.opts...)
			if err == nil {
				t.Fatalf("expected error")
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("got %q want %q", err.Error(), tc.wantErr)
			}
		})
	}
}

func TestParseArrayObjects(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  any
	}{
		{
			name:  "empty_array",
			input: "[]",
			want:  []any{},
		},
		{
			name:  "numbers_array",
			input: "[1, 2, 3, 4]",
			want: []any{
				json.Number("1"),
				json.Number("2"),
				json.Number("3"),
				json.Number("4"),
			},
		},
		{
			name:  "unfinished_array",
			input: "[",
			want:  []any{},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := Loads(tc.input)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got %#v want %#v", got, tc.want)
			}
		})
	}
}

func TestParseArrayEdgeCases(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "nested_newlines",
			input: "[[1\n\n]",
			want:  "[[1]]",
		},
		{
			name:  "array_with_object_end",
			input: "[{]",
			want:  "[]",
		},
		{
			name:  "just_open_bracket",
			input: "[",
			want:  "[]",
		},
		{
			name:  "dangling_quote",
			input: "[\"",
			want:  "[]",
		},
		{
			name:  "just_close_bracket",
			input: "]",
			want:  "",
		},
		{
			name:  "trailing_comma",
			input: "[1, 2, 3,",
			want:  "[1, 2, 3]",
		},
		{
			name:  "ellipsis_end",
			input: "[1, 2, 3, ...]",
			want:  "[1, 2, 3]",
		},
		{
			name:  "ellipsis_middle",
			input: "[1, 2, ... , 3]",
			want:  "[1, 2, 3]",
		},
		{
			name:  "ellipsis_string",
			input: "[1, 2, '...', 3]",
			want:  "[1, 2, \"...\", 3]",
		},
		{
			name:  "ellipsis_bools",
			input: "[true, false, null, ...]",
			want:  "[true, false, null]",
		},
		{
			name:  "missing_commas",
			input: "[\"a\" \"b\" \"c\" 1",
			want:  "[\"a\", \"b\", \"c\", 1]",
		},
		{
			name:  "object_array_missing_end",
			input: "{\"employees\":[\"John\", \"Anna\",",
			want:  "{\"employees\": [\"John\", \"Anna\"]}",
		},
		{
			name:  "object_array_missing_quote",
			input: "{\"employees\":[\"John\", \"Anna\", \"Peter",
			want:  "{\"employees\": [\"John\", \"Anna\", \"Peter\"]}",
		},
		{
			name:  "nested_object_array",
			input: "{\"key1\": {\"key2\": [1, 2, 3",
			want:  "{\"key1\": {\"key2\": [1, 2, 3]}}",
		},
		{
			name:  "missing_array_quote",
			input: "{\"key\": [\"value]}",
			want:  "{\"key\": [\"value\"]}",
		},
		{
			name:  "embedded_quotes",
			input: "[\"lorem \"ipsum\" sic\"]",
			want:  "[\"lorem \\\"ipsum\\\" sic\"]",
		},
		{
			name:  "array_closes_object",
			input: "{\"key1\": [\"value1\", \"value2\"}, \"key2\": [\"value3\", \"value4\"]}",
			want:  "{\"key1\": [\"value1\", \"value2\"], \"key2\": [\"value3\", \"value4\"]}",
		},
		{
			name:  "rows_missing_bracket",
			input: "{\"headers\": [\"A\", \"B\", \"C\"], \"rows\": [[\"r1a\", \"r1b\", \"r1c\"], [\"r2a\", \"r2b\", \"r2c\"], \"r3a\", \"r3b\", \"r3c\"], [\"r4a\", \"r4b\", \"r4c\"], [\"r5a\", \"r5b\", \"r5c\"]]}",
			want:  "{\"headers\": [\"A\", \"B\", \"C\"], \"rows\": [[\"r1a\", \"r1b\", \"r1c\"], [\"r2a\", \"r2b\", \"r2c\"], [\"r3a\", \"r3b\", \"r3c\"], [\"r4a\", \"r4b\", \"r4c\"], [\"r5a\", \"r5b\", \"r5c\"]]}",
		},
		{
			name:  "array_missing_commas",
			input: "{\"key\": [\"value\" \"value1\" \"value2\"]}",
			want:  "{\"key\": [\"value\", \"value1\", \"value2\"]}",
		},
		{
			name:  "array_many_quotes",
			input: "{\"key\": [\"lorem \"ipsum\" dolor \"sit\" amet, \"consectetur\" \", \"lorem \"ipsum\" dolor\", \"lorem\"]}",
			want:  "{\"key\": [\"lorem \\\"ipsum\\\" dolor \\\"sit\\\" amet, \\\"consectetur\\\" \", \"lorem \\\"ipsum\\\" dolor\", \"lorem\"]}",
		},
		{
			name:  "quoted_key_characters",
			input: "{\"k\"e\"y\": \"value\"}",
			want:  "{\"k\\\"e\\\"y\": \"value\"}",
		},
		{
			name:  "array_object_mixed",
			input: "[\"key\":\"value\"}]",
			want:  "[{\"key\": \"value\"}]",
		},
		{
			name:  "array_object_followed_by_literal",
			input: "[{\"key\": \"value\", \"key",
			want:  "[{\"key\": \"value\"}, [\"key\"]]",
		},
		{
			name:  "set_like_array",
			input: "{'key1', 'key2'}",
			want:  "[\"key1\", \"key2\"]",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := RepairJSON(tc.input)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestParseArrayMissingQuotes(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "value_missing_quote",
			input: "[\"value1\" value2\", \"value3\"]",
			want:  "[\"value1\", \"value2\", \"value3\"]",
		},
		{
			name:  "comment_token",
			input: "{\"bad_one\":[\"Lorem Ipsum\", \"consectetur\" comment\" ], \"good_one\":[ \"elit\", \"sed\", \"tempor\"]}",
			want:  "{\"bad_one\": [\"Lorem Ipsum\", \"consectetur\", \"comment\"], \"good_one\": [\"elit\", \"sed\", \"tempor\"]}",
		},
		{
			name:  "comment_token_no_space",
			input: "{\"bad_one\": [\"Lorem Ipsum\",\"consectetur\" comment],\"good_one\": [\"elit\",\"sed\",\"tempor\"]}",
			want:  "{\"bad_one\": [\"Lorem Ipsum\", \"consectetur\", \"comment\"], \"good_one\": [\"elit\", \"sed\", \"tempor\"]}",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := RepairJSON(tc.input)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestParseComment(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "just_slash",
			input: "/",
			want:  "",
		},
		{
			name:  "block_comment_prefix",
			input: "/* comment */ {\"key\": \"value\"}",
			want:  "{\"key\": \"value\"}",
		},
		{
			name:  "line_comment",
			input: "{ \"key\": { \"key2\": \"value2\" // comment }, \"key3\": \"value3\" }",
			want:  "{\"key\": {\"key2\": \"value2\"}, \"key3\": \"value3\"}",
		},
		{
			name:  "hash_comment",
			input: "{ \"key\": { \"key2\": \"value2\" # comment }, \"key3\": \"value3\" }",
			want:  "{\"key\": {\"key2\": \"value2\"}, \"key3\": \"value3\"}",
		},
		{
			name:  "block_comment_inside",
			input: "{ \"key\": { \"key2\": \"value2\" /* comment */ }, \"key3\": \"value3\" }",
			want:  "{\"key\": {\"key2\": \"value2\"}, \"key3\": \"value3\"}",
		},
		{
			name:  "array_block_comment",
			input: "[ \"value\", /* comment */ \"value2\" ]",
			want:  "[\"value\", \"value2\"]",
		},
		{
			name:  "unterminated_comment",
			input: "{ \"key\": \"value\" /* comment",
			want:  "{\"key\": \"value\"}",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := RepairJSON(tc.input)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestParseNumber(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  any
	}{
		{
			name:  "integer",
			input: "1",
			want:  json.Number("1"),
		},
		{
			name:  "float",
			input: "1.2",
			want:  json.Number("1.2"),
		},
		{
			name:  "underscored_integer",
			input: "{\"value\": 82_461_110}",
			want: map[string]any{
				"value": json.Number("82461110"),
			},
		},
		{
			name:  "underscored_float",
			input: "{\"value\": 1_234.5_6}",
			want: map[string]any{
				"value": json.Number("1234.56"),
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := Loads(tc.input)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got %#v want %#v", got, tc.want)
			}
		})
	}
}

func TestParseNumberEdgeCases(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "leading_dash",
			input: " - { \"test_key\": [\"test_value\", \"test_value2\"] }",
			want:  "{\"test_key\": [\"test_value\", \"test_value2\"]}",
		},
		{
			name:  "fraction",
			input: "{\"key\": 1/3}",
			want:  "{\"key\": \"1/3\"}",
		},
		{
			name:  "leading_decimal",
			input: "{\"key\": .25}",
			want:  "{\"key\": 0.25}",
		},
		{
			name:  "fraction_in_object",
			input: "{\"here\": \"now\", \"key\": 1/3, \"foo\": \"bar\"}",
			want:  "{\"here\": \"now\", \"key\": \"1/3\", \"foo\": \"bar\"}",
		},
		{
			name:  "fraction_long",
			input: "{\"key\": 12345/67890}",
			want:  "{\"key\": \"12345/67890\"}",
		},
		{
			name:  "array_incomplete",
			input: "[105,12",
			want:  "[105, 12]",
		},
		{
			name:  "object_numbers",
			input: "{\"key\", 105,12,",
			want:  "{\"key\": \"105,12\"}",
		},
		{
			name:  "fraction_trailing",
			input: "{\"key\": 1/3, \"foo\": \"bar\"}",
			want:  "{\"key\": \"1/3\", \"foo\": \"bar\"}",
		},
		{
			name:  "dash_number",
			input: "{\"key\": 10-20}",
			want:  "{\"key\": \"10-20\"}",
		},
		{
			name:  "double_dot",
			input: "{\"key\": 1.1.1}",
			want:  "{\"key\": \"1.1.1\"}",
		},
		{
			name:  "dash_array",
			input: "[- ",
			want:  "[]",
		},
		{
			name:  "trailing_decimal",
			input: "{\"key\": 1. }",
			want:  "{\"key\": 1.0}",
		},
		{
			name:  "exponent",
			input: "{\"key\": 1e10 }",
			want:  "{\"key\": 10000000000.0}",
		},
		{
			name:  "bad_exponent",
			input: "{\"key\": 1e }",
			want:  "{\"key\": 1}",
		},
		{
			name:  "non_number_suffix",
			input: "{\"key\": 1notanumber }",
			want:  "{\"key\": \"1notanumber\"}",
		},
		{
			name:  "uuid_literal",
			input: "{\"rowId\": 57eeeeb1-450b-482c-81b9-4be77e95dee2}",
			want:  "{\"rowId\": \"57eeeeb1-450b-482c-81b9-4be77e95dee2\"}",
		},
		{
			name:  "array_non_number",
			input: "[1, 2notanumber]",
			want:  "[1, \"2notanumber\"]",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := RepairJSON(tc.input)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestParseObjectObjects(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  any
	}{
		{
			name:  "empty_object",
			input: "{}",
			want:  map[string]any{},
		},
		{
			name:  "object_values",
			input: "{ \"key\": \"value\", \"key2\": 1, \"key3\": True }",
			want: map[string]any{
				"key":  "value",
				"key2": json.Number("1"),
				"key3": true,
			},
		},
		{
			name:  "unfinished_object",
			input: "{",
			want:  map[string]any{},
		},
		{
			name:  "object_with_literals",
			input: "{ \"key\": value, \"key2\": 1 \"key3\": null }",
			want: map[string]any{
				"key":  "value",
				"key2": json.Number("1"),
				"key3": nil,
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := Loads(tc.input)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got %#v want %#v", got, tc.want)
			}
		})
	}
}

func TestParseObjectEdgeCases(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "empty_trim",
			input: "   {  }   ",
			want:  "{}",
		},
		{
			name:  "just_open",
			input: "{",
			want:  "{}",
		},
		{
			name:  "just_close",
			input: "}",
			want:  "",
		},
		{
			name:  "dangling_quote",
			input: "{\"",
			want:  "{}",
		},
		{
			name:  "object_array_merge",
			input: "{foo: [}",
			want:  "{\"foo\": []}",
		},
		{
			name:  "empty_key",
			input: "{\"\": \"value\"",
			want:  "{\"\": \"value\"}",
		},
		{
			name:  "embedded_quotes",
			input: "{\"key\": \"v\"alue\"}",
			want:  "{\"key\": \"v\\\"alue\\\"\"}",
		},
		{
			name:  "comment_literal",
			input: "{\"value_1\": true, COMMENT \"value_2\": \"data\"}",
			want:  "{\"value_1\": true, \"value_2\": \"data\"}",
		},
		{
			name:  "comment_literal_trailing",
			input: "{\"value_1\": true, SHOULD_NOT_EXIST \"value_2\": \"data\" AAAA }",
			want:  "{\"value_1\": true, \"value_2\": \"data\"}",
		},
		{
			name:  "empty_key_bool",
			input: "{\"\" : true, \"key2\": \"value2\"}",
			want:  "{\"\": true, \"key2\": \"value2\"}",
		},
		{
			name:  "double_quotes",
			input: "{\"\"answer\"\":[{\"\"traits\"\":''Female aged 60+'',\"\"answer1\"\":\"\"5\"\"}]}",
			want:  "{\"answer\": [{\"traits\": \"Female aged 60+\", \"answer1\": \"5\"}]}",
		},
		{
			name:  "missing_quotes",
			input: "{ \"words\": abcdef\", \"numbers\": 12345\", \"words2\": ghijkl\" }",
			want:  "{\"words\": \"abcdef\", \"numbers\": 12345, \"words2\": \"ghijkl\"}",
		},
		{
			name:  "broken_split_key",
			input: "{\"number\": 1,\"reason\": \"According...\"\"ans\": \"YES\"}",
			want:  "{\"number\": 1, \"reason\": \"According...\", \"ans\": \"YES\"}",
		},
		{
			name:  "nested_braces_in_string",
			input: "{ \"a\" : \"{ b\": {} }\" }",
			want:  "{\"a\": \"{ b\"}",
		},
		{
			name:  "literal_after_string",
			input: "{\"b\": \"xxxxx\" true}",
			want:  "{\"b\": \"xxxxx\"}",
		},
		{
			name:  "string_with_quotes",
			input: "{\"key\": \"Lorem \"ipsum\" s,\"}",
			want:  "{\"key\": \"Lorem \\\"ipsum\\\" s,\"}",
		},
		{
			name:  "literal_list",
			input: "{\"lorem\": ipsum, sic, datum.\",}",
			want:  "{\"lorem\": \"ipsum, sic, datum.\"}",
		},
		{
			name:  "multiple_keys",
			input: "{\"lorem\": sic tamet. \"ipsum\": sic tamet, quick brown fox. \"sic\": ipsum}",
			want:  "{\"lorem\": \"sic tamet.\", \"ipsum\": \"sic tamet\", \"sic\": \"ipsum\"}",
		},
		{
			name:  "unfinished_string",
			input: "{\"lorem_ipsum\": \"sic tamet, quick brown fox. }",
			want:  "{\"lorem_ipsum\": \"sic tamet, quick brown fox.\"}",
		},
		{
			name:  "missing_quotes_keys",
			input: "{\"key\":value, \" key2\":\"value2\" }",
			want:  "{\"key\": \"value\", \" key2\": \"value2\"}",
		},
		{
			name:  "missing_quotes_key_separator",
			input: "{\"key\":value \"key2\":\"value2\" }",
			want:  "{\"key\": \"value\", \"key2\": \"value2\"}",
		},
		{
			name:  "single_quotes_braces",
			input: "{'text': 'words{words in brackets}more words'}",
			want:  "{\"text\": \"words{words in brackets}more words\"}",
		},
		{
			name:  "literal_with_braces",
			input: "{text:words{words in brackets}}",
			want:  "{\"text\": \"words{words in brackets}\"}",
		},
		{
			name:  "literal_with_braces_suffix",
			input: "{text:words{words in brackets}m}",
			want:  "{\"text\": \"words{words in brackets}m\"}",
		},
		{
			name:  "trailing_markdown",
			input: "{\"key\": \"value, value2\"```",
			want:  "{\"key\": \"value, value2\"}",
		},
		{
			name:  "trailing_markdown_quote",
			input: "{\"key\": \"value}```",
			want:  "{\"key\": \"value\"}",
		},
		{
			name:  "bare_keys",
			input: "{key:value,key2:value2}",
			want:  "{\"key\": \"value\", \"key2\": \"value2\"}",
		},
		{
			name:  "missing_key_quote",
			input: "{\"key:\"value\"}",
			want:  "{\"key\": \"value\"}",
		},
		{
			name:  "missing_value_quote",
			input: "{\"key:value}",
			want:  "{\"key\": \"value\"}",
		},
		{
			name:  "array_double_quotes",
			input: "[{\"lorem\": {\"ipsum\": \"sic\"}, \"\"\"\" \"lorem\": {\"ipsum\": \"sic\"}]",
			want:  "[{\"lorem\": {\"ipsum\": \"sic\"}}, {\"lorem\": {\"ipsum\": \"sic\"}}]",
		},
		{
			name:  "arrays_in_object",
			input: "{ \"key\": [\"arrayvalue\"], [\"arrayvalue1\"], [\"arrayvalue2\"], \"key3\": \"value3\" }",
			want:  "{\"key\": [\"arrayvalue\", \"arrayvalue1\", \"arrayvalue2\"], \"key3\": \"value3\"}",
		},
		{
			name:  "nested_arrays_in_object",
			input: "{ \"key\": [[1, 2, 3], \"a\", \"b\"], [[4, 5, 6], [7, 8, 9]] }",
			want:  "{\"key\": [[1, 2, 3], \"a\", \"b\", [4, 5, 6], [7, 8, 9]]}",
		},
		{
			name:  "array_key_missing_value",
			input: "{ \"key\": [\"arrayvalue\"], \"key3\": \"value3\", [\"arrayvalue1\"] }",
			want:  "{\"key\": [\"arrayvalue\"], \"key3\": \"value3\", \"arrayvalue1\": \"\"}",
		},
		{
			name:  "json_string_literal",
			input: "{\"key\": \"{\\\"key\\\":[\\\"value\\\"],\\\"key2\\\":\"value2\"}\"}",
			want:  "{\"key\": \"{\\\"key\\\":[\\\"value\\\"],\\\"key2\\\":\\\"value2\\\"}\"}",
		},
		{
			name:  "empty_value",
			input: "{\"key\": , \"key2\": \"value2\"}",
			want:  "{\"key\": \"\", \"key2\": \"value2\"}",
		},
		{
			name:  "array_missing_object_end",
			input: "{\"array\":[{\"key\": \"value\"], \"key2\": \"value2\"}",
			want:  "{\"array\": [{\"key\": \"value\"}], \"key2\": \"value2\"}",
		},
		{
			name:  "object_double_close",
			input: "[{\"key\":\"value\"}},{\"key\":\"value\"}]",
			want:  "[{\"key\": \"value\"}, {\"key\": \"value\"}]",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := RepairJSON(tc.input)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestParseObjectMergeAtEnd(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "merge_key",
			input: "{\"key\": \"value\"}, \"key2\": \"value2\"}",
			want:  "{\"key\": \"value\", \"key2\": \"value2\"}",
		},
		{
			name:  "merge_empty_value",
			input: "{\"key\": \"value\"}, \"key2\": }",
			want:  "{\"key\": \"value\", \"key2\": \"\"}",
		},
		{
			name:  "merge_array_discard",
			input: "{\"key\": \"value\"}, []",
			want:  "{\"key\": \"value\"}",
		},
		{
			name:  "merge_array_keep",
			input: "{\"key\": \"value\"}, [\"abc\"]",
			want:  "[{\"key\": \"value\"}, [\"abc\"]]",
		},
		{
			name:  "merge_object",
			input: "{\"key\": \"value\"}, {}",
			want:  "{\"key\": \"value\"}",
		},
		{
			name:  "merge_empty_key",
			input: "{\"key\": \"value\"}, \"\" : \"value2\"}",
			want:  "{\"key\": \"value\", \"\": \"value2\"}",
		},
		{
			name:  "merge_missing_colon",
			input: "{\"key\": \"value\"}, \"key2\" \"value2\"}",
			want:  "{\"key\": \"value\", \"key2\": \"value2\"}",
		},
		{
			name:  "merge_multiple_keys",
			input: "{\"key1\": \"value1\"}, \"key2\": \"value2\", \"key3\": \"value3\"}",
			want:  "{\"key1\": \"value1\", \"key2\": \"value2\", \"key3\": \"value3\"}",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := RepairJSON(tc.input)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestParseStringBasics(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "just_quote",
			input: "\"",
			want:  "",
		},
		{
			name:  "newline",
			input: "\n",
			want:  "",
		},
		{
			name:  "space",
			input: " ",
			want:  "",
		},
		{
			name:  "string_literal",
			input: "string",
			want:  "",
		},
		{
			name:  "string_before_object",
			input: "stringbeforeobject {}",
			want:  "{}",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := RepairJSON(tc.input)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestMissingAndMixedQuotes(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "mixed_quotes",
			input: "{'key': 'string', 'key2': false, \"key3\": null, \"key4\": unquoted}",
			want:  "{\"key\": \"string\", \"key2\": false, \"key3\": null, \"key4\": \"unquoted\"}",
		},
		{
			name:  "missing_last_quote",
			input: "{\"name\": \"John\", \"age\": 30, \"city\": \"New York",
			want:  "{\"name\": \"John\", \"age\": 30, \"city\": \"New York\"}",
		},
		{
			name:  "missing_quotes_key",
			input: "{\"name\": \"John\", \"age\": 30, city: \"New York\"}",
			want:  "{\"name\": \"John\", \"age\": 30, \"city\": \"New York\"}",
		},
		{
			name:  "missing_quotes_value",
			input: "{\"name\": \"John\", \"age\": 30, \"city\": New York}",
			want:  "{\"name\": \"John\", \"age\": 30, \"city\": \"New York\"}",
		},
		{
			name:  "missing_quotes_value_with_name",
			input: "{\"name\": John, \"age\": 30, \"city\": \"New York\"}",
			want:  "{\"name\": \"John\", \"age\": 30, \"city\": \"New York\"}",
		},
		{
			name:  "slanted_delimiter",
			input: "{“slanted_delimiter”: \"value\"}",
			want:  "{\"slanted_delimiter\": \"value\"}",
		},
		{
			name:  "shortened_string",
			input: "{\"name\": \"John\", \"age\": 30, \"city\": \"New",
			want:  "{\"name\": \"John\", \"age\": 30, \"city\": \"New\"}",
		},
		{
			name:  "missing_quote_in_middle",
			input: "{\"name\": \"John\", \"age\": 30, \"city\": \"New York, \"gender\": \"male\"}",
			want:  "{\"name\": \"John\", \"age\": 30, \"city\": \"New York\", \"gender\": \"male\"}",
		},
		{
			name:  "comment_literal_in_array",
			input: "[{\"key\": \"value\", COMMENT \"notes\": \"lorem \"ipsum\", sic.\" }]",
			want:  "[{\"key\": \"value\", \"notes\": \"lorem \\\"ipsum\\\", sic.\"}]",
		},
		{
			name:  "double_quote_prefix",
			input: "{\"key\": \"\"value\"}",
			want:  "{\"key\": \"value\"}",
		},
		{
			name:  "numeric_key",
			input: "{\"key\": \"value\", 5: \"value\"}",
			want:  "{\"key\": \"value\", \"5\": \"value\"}",
		},
		{
			name:  "escaped_quotes",
			input: "{\"foo\": \"\\\\\"bar\\\\\"\"",
			want:  "{\"foo\": \"\\\"bar\\\"\"}",
		},
		{
			name:  "empty_key_prefix",
			input: "{\"\" key\":\"val\"",
			want:  "{\" key\": \"val\"}",
		},
		{
			name:  "missing_comma",
			input: "{\"key\": value \"key2\" : \"value2\" ",
			want:  "{\"key\": \"value\", \"key2\": \"value2\"}",
		},
		{
			name:  "ellipsis_quotes",
			input: "{\"key\": \"lorem ipsum ... \"sic \" tamet. ...}",
			want:  "{\"key\": \"lorem ipsum ... \\\"sic \\\" tamet. ...\"}",
		},
		{
			name:  "trailing_comma",
			input: "{\"key\": value , }",
			want:  "{\"key\": \"value\"}",
		},
		{
			name:  "comment_in_string",
			input: "{\"comment\": \"lorem, \"ipsum\" sic \"tamet\". To improve\"}",
			want:  "{\"comment\": \"lorem, \\\"ipsum\\\" sic \\\"tamet\\\". To improve\"}",
		},
		{
			name:  "value_with_embedded_quotes",
			input: "{\"key\": \"v\"alu\"e\"} key:",
			want:  "{\"key\": \"v\\\"alu\\\"e\"}",
		},
		{
			name:  "value_with_embedded_quote",
			input: "{\"key\": \"v\"alue\", \"key2\": \"value2\"}",
			want:  "{\"key\": \"v\\\"alue\", \"key2\": \"value2\"}",
		},
		{
			name:  "array_value_with_quote",
			input: "[{\"key\": \"v\"alu,e\", \"key2\": \"value2\"}]",
			want:  "[{\"key\": \"v\\\"alu,e\", \"key2\": \"value2\"}]",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := RepairJSON(tc.input)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestEscaping(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "just_quotes",
			input: "'\"'",
			want:  "",
		},
		{
			name:  "escaped_chars",
			input: "{\"key\": 'string\"\n\t\\\\le'",
			want:  "{\"key\": \"string\\\"\\n\\t\\\\le\"}",
		},
		{
			name:  "html_escape",
			input: "{\"real_content\": \"Some string: Some other string \\t Some string <a href=\\\"https://domain.com\\\">Some link</a>\"",
			want:  "{\"real_content\": \"Some string: Some other string \\t Some string <a href=\\\"https://domain.com\\\">Some link</a>\"}",
		},
		{
			name:  "newline_in_key",
			input: "{\"key_1\n\": \"value\"}",
			want:  "{\"key_1\": \"value\"}",
		},
		{
			name:  "tab_in_key",
			input: "{\"key\t_\": \"value\"}",
			want:  "{\"key\\t_\": \"value\"}",
		},
		{
			name:  "unicode_escape",
			input: "{\"key\": '\u0076\u0061\u006c\u0075\u0065'}",
			want:  "{\"key\": \"value\"}",
		},
		{
			name:  "unicode_escape_skip_loads",
			input: "{\"key\": \"\\u0076\\u0061\\u006C\\u0075\\u0065\"}",
			want:  "{\"key\": \"value\"}",
		},
		{
			name:  "escaped_single_quote",
			input: "{\"key\": \"valu\\'e\"}",
			want:  "{\"key\": \"valu'e\"}",
		},
		{
			name:  "escaped_object",
			input: "{'key': \"{\\\"key\\\": 1, \\\"key2\\\": 1}\"}",
			want:  "{\"key\": \"{\\\"key\\\": 1, \\\"key2\\\": 1}\"}",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			opts := []Option{}
			if tc.name == "unicode_escape_skip_loads" {
				opts = append(opts, WithSkipJSONLoads())
			}
			got, err := RepairJSON(tc.input, opts...)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestMarkdown(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "markdown_link",
			input: "{ \"content\": \"[LINK](\"https://google.com\")\" }",
			want:  "{\"content\": \"[LINK](\\\"https://google.com\\\")\"}",
		},
		{
			name:  "markdown_incomplete",
			input: "{ \"content\": \"[LINK](\" }",
			want:  "{\"content\": \"[LINK](\"}",
		},
		{
			name:  "markdown_in_object",
			input: "{ \"content\": \"[LINK](\", \"key\": true }",
			want:  "{\"content\": \"[LINK](\", \"key\": true}",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := RepairJSON(tc.input)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestLeadingTrailingCharacters(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "wrapped_markdown",
			input: "````{ \"key\": \"value\" }```",
			want:  "{\"key\": \"value\"}",
		},
		{
			name:  "trailing_markdown_block",
			input: "{    \"a\": \"\",    \"b\": [ { \"c\": 1} ] \n}```",
			want:  "{\"a\": \"\", \"b\": [{\"c\": 1}]}",
		},
		{
			name:  "preface_text",
			input: "Based on the information extracted, here is the filled JSON output: ```json { 'a': 'b' } ```",
			want:  "{\"a\": \"b\"}",
		},
		{
			name:  "multiline_markdown",
			input: "\n                       The next 64 elements are:\n                       ```json\n                       { \"key\": \"value\" }\n                       ```",
			want:  "{\"key\": \"value\"}",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := RepairJSON(tc.input)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestStringJSONLLMBlock(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "backticks",
			input: "{\"key\": \"``\"",
			want:  "{\"key\": \"``\"}",
		},
		{
			name:  "backticks_json",
			input: "{\"key\": \"```json\"",
			want:  "{\"key\": \"```json\"}",
		},
		{
			name:  "json_block_inside_string",
			input: "{\"key\": \"```json {\"key\": [{\"key1\": 1},{\"key2\": 2}]}```\"}",
			want:  "{\"key\": {\"key\": [{\"key1\": 1}, {\"key2\": 2}]}}",
		},
		{
			name:  "response_prefix",
			input: "{\"response\": \"```json{}\"",
			want:  "{\"response\": \"```json{}\"}",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := RepairJSON(tc.input)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestParseBooleanOrNull(t *testing.T) {
	loadCases := []struct {
		name  string
		input string
		want  any
	}{
		{
			name:  "upper_true",
			input: "True",
			want:  "",
		},
		{
			name:  "upper_false",
			input: "False",
			want:  "",
		},
		{
			name:  "upper_null",
			input: "Null",
			want:  "",
		},
		{
			name:  "lower_true",
			input: "true",
			want:  true,
		},
		{
			name:  "lower_false",
			input: "false",
			want:  false,
		},
		{
			name:  "lower_null",
			input: "null",
			want:  nil,
		},
	}

	for _, tc := range loadCases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := Loads(tc.input)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got %#v want %#v", got, tc.want)
			}
		})
	}

	stringCases := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "bools_in_object",
			input: "  {\"key\": true, \"key2\": false, \"key3\": null}",
			want:  "{\"key\": true, \"key2\": false, \"key3\": null}",
		},
		{
			name:  "uppercase_bools",
			input: "{\"key\": TRUE, \"key2\": FALSE, \"key3\": Null}   ",
			want:  "{\"key\": true, \"key2\": false, \"key3\": null}",
		},
	}

	for _, tc := range stringCases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := RepairJSON(tc.input)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}
