package schema

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestGenerateTypedEnums(t *testing.T) {
	type level int
	type options struct {
		Count   int     `json:"count" enum:"1, 2"`
		Ratio   float64 `json:"ratio" enum:"0.5, 1.5"`
		Enabled bool    `json:"enabled" enum:"true, false"`
		Level   *level  `json:"level" enum:"1, 2"`
		Label   string  `json:"label" enum:"true, 1"`
	}
	s := Generate(reflect.TypeFor[options]())
	_, err := ParseAndValidate(`{"count":1,"ratio":0.5,"enabled":false,"level":2,"label":"true"}`, s)
	require.NoError(t, err)

	_, err = ParseAndValidate(`{"count":3,"ratio":0.5,"enabled":false,"level":2,"label":"true"}`, s)
	require.Error(t, err)
	_, err = ParseAndValidate(`{"count":1,"ratio":2.5,"enabled":false,"level":2,"label":"true"}`, s)
	require.Error(t, err)
	require.Equal(t, []any{"true", "1"}, s.Properties["label"].Enum)
}

func TestGenerateIntegerEnumsPreservePrecision(t *testing.T) {
	type options struct {
		Signed   int64  `json:"signed" enum:"-9223372036854775808,9223372036854775807"`
		Unsigned uint64 `json:"unsigned" enum:"18446744073709551615"`
	}
	s := Generate(reflect.TypeFor[options]())
	data, err := json.Marshal(ToParameters(s))
	require.NoError(t, err)
	require.Contains(t, string(data), `"enum":[-9223372036854775808,9223372036854775807]`)
	require.Contains(t, string(data), `"enum":[18446744073709551615]`)
}

func TestGenerateInvalidNumericEnumsKeepOriginalValues(t *testing.T) {
	type options struct {
		Count   int     `json:"count" enum:"invalid"`
		Ratio   float64 `json:"ratio" enum:"NaN,null"`
		Enabled bool    `json:"enabled" enum:"invalid"`
	}
	s := Generate(reflect.TypeFor[options]())
	require.Equal(t, []any{"invalid"}, s.Properties["count"].Enum)
	require.Equal(t, []any{"NaN", "null"}, s.Properties["ratio"].Enum)
	require.Equal(t, []any{"invalid"}, s.Properties["enabled"].Enum)
	_, err := json.Marshal(s)
	require.NoError(t, err)
}
