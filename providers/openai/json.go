package openai

import (
	"encoding/json"
)

func isValidJSON[T string | []byte](data T) bool {
	if len(data) == 0 { // hot path
		return false
	}
	var m json.RawMessage
	err := json.Unmarshal([]byte(data), &m)
	return err == nil
}
