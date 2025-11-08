package fantasy

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	jsonrepair "github.com/RealAlexandreAI/json-repair"
	"github.com/kaptinlin/jsonschema"
)

// ParseState represents the state of JSON parsing.
type ParseState string

const (
	// ParseStateUndefined means input was undefined/empty.
	ParseStateUndefined ParseState = "undefined"

	// ParseStateSuccessful means JSON parsed without repair.
	ParseStateSuccessful ParseState = "successful"

	// ParseStateRepaired means JSON parsed after repair.
	ParseStateRepaired ParseState = "repaired"

	// ParseStateFailed means JSON could not be parsed even after repair.
	ParseStateFailed ParseState = "failed"
)

// SchemaToJSONSchema converts a Schema to a JSON Schema map.
func SchemaToJSONSchema(schema Schema) map[string]any {
	return SchemaToMap(schema)
}

// ParsePartialJSON attempts to parse potentially incomplete JSON.
// It first tries standard JSON parsing, then attempts repair if that fails.
//
// Returns:
//   - result: The parsed JSON value (map, slice, or primitive)
//   - state: Indicates whether parsing succeeded, needed repair, or failed
//   - err: The error if parsing failed completely
//
// Example:
//
//	obj, state, err := ParsePartialJSON(`{"name": "John", "age": 25`)
//	// Result: map[string]any{"name": "John", "age": 25}, ParseStateRepaired, nil
func ParsePartialJSON(text string) (any, ParseState, error) {
	if text == "" {
		return nil, ParseStateUndefined, nil
	}

	var result any
	if err := json.Unmarshal([]byte(text), &result); err == nil {
		return result, ParseStateSuccessful, nil
	}

	repaired, err := jsonrepair.RepairJSON(text)
	if err != nil {
		return nil, ParseStateFailed, fmt.Errorf("json repair failed: %w", err)
	}

	if err := json.Unmarshal([]byte(repaired), &result); err != nil {
		return nil, ParseStateFailed, fmt.Errorf("failed to parse repaired json: %w", err)
	}

	return result, ParseStateRepaired, nil
}

// ParseAndValidate combines JSON parsing and validation.
// Returns the parsed object if both parsing and validation succeed.
func ParseAndValidate(text string, schema Schema) (any, error) {
	obj, state, err := ParsePartialJSON(text)
	if state == ParseStateFailed {
		return nil, &NoObjectGeneratedError{
			RawText:    text,
			ParseError: err,
		}
	}

	if err := validateAgainstSchema(obj, schema); err != nil {
		return nil, &NoObjectGeneratedError{
			RawText:         text,
			ValidationError: err,
		}
	}

	return obj, nil
}

// ValidateAgainstSchema validates a parsed object against a Schema.
func ValidateAgainstSchema(obj any, schema Schema) error {
	return validateAgainstSchema(obj, schema)
}

func validateAgainstSchema(obj any, schema Schema) error {
	jsonSchemaBytes, err := json.Marshal(schema)
	if err != nil {
		return fmt.Errorf("failed to marshal schema: %w", err)
	}

	compiler := jsonschema.NewCompiler()
	validator, err := compiler.Compile(jsonSchemaBytes)
	if err != nil {
		return fmt.Errorf("invalid schema: %w", err)
	}

	result := validator.Validate(obj)
	if !result.IsValid() {
		var errMsgs []string
		for field, validationErr := range result.Errors {
			errMsgs = append(errMsgs, fmt.Sprintf("%s: %s", field, validationErr.Message))
		}
		return fmt.Errorf("validation failed: %s", strings.Join(errMsgs, "; "))
	}

	return nil
}

// ParseAndValidateWithRepair attempts parsing, validation, and custom repair.
func ParseAndValidateWithRepair(
	ctx context.Context,
	text string,
	schema Schema,
	repair ObjectRepairFunc,
) (any, error) {
	obj, state, parseErr := ParsePartialJSON(text)

	if state == ParseStateSuccessful || state == ParseStateRepaired {
		validationErr := validateAgainstSchema(obj, schema)
		if validationErr == nil {
			return obj, nil
		}

		if repair != nil {
			repairedText, repairErr := repair(ctx, text, validationErr)
			if repairErr == nil {
				obj2, state2, _ := ParsePartialJSON(repairedText)
				if state2 == ParseStateSuccessful || state2 == ParseStateRepaired {
					if err := validateAgainstSchema(obj2, schema); err == nil {
						return obj2, nil
					}
				}
			}
		}

		return nil, &NoObjectGeneratedError{
			RawText:         text,
			ValidationError: validationErr,
		}
	}

	if repair != nil {
		repairedText, repairErr := repair(ctx, text, parseErr)
		if repairErr == nil {
			obj2, state2, parseErr2 := ParsePartialJSON(repairedText)
			if state2 == ParseStateSuccessful || state2 == ParseStateRepaired {
				if err := validateAgainstSchema(obj2, schema); err == nil {
					return obj2, nil
				}
			}
			return nil, &NoObjectGeneratedError{
				RawText:    repairedText,
				ParseError: parseErr2,
			}
		}
	}

	return nil, &NoObjectGeneratedError{
		RawText:    text,
		ParseError: parseErr,
	}
}
