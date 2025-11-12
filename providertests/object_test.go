package providertests

import (
	"context"
	"strings"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

// Object generation tests for providers.
//
// These test functions can be used to test structured object generation
// (GenerateObject and StreamObject) for any provider implementation.
//
// Usage example:
//
//	func TestMyProviderObjectGeneration(t *testing.T) {
//		var pairs []builderPair
//		for _, m := range myTestModels {
//			pairs = append(pairs, builderPair{m.name, myBuilder(m.model), nil, nil})
//		}
//		testObjectGeneration(t, pairs)
//	}
//
// The tests cover:
// - Simple object generation (flat schema with basic types)
// - Complex object generation (nested objects and arrays)
// - Streaming object generation (progressive updates)
// - Object generation with custom repair functions

// testObjectGeneration tests structured object generation for a provider.
// It includes both non-streaming (GenerateObject) and streaming (StreamObject) tests.
func testObjectGeneration(t *testing.T, pairs []builderPair) {
	for _, pair := range pairs {
		t.Run(pair.name, func(t *testing.T) {
			testSimpleObject(t, pair)
			testComplexObject(t, pair)
		})
	}
}

func testSimpleObject(t *testing.T, pair builderPair) {
	// Define a simple schema for a person object
	schema := fantasy.Schema{
		Type: "object",
		Properties: map[string]*fantasy.Schema{
			"name": {
				Type:        "string",
				Description: "The person's name",
			},
			"age": {
				Type:        "integer",
				Description: "The person's age",
			},
			"city": {
				Type:        "string",
				Description: "The city where the person lives",
			},
		},
		Required: []string{"name", "age", "city"},
	}

	checkResult := func(t *testing.T, obj any, rawText string, usage fantasy.Usage) {
		require.NotNil(t, obj, "object should not be nil")
		require.NotEmpty(t, rawText, "raw text should not be empty")
		require.Greater(t, usage.TotalTokens, int64(0), "usage should be tracked")

		// Validate structure
		objMap, ok := obj.(map[string]any)
		require.True(t, ok, "object should be a map")
		require.Contains(t, objMap, "name")
		require.Contains(t, objMap, "age")
		require.Contains(t, objMap, "city")

		// Validate types
		name, ok := objMap["name"].(string)
		require.True(t, ok, "name should be a string")
		require.NotEmpty(t, name, "name should not be empty")

		// Age could be float64 from JSON unmarshaling
		age, ok := objMap["age"].(float64)
		require.True(t, ok, "age should be a number")
		require.Greater(t, age, 0.0, "age should be greater than 0")

		city, ok := objMap["city"].(string)
		require.True(t, ok, "city should be a string")
		require.NotEmpty(t, city, "city should not be empty")
	}

	t.Run("simple object", func(t *testing.T) {
		r := newRecorder(t)

		languageModel, err := pair.builder(t, r)
		require.NoError(t, err, "failed to build language model")

		prompt := fantasy.Prompt{
			fantasy.NewUserMessage("Generate information about a person named Alice who is 30 years old and lives in Paris."),
		}

		response, err := languageModel.GenerateObject(t.Context(), fantasy.ObjectCall{
			Prompt:            prompt,
			Schema:            schema,
			SchemaName:        "Person",
			SchemaDescription: "A person with name, age, and city",
			MaxOutputTokens:   fantasy.Opt(int64(4000)),
			ProviderOptions:   pair.providerOptions,
		})
		require.NoError(t, err, "failed to generate object")
		require.NotNil(t, response, "response should not be nil")
		checkResult(t, response.Object, response.RawText, response.Usage)
	})

	t.Run("simple object streaming", func(t *testing.T) {
		r := newRecorder(t)

		languageModel, err := pair.builder(t, r)
		require.NoError(t, err, "failed to build language model")

		prompt := fantasy.Prompt{
			fantasy.NewUserMessage("Generate information about a person named Alice who is 30 years old and lives in Paris."),
		}

		stream, err := languageModel.StreamObject(t.Context(), fantasy.ObjectCall{
			Prompt:            prompt,
			Schema:            schema,
			SchemaName:        "Person",
			SchemaDescription: "A person with name, age, and city",
			MaxOutputTokens:   fantasy.Opt(int64(4000)),
			ProviderOptions:   pair.providerOptions,
		})
		require.NoError(t, err, "failed to create object stream")
		require.NotNil(t, stream, "stream should not be nil")

		var lastObject any
		var rawText string
		var usage fantasy.Usage
		var finishReason fantasy.FinishReason
		objectCount := 0

		for part := range stream {
			switch part.Type {
			case fantasy.ObjectStreamPartTypeObject:
				lastObject = part.Object
				objectCount++
			case fantasy.ObjectStreamPartTypeTextDelta:
				rawText += part.Delta
			case fantasy.ObjectStreamPartTypeFinish:
				usage = part.Usage
				finishReason = part.FinishReason
			case fantasy.ObjectStreamPartTypeError:
				t.Fatalf("stream error: %v", part.Error)
			}
		}

		require.NotNil(t, lastObject, "should have received at least one object")
		require.Greater(t, objectCount, 0, "should have received object updates")
		require.NotEqual(t, fantasy.FinishReasonUnknown, finishReason, "should have a finish reason")

		// Validate object structure without requiring rawText (may be empty in tool-based mode)
		require.NotNil(t, lastObject, "object should not be nil")
		require.Greater(t, usage.TotalTokens, int64(0), "usage should be tracked")

		// Validate structure
		objMap, ok := lastObject.(map[string]any)
		require.True(t, ok, "object should be a map")
		require.Contains(t, objMap, "name")
		require.Contains(t, objMap, "age")
		require.Contains(t, objMap, "city")

		// Validate types
		name, ok := objMap["name"].(string)
		require.True(t, ok, "name should be a string")
		require.NotEmpty(t, name, "name should not be empty")

		// Age could be float64 from JSON unmarshaling
		age, ok := objMap["age"].(float64)
		require.True(t, ok, "age should be a number")
		require.Greater(t, age, 0.0, "age should be greater than 0")

		city, ok := objMap["city"].(string)
		require.True(t, ok, "city should be a string")
		require.NotEmpty(t, city, "city should not be empty")
	})
}

func testComplexObject(t *testing.T, pair builderPair) {
	// Define a more complex schema with nested objects and arrays
	schema := fantasy.Schema{
		Type: "object",
		Properties: map[string]*fantasy.Schema{
			"title": {
				Type:        "string",
				Description: "The book title",
			},
			"author": {
				Type: "object",
				Properties: map[string]*fantasy.Schema{
					"name": {
						Type:        "string",
						Description: "Author's name",
					},
					"nationality": {
						Type:        "string",
						Description: "Author's nationality",
					},
				},
				Required: []string{"name", "nationality"},
			},
			"genres": {
				Type: "array",
				Items: &fantasy.Schema{
					Type: "string",
				},
				Description: "List of genres",
			},
			"published_year": {
				Type:        "integer",
				Description: "Year the book was published",
			},
		},
		Required: []string{"title", "author", "genres", "published_year"},
	}

	checkResult := func(t *testing.T, obj any, rawText string, usage fantasy.Usage) {
		require.NotNil(t, obj, "object should not be nil")
		require.NotEmpty(t, rawText, "raw text should not be empty")
		require.Greater(t, usage.TotalTokens, int64(0), "usage should be tracked")

		// Validate structure
		objMap, ok := obj.(map[string]any)
		require.True(t, ok, "object should be a map")
		require.Contains(t, objMap, "title")
		require.Contains(t, objMap, "author")
		require.Contains(t, objMap, "genres")
		require.Contains(t, objMap, "published_year")

		// Validate title
		title, ok := objMap["title"].(string)
		require.True(t, ok, "title should be a string")
		require.True(t, strings.Contains(strings.ToLower(title), "rings"), "title should contain 'rings'")

		// Validate nested author object
		author, ok := objMap["author"].(map[string]any)
		require.True(t, ok, "author should be an object")
		require.Contains(t, author, "name")
		require.Contains(t, author, "nationality")

		// Validate genres array
		genres, ok := objMap["genres"].([]any)
		require.True(t, ok, "genres should be an array")
		require.Greater(t, len(genres), 0, "genres should have at least one item")
		for _, genre := range genres {
			_, ok := genre.(string)
			require.True(t, ok, "each genre should be a string")
		}

		// Validate published_year
		year, ok := objMap["published_year"].(float64)
		require.True(t, ok, "published_year should be a number")
		require.Greater(t, year, 1900.0, "published_year should be after 1900")
	}

	t.Run("complex object", func(t *testing.T) {
		r := newRecorder(t)

		languageModel, err := pair.builder(t, r)
		require.NoError(t, err, "failed to build language model")

		prompt := fantasy.Prompt{
			fantasy.NewUserMessage("Generate information about 'The Lord of the Rings' book by J.R.R. Tolkien, including genres like fantasy and adventure, and its publication year (1954)."),
		}

		response, err := languageModel.GenerateObject(t.Context(), fantasy.ObjectCall{
			Prompt:            prompt,
			Schema:            schema,
			SchemaName:        "Book",
			SchemaDescription: "A book with title, author, genres, and publication year",
			MaxOutputTokens:   fantasy.Opt(int64(4000)),
			ProviderOptions:   pair.providerOptions,
		})
		require.NoError(t, err, "failed to generate object")
		require.NotNil(t, response, "response should not be nil")
		checkResult(t, response.Object, response.RawText, response.Usage)
	})

	t.Run("complex object streaming", func(t *testing.T) {
		r := newRecorder(t)

		languageModel, err := pair.builder(t, r)
		require.NoError(t, err, "failed to build language model")

		prompt := fantasy.Prompt{
			fantasy.NewUserMessage("Generate information about 'The Lord of the Rings' book by J.R.R. Tolkien, including genres like fantasy and adventure, and its publication year (1954)."),
		}

		stream, err := languageModel.StreamObject(t.Context(), fantasy.ObjectCall{
			Prompt:            prompt,
			Schema:            schema,
			SchemaName:        "Book",
			SchemaDescription: "A book with title, author, genres, and publication year",
			MaxOutputTokens:   fantasy.Opt(int64(4000)),
			ProviderOptions:   pair.providerOptions,
		})
		require.NoError(t, err, "failed to create object stream")
		require.NotNil(t, stream, "stream should not be nil")

		var lastObject any
		var rawText string
		var usage fantasy.Usage
		var finishReason fantasy.FinishReason
		objectCount := 0

		for part := range stream {
			switch part.Type {
			case fantasy.ObjectStreamPartTypeObject:
				lastObject = part.Object
				objectCount++
			case fantasy.ObjectStreamPartTypeTextDelta:
				rawText += part.Delta
			case fantasy.ObjectStreamPartTypeFinish:
				usage = part.Usage
				finishReason = part.FinishReason
			case fantasy.ObjectStreamPartTypeError:
				t.Fatalf("stream error: %v", part.Error)
			}
		}

		require.NotNil(t, lastObject, "should have received at least one object")
		require.Greater(t, objectCount, 0, "should have received object updates")
		require.NotEqual(t, fantasy.FinishReasonUnknown, finishReason, "should have a finish reason")

		// Validate object structure without requiring rawText (may be empty in tool-based mode)
		require.NotNil(t, lastObject, "object should not be nil")
		require.Greater(t, usage.TotalTokens, int64(0), "usage should be tracked")

		// Validate structure
		objMap, ok := lastObject.(map[string]any)
		require.True(t, ok, "object should be a map")
		require.Contains(t, objMap, "title")
		require.Contains(t, objMap, "author")
		require.Contains(t, objMap, "genres")
		require.Contains(t, objMap, "published_year")

		// Validate title
		title, ok := objMap["title"].(string)
		require.True(t, ok, "title should be a string")
		require.True(t, strings.Contains(strings.ToLower(title), "rings"), "title should contain 'rings'")

		// Validate nested author object
		author, ok := objMap["author"].(map[string]any)
		require.True(t, ok, "author should be an object")
		require.Contains(t, author, "name")
		require.Contains(t, author, "nationality")

		// Validate genres array
		genres, ok := objMap["genres"].([]any)
		require.True(t, ok, "genres should be an array")
		require.Greater(t, len(genres), 0, "genres should have at least one item")
		for _, genre := range genres {
			_, ok := genre.(string)
			require.True(t, ok, "each genre should be a string")
		}

		// Validate published_year
		year, ok := objMap["published_year"].(float64)
		require.True(t, ok, "published_year should be a number")
		require.Greater(t, year, 1900.0, "published_year should be after 1900")
	})
}

// testObjectWithRepair tests object generation with custom repair functionality.
func testObjectWithRepair(t *testing.T, pairs []builderPair) {
	for _, pair := range pairs {
		t.Run(pair.name, func(t *testing.T) {
			t.Run("object with repair", func(t *testing.T) {
				r := newRecorder(t)

				languageModel, err := pair.builder(t, r)
				require.NoError(t, err, "failed to build language model")

				minVal := 1.0
				schema := fantasy.Schema{
					Type: "object",
					Properties: map[string]*fantasy.Schema{
						"count": {
							Type:        "integer",
							Description: "A count that must be positive",
							Minimum:     &minVal,
						},
					},
					Required: []string{"count"},
				}

				prompt := fantasy.Prompt{
					fantasy.NewUserMessage("Return a count of 5"),
				}

				repairFunc := func(ctx context.Context, text string, err error) (string, error) {
					// Simple repair: if the JSON is malformed, try to fix it
					// This is a placeholder - real repair would be more sophisticated
					return text, nil
				}

				response, err := languageModel.GenerateObject(t.Context(), fantasy.ObjectCall{
					Prompt:            prompt,
					Schema:            schema,
					SchemaName:        "Count",
					SchemaDescription: "A simple count object",
					MaxOutputTokens:   fantasy.Opt(int64(4000)),
					RepairText:        repairFunc,
					ProviderOptions:   pair.providerOptions,
				})
				require.NoError(t, err, "failed to generate object")
				require.NotNil(t, response, "response should not be nil")
				require.NotNil(t, response.Object, "object should not be nil")
			})
		})
	}
}
