package openai

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestHTTPErrorEnvelopes(t *testing.T) {
	t.Parallel()

	for _, useResponses := range []bool{false, true} {
		for _, operation := range []string{"generate", "stream", "generate_object", "stream_object"} {
			for _, test := range []struct {
				name    string
				body    string
				message string
				status  int
			}{
				{"string_auth", `{"error":"authentication failed"}`, "authentication failed", http.StatusUnauthorized},
				{"string_rate_limit", `{"error":"rate limited"}`, "rate limited", http.StatusTooManyRequests},
				{"string_server", `{"error":"unavailable"}`, "unavailable", http.StatusServiceUnavailable},
				{"string_invalid", `{"error":"invalid request"}`, "invalid request", http.StatusBadRequest},
				{"object", `{"error":{"message":"authentication failed","type":"authentication_error"}}`, "authentication failed", http.StatusUnauthorized},
				{"array", `{"error":["unavailable"]}`, `{"error":["unavailable"]}`, http.StatusServiceUnavailable},
			} {
				api := "chat"
				if useResponses {
					api = "responses"
				}
				t.Run(api+"/"+operation+"/"+test.name, func(t *testing.T) {
					t.Parallel()
					server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
						writer.Header().Set("Content-Type", "application/json")
						writer.Header().Set("Retry-After", "3")
						writer.WriteHeader(test.status)
						_, _ = io.WriteString(writer, test.body)
					}))
					defer server.Close()

					opts := []Option{WithAPIKey("test"), WithBaseURL(server.URL)}
					if useResponses {
						opts = append(opts, WithUseResponsesAPI())
					}
					provider, err := New(opts...)
					require.NoError(t, err)
					model, err := provider.LanguageModel(t.Context(), "gpt-4.1")
					require.NoError(t, err)
					call := fantasy.Call{Prompt: testPrompt}
					objectCall := fantasy.ObjectCall{Prompt: testPrompt, Schema: fantasy.Schema{Type: "object", Properties: map[string]*fantasy.Schema{"answer": {Type: "string"}}}}
					var callErr error
					switch operation {
					case "generate":
						_, callErr = model.Generate(t.Context(), call)
					case "stream":
						stream, err := model.Stream(t.Context(), call)
						require.NoError(t, err)
						for part := range stream {
							if part.Type == fantasy.StreamPartTypeError {
								callErr = part.Error
							}
						}
					case "generate_object":
						_, callErr = model.GenerateObject(t.Context(), objectCall)
					case "stream_object":
						stream, err := model.StreamObject(t.Context(), objectCall)
						require.NoError(t, err)
						for part := range stream {
							if part.Type == fantasy.ObjectStreamPartTypeError {
								callErr = part.Error
							}
						}
					}
					var providerErr *fantasy.ProviderError
					require.ErrorAs(t, callErr, &providerErr)
					require.Equal(t, test.status, providerErr.StatusCode)
					require.Equal(t, test.message, providerErr.Message)
					require.Equal(t, "3", providerErr.ResponseHeaders["Retry-After"])
					require.Contains(t, string(providerErr.ResponseBody), test.body)
					require.Contains(t, providerErr.URL, server.URL)
					require.NotNil(t, providerErr.Cause)
					require.Equal(t, test.status == http.StatusTooManyRequests || test.status >= 500, providerErr.IsRetryable())
				})
			}
		}
	}
}
