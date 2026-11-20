package fantasy

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
)

func telemetryRecorder(t *testing.T) (*sdktrace.TracerProvider, *tracetest.SpanRecorder) {
	t.Helper()
	r := tracetest.NewSpanRecorder()
	p := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(r))
	t.Cleanup(func() { require.NoError(t, p.Shutdown(context.Background())) })
	return p, r
}

func spanAttributes(s sdktrace.ReadOnlySpan) map[attribute.Key]attribute.Value {
	attrs := make(map[attribute.Key]attribute.Value)
	for _, a := range s.Attributes() {
		attrs[a.Key] = a.Value
	}
	return attrs
}

func TestInstrumentModel(t *testing.T) {
	t.Parallel()
	tp, recorder := telemetryRecorder(t)
	reader := sdkmetric.NewManualReader()
	mp := sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader))
	t.Cleanup(func() { require.NoError(t, mp.Shutdown(context.Background())) })
	model := &mockLanguageModel{generateFunc: func(ctx context.Context, call Call) (*Response, error) {
		require.True(t, trace.SpanFromContext(ctx).IsRecording())
		require.Equal(t, "private prompt", call.Prompt[0].Content[0].(TextPart).Text)
		return &Response{Usage: Usage{InputTokens: 7, OutputTokens: 3}, FinishReason: FinishReasonStop}, nil
	}}
	require.Same(t, model, InstrumentModel(model, nil, nil))
	wrapped := InstrumentModel(model, tp, mp)
	require.Equal(t, model.Provider(), wrapped.Provider())
	require.Equal(t, model.Model(), wrapped.Model())
	_, err := wrapped.Generate(t.Context(), Call{Prompt: Prompt{{Role: MessageRoleUser, Content: []MessagePart{TextPart{Text: "private prompt"}}}}})
	require.NoError(t, err)
	require.Len(t, recorder.Ended(), 1)
	span := recorder.Ended()[0]
	require.Equal(t, "chat mock-model", span.Name())
	require.Equal(t, trace.SpanKindClient, span.SpanKind())
	attrs := spanAttributes(span)
	require.Equal(t, "mock-provider", attrs["gen_ai.provider.name"].AsString())
	require.Equal(t, "mock-model", attrs["gen_ai.request.model"].AsString())
	require.Equal(t, []string{"stop"}, attrs["gen_ai.response.finish_reasons"].AsStringSlice())
	require.EqualValues(t, 7, attrs["gen_ai.usage.input_tokens"].AsInt64())
	require.EqualValues(t, 3, attrs["gen_ai.usage.output_tokens"].AsInt64())
	require.NotContains(t, attrs, attribute.Key("gen_ai.input.messages"))
	var data metricdata.ResourceMetrics
	require.NoError(t, reader.Collect(t.Context(), &data))
	require.Len(t, data.ScopeMetrics, 1)
	require.Len(t, data.ScopeMetrics[0].Metrics, 2)
	for _, m := range data.ScopeMetrics[0].Metrics {
		switch m.Name {
		case "gen_ai.client.operation.duration":
			h := m.Data.(metricdata.Histogram[float64])
			require.Len(t, h.DataPoints, 1)
			require.EqualValues(t, 1, h.DataPoints[0].Count)
		case "gen_ai.client.token.usage":
			h := m.Data.(metricdata.Histogram[int64])
			require.Len(t, h.DataPoints, 2)
			for _, p := range h.DataPoints {
				tokenType, _ := p.Attributes.Value("gen_ai.token.type")
				if tokenType.AsString() == "input" {
					require.EqualValues(t, 7, p.Sum)
				} else {
					require.EqualValues(t, 3, p.Sum)
				}
			}
		default:
			t.Fatalf("unexpected metric %s", m.Name)
		}
	}
}

func TestInstrumentModelStream(t *testing.T) {
	t.Parallel()
	for _, name := range []string{"finish", "error", "creation error", "early stop", "cancellation"} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			tp, recorder := telemetryRecorder(t)
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			wantErr := errors.New("private error content")
			closed := false
			model := &mockLanguageModel{streamFunc: func(ctx context.Context, _ Call) (StreamResponse, error) {
				require.True(t, trace.SpanFromContext(ctx).IsRecording())
				if name == "creation error" {
					return nil, wantErr
				}
				return func(yield func(StreamPart) bool) {
					defer func() { closed = true }()
					if !yield(StreamPart{Type: StreamPartTypeTextDelta, Delta: "private response"}) {
						return
					}
					if name == "error" {
						yield(StreamPart{Type: StreamPartTypeError, Error: wantErr})
						return
					}
					if name == "cancellation" {
						cancel()
						return
					}
					yield(StreamPart{Type: StreamPartTypeFinish, FinishReason: FinishReasonStop, Usage: Usage{InputTokens: 2, OutputTokens: 4}})
				}, nil
			}}
			stream, err := InstrumentModel(model, tp, nil).Stream(ctx, Call{})
			if name == "creation error" {
				require.ErrorIs(t, err, wantErr)
			} else {
				require.NoError(t, err)
				require.Empty(t, recorder.Ended())
				for range stream {
					if name == "early stop" {
						break
					}
				}
				require.True(t, closed)
			}
			require.Len(t, recorder.Ended(), 1)
			span := recorder.Ended()[0]
			if name == "error" || name == "creation error" || name == "cancellation" {
				require.Equal(t, codes.Error, span.Status().Code)
				require.Empty(t, span.Status().Description)
			} else {
				require.Equal(t, codes.Unset, span.Status().Code)
			}
			if name == "finish" {
				require.EqualValues(t, 4, spanAttributes(span)["gen_ai.usage.output_tokens"].AsInt64())
			}
		})
	}
}

func TestAgentTelemetry(t *testing.T) {
	t.Parallel()
	for _, stream := range []bool{false, true} {
		t.Run(map[bool]string{false: "generate", true: "stream"}[stream], func(t *testing.T) {
			t.Parallel()
			tp, recorder := telemetryRecorder(t)
			calls := 0
			response := func(ctx context.Context) *Response {
				require.True(t, trace.SpanFromContext(ctx).IsRecording())
				calls++
				if calls == 1 {
					return &Response{Content: ResponseContent{ToolCallContent{ToolCallID: "call-1", ToolName: "lookup", Input: `{}`}}, FinishReason: FinishReasonToolCalls}
				}
				return &Response{Content: ResponseContent{TextContent{Text: "done"}}, FinishReason: FinishReasonStop}
			}
			model := &mockLanguageModel{
				generateFunc: func(ctx context.Context, _ Call) (*Response, error) { return response(ctx), nil },
				streamFunc: func(ctx context.Context, _ Call) (StreamResponse, error) {
					r := response(ctx)
					return func(yield func(StreamPart) bool) {
						for _, c := range r.Content.ToolCalls() {
							if !yield(StreamPart{Type: StreamPartTypeToolCall, ID: c.ToolCallID, ToolCallName: c.ToolName, ToolCallInput: c.Input}) {
								return
							}
						}
						yield(StreamPart{Type: StreamPartTypeFinish, FinishReason: r.FinishReason})
					}, nil
				},
			}
			tool := &mockTool{name: "lookup", parameters: map[string]any{}, executeFunc: func(ctx context.Context, _ ToolCall) (ToolResponse, error) {
				require.True(t, trace.SpanFromContext(ctx).IsRecording())
				return NewTextErrorResponse("private tool failure"), nil
			}}
			a := NewAgent(model, WithTracerProvider(tp), WithTools(tool))
			var err error
			if stream {
				_, err = a.Stream(t.Context(), AgentStreamCall{Prompt: "hi"})
			} else {
				_, err = a.Generate(t.Context(), AgentCall{Prompt: "hi"})
			}
			require.NoError(t, err)
			require.Equal(t, 2, calls)
			spans := recorder.Ended()
			require.Len(t, spans, 6) // Agent, two steps, two model calls, one tool.
			byID := make(map[trace.SpanID]sdktrace.ReadOnlySpan)
			for _, span := range spans {
				byID[span.SpanContext().SpanID()] = span
			}
			for _, span := range spans {
				switch span.Name() {
				case "chat mock-model", "execute_tool lookup":
					require.Equal(t, "agent.step", byID[span.Parent().SpanID()].Name())
				case "agent.step":
					require.Equal(t, "invoke_agent", byID[span.Parent().SpanID()].Name())
				}
				if span.Name() == "execute_tool lookup" {
					require.Equal(t, codes.Error, span.Status().Code)
				}
			}
		})
	}
}

// Object methods must call the provider's object implementation, rather than
// falling back to Generate/Stream and losing schema options or repair behavior.
type telemetryObjectModel struct {
	LanguageModel
	err error
}

func (m telemetryObjectModel) GenerateObject(ctx context.Context, call ObjectCall) (*ObjectResponse, error) {
	if m.err != nil {
		return nil, m.err
	}
	return &ObjectResponse{Object: call.SchemaName, Usage: Usage{InputTokens: 9, OutputTokens: 5}, FinishReason: FinishReasonStop}, nil
}

func (m telemetryObjectModel) StreamObject(ctx context.Context, call ObjectCall) (ObjectStreamResponse, error) {
	if m.err != nil {
		return nil, m.err
	}
	return func(yield func(ObjectStreamPart) bool) {
		if !yield(ObjectStreamPart{Type: ObjectStreamPartTypeObject, Object: call.SchemaName}) {
			return
		}
		yield(ObjectStreamPart{Type: ObjectStreamPartTypeFinish, Usage: Usage{InputTokens: 9, OutputTokens: 5}, FinishReason: FinishReasonStop})
	}, nil
}

func TestInstrumentModelObject(t *testing.T) {
	t.Parallel()
	for _, name := range []string{"generate", "stream", "generate error", "stream error"} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			tp, recorder := telemetryRecorder(t)
			model := telemetryObjectModel{LanguageModel: &mockLanguageModel{}}
			if name == "generate error" || name == "stream error" {
				model.err = errors.New("private provider response")
			}
			wrapped := InstrumentModel(model, tp, nil)
			var err error
			if name == "generate" || name == "generate error" {
				var result *ObjectResponse
				result, err = wrapped.GenerateObject(t.Context(), ObjectCall{SchemaName: "schema"})
				if err == nil {
					require.Equal(t, "schema", result.Object)
				}
			} else {
				var stream ObjectStreamResponse
				stream, err = wrapped.StreamObject(t.Context(), ObjectCall{SchemaName: "schema"})
				if err == nil {
					require.Empty(t, recorder.Ended())
					for part := range stream {
						if part.Type == ObjectStreamPartTypeObject {
							require.Equal(t, "schema", part.Object)
						}
					}
				}
			}
			require.ErrorIs(t, err, model.err)
			require.Len(t, recorder.Ended(), 1)
			span := recorder.Ended()[0]
			if model.err != nil {
				require.Equal(t, codes.Error, span.Status().Code)
			} else {
				require.EqualValues(t, 5, spanAttributes(span)["gen_ai.usage.output_tokens"].AsInt64())
			}
		})
	}
}

func TestAgentTelemetryErrors(t *testing.T) {
	t.Parallel()
	for _, stream := range []bool{false, true} {
		t.Run(map[bool]string{false: "generate", true: "stream"}[stream], func(t *testing.T) {
			t.Parallel()
			tp, recorder := telemetryRecorder(t)
			wantErr := errors.New("private provider error")
			model := &mockLanguageModel{
				generateFunc: func(context.Context, Call) (*Response, error) { return nil, wantErr },
				streamFunc:   func(context.Context, Call) (StreamResponse, error) { return nil, wantErr },
			}
			a := NewAgent(model, WithTracerProvider(tp), WithMaxRetries(0))
			var err error
			if stream {
				_, err = a.Stream(t.Context(), AgentStreamCall{Prompt: "hi"})
			} else {
				_, err = a.Generate(t.Context(), AgentCall{Prompt: "hi"})
			}
			require.ErrorIs(t, err, wantErr)
			require.Len(t, recorder.Ended(), 3)
			for _, span := range recorder.Ended() {
				require.Equal(t, codes.Error, span.Status().Code)
			}
		})
	}
}

func TestAgentTelemetryAuthRefresh(t *testing.T) {
	t.Parallel()
	tp, recorder := telemetryRecorder(t)
	first := &mockLanguageModel{generateFunc: func(context.Context, Call) (*Response, error) { return nil, &ProviderError{StatusCode: 401} }}
	second := &mockLanguageModel{}
	current := first
	a := NewAgent(first, WithTracerProvider(tp), WithMaxRetries(0))
	_, err := a.Generate(t.Context(), AgentCall{
		Prompt:        "hi",
		ModelProvider: func() LanguageModel { return current },
		OnAuthRefresh: func(context.Context, *ProviderError) error { current = second; return nil },
	})
	require.NoError(t, err)
	spans := recorder.Ended()
	require.Len(t, spans, 4)
	require.Equal(t, codes.Error, spans[0].Status().Code)
	require.Equal(t, codes.Unset, spans[1].Status().Code)
	require.Equal(t, spans[0].Parent(), spans[1].Parent())
}
