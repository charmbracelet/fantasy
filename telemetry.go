package fantasy

import (
	"context"
	"fmt"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/metric"
	semconv "go.opentelemetry.io/otel/semconv/v1.39.0"
	"go.opentelemetry.io/otel/trace"
)

const telemetryScope = "charm.land/fantasy"

type agentObservation struct {
	telemetry *telemetry
	span      trace.Span
	step      trace.Span
}

func (t *telemetry) startAgent(ctx context.Context) (context.Context, *agentObservation) {
	if t == nil || t.tracer == nil {
		return ctx, nil
	}
	ctx, span := t.start(ctx, "invoke_agent", trace.SpanKindInternal, semconv.GenAIOperationNameInvokeAgent)
	return ctx, &agentObservation{telemetry: t, span: span}
}

func (o *agentObservation) startStep(ctx context.Context) context.Context {
	if o == nil {
		return ctx
	}
	ctx, o.step = o.telemetry.start(ctx, "agent.step", trace.SpanKindInternal)
	return ctx
}

func (o *agentObservation) endStep(err error) {
	if o == nil {
		return
	}
	endTelemetrySpan(o.step, err)
	o.step = nil
}

func (o *agentObservation) end(err error) {
	if o == nil {
		return
	}
	o.endStep(err)
	endTelemetrySpan(o.span, err)
}

func (t *telemetry) startTool(ctx context.Context, call ToolCallContent) (context.Context, trace.Span) {
	if t == nil || t.tracer == nil {
		return ctx, nil
	}
	return t.start(ctx, "execute_tool "+call.ToolName, trace.SpanKindInternal,
		semconv.GenAIOperationNameExecuteTool, semconv.GenAIToolName(call.ToolName), semconv.GenAIToolCallID(call.ToolCallID))
}

// WithTracerProvider enables agent, step, model-call, and tool-call spans.
// A nil provider disables tracing. Fantasy does not use the global provider or
// record prompts, generated content, tool arguments, or tool results.
func WithTracerProvider(provider trace.TracerProvider) AgentOption {
	return func(s *agentSettings) { s.tracerProvider = provider }
}

// WithMeterProvider enables model-call duration and token-usage histograms.
// A nil provider disables metrics. The caller owns provider shutdown and export.
func WithMeterProvider(provider metric.MeterProvider) AgentOption {
	return func(s *agentSettings) { s.meterProvider = provider }
}

type telemetry struct {
	tracer   trace.Tracer
	duration metric.Float64Histogram
	tokens   metric.Int64Histogram
}

func newTelemetry(tp trace.TracerProvider, mp metric.MeterProvider) *telemetry {
	if tp == nil && mp == nil {
		return nil
	}
	t := &telemetry{}
	if tp != nil {
		t.tracer = tp.Tracer(telemetryScope, trace.WithSchemaURL(semconv.SchemaURL))
	}
	if mp != nil {
		meter := mp.Meter(telemetryScope, metric.WithSchemaURL(semconv.SchemaURL))
		var err error
		t.duration, err = meter.Float64Histogram("gen_ai.client.operation.duration", metric.WithUnit("s"),
			metric.WithExplicitBucketBoundaries(0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92))
		if err != nil {
			otel.Handle(err)
		}
		t.tokens, err = meter.Int64Histogram("gen_ai.client.token.usage", metric.WithUnit("{token}"),
			metric.WithExplicitBucketBoundaries(1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864))
		if err != nil {
			otel.Handle(err)
		}
	}
	return t
}

func (t *telemetry) start(ctx context.Context, name string, kind trace.SpanKind, attrs ...attribute.KeyValue) (context.Context, trace.Span) {
	if t == nil || t.tracer == nil {
		return ctx, nil
	}
	return t.tracer.Start(ctx, name, trace.WithSpanKind(kind), trace.WithAttributes(attrs...))
}

func endTelemetrySpan(span trace.Span, err error) {
	if span == nil {
		return
	}
	if err != nil {
		// Error messages can contain request or response content. Record only the
		// error type, leaving content capture to application instrumentation.
		span.SetAttributes(semconv.ErrorTypeKey.String(fmt.Sprintf("%T", err)))
		span.SetStatus(codes.Error, "")
	}
	span.End()
}

func (t *telemetry) instrument(model LanguageModel) LanguageModel {
	if t == nil {
		return model
	}
	if wrapped, ok := model.(*instrumentedModel); ok && wrapped.telemetry == t {
		return model
	}
	return &instrumentedModel{LanguageModel: model, telemetry: t}
}

// InstrumentModel adds GenAI spans and metrics to all four LanguageModel call
// methods, including structured output. Nil providers disable their signal;
// with both nil, the original model is returned. Stream spans end when the
// returned iterator finishes or its consumer stops early. Callers must consume
// the iterator to release the stream and end its span.
// Do not also configure telemetry on an agent using this wrapped model.
func InstrumentModel(model LanguageModel, tp trace.TracerProvider, mp metric.MeterProvider) LanguageModel {
	return newTelemetry(tp, mp).instrument(model)
}

type instrumentedModel struct {
	LanguageModel
	telemetry *telemetry
}

type modelObservation struct {
	telemetry *telemetry
	ctx       context.Context
	span      trace.Span
	start     time.Time
	attrs     []attribute.KeyValue
	usage     *Usage
	finish    FinishReason
	err       error
}

func (m *instrumentedModel) observe(ctx context.Context) *modelObservation {
	provider := m.Provider()
	switch provider {
	case "bedrock":
		provider = "aws.bedrock"
	case "azure":
		provider = "azure.ai.openai"
	case "google":
		provider = "gcp.gen_ai"
	}
	attrs := []attribute.KeyValue{semconv.GenAIOperationNameChat, semconv.GenAIProviderNameKey.String(provider), semconv.GenAIRequestModel(m.Model())}
	ctx, span := m.telemetry.start(ctx, "chat "+m.Model(), trace.SpanKindClient, attrs...)
	return &modelObservation{telemetry: m.telemetry, ctx: ctx, span: span, start: time.Now(), attrs: attrs}
}

func (o *modelObservation) end() {
	if o.err == nil {
		o.err = o.ctx.Err()
	}
	if o.span != nil {
		if o.finish != "" {
			o.span.SetAttributes(semconv.GenAIResponseFinishReasons(string(o.finish)))
		}
		if o.usage != nil {
			o.span.SetAttributes(semconv.GenAIUsageInputTokensKey.Int64(o.usage.InputTokens), semconv.GenAIUsageOutputTokensKey.Int64(o.usage.OutputTokens))
		}
	}
	if o.telemetry.duration != nil {
		attrs := o.attrs
		if o.err != nil {
			attrs = append(attrs, semconv.ErrorTypeKey.String(fmt.Sprintf("%T", o.err)))
		}
		o.telemetry.duration.Record(o.ctx, time.Since(o.start).Seconds(), metric.WithAttributes(attrs...))
	}
	if o.telemetry.tokens != nil && o.usage != nil {
		if o.usage.InputTokens > 0 {
			o.telemetry.tokens.Record(o.ctx, o.usage.InputTokens, metric.WithAttributes(append(o.attrs, semconv.GenAITokenTypeInput)...))
		}
		if o.usage.OutputTokens > 0 {
			o.telemetry.tokens.Record(o.ctx, o.usage.OutputTokens, metric.WithAttributes(append(o.attrs, semconv.GenAITokenTypeOutput)...))
		}
	}
	endTelemetrySpan(o.span, o.err)
}

func (m *instrumentedModel) Generate(ctx context.Context, call Call) (*Response, error) {
	o := m.observe(ctx)
	defer o.end()
	response, err := m.LanguageModel.Generate(o.ctx, call)
	o.err = err
	if response != nil {
		o.usage, o.finish = &response.Usage, response.FinishReason
	}
	return response, err
}

func (m *instrumentedModel) Stream(ctx context.Context, call Call) (StreamResponse, error) {
	o := m.observe(ctx)
	stream, err := m.LanguageModel.Stream(o.ctx, call)
	if err != nil {
		o.err = err
		o.end()
		return stream, err
	}
	return func(yield func(StreamPart) bool) {
		defer o.end()
		for part := range stream {
			if part.Type == StreamPartTypeFinish {
				usage := part.Usage
				o.usage, o.finish = &usage, part.FinishReason
			}
			if part.Type == StreamPartTypeError {
				o.err = part.Error
			}
			if !yield(part) {
				return
			}
		}
	}, nil
}

func (m *instrumentedModel) GenerateObject(ctx context.Context, call ObjectCall) (*ObjectResponse, error) {
	o := m.observe(ctx)
	defer o.end()
	response, err := m.LanguageModel.GenerateObject(o.ctx, call)
	o.err = err
	if response != nil {
		o.usage, o.finish = &response.Usage, response.FinishReason
	}
	return response, err
}

func (m *instrumentedModel) StreamObject(ctx context.Context, call ObjectCall) (ObjectStreamResponse, error) {
	o := m.observe(ctx)
	stream, err := m.LanguageModel.StreamObject(o.ctx, call)
	if err != nil {
		o.err = err
		o.end()
		return stream, err
	}
	return func(yield func(ObjectStreamPart) bool) {
		defer o.end()
		for part := range stream {
			if part.Type == ObjectStreamPartTypeFinish {
				usage := part.Usage
				o.usage, o.finish = &usage, part.FinishReason
			}
			if part.Type == ObjectStreamPartTypeError {
				o.err = part.Error
			}
			if !yield(part) {
				return
			}
		}
	}, nil
}
