package replaytest

import (
	"encoding/json"
	"regexp"

	"charm.land/fantasy"
)

// UsageRecord captures the token usage carried by a stream part or
// response. Zero fields are omitted so goldens show exactly which numbers
// the provider actually reported.
type UsageRecord struct {
	InputTokens         int64 `json:"input_tokens,omitempty"`
	OutputTokens        int64 `json:"output_tokens,omitempty"`
	TotalTokens         int64 `json:"total_tokens,omitempty"`
	ReasoningTokens     int64 `json:"reasoning_tokens,omitempty"`
	CacheCreationTokens int64 `json:"cache_creation_tokens,omitempty"`
	CacheReadTokens     int64 `json:"cache_read_tokens,omitempty"`
}

// PartRecord is a normalized view of one fantasy.StreamPart (or one response
// content part) for golden comparison. Only the fields meaningful for the
// part type are set; the rest are omitted from the JSON:
//
//   - delta carries streamed text, reasoning, and tool-input fragments, and
//     the full text of text and reasoning content parts
//   - name and input carry tool call names and arguments; for tool-result
//     content parts, input carries the text result
//   - reason carries the finish reason on finish parts
//   - metadata carries provider metadata (and source fields for source parts)
//   - warnings and error carry call warnings and provider errors
type PartRecord struct {
	Type     string                     `json:"type"`
	ID       string                     `json:"id,omitempty"`
	Delta    string                     `json:"delta,omitempty"`
	Name     string                     `json:"name,omitempty"`
	Input    string                     `json:"input,omitempty"`
	Reason   string                     `json:"reason,omitempty"`
	Usage    *UsageRecord               `json:"usage,omitempty"`
	Metadata map[string]json.RawMessage `json:"metadata,omitempty"`
	Warnings []string                   `json:"warnings,omitempty"`
	Error    string                     `json:"error,omitempty"`
}

// Collect drains the stream and returns a PartRecord for every StreamPart,
// in emission order.
func Collect(stream fantasy.StreamResponse) []PartRecord {
	records := []PartRecord{}
	for part := range stream {
		records = append(records, recordFromStreamPart(part))
	}
	return records
}

// localAddrPattern replaces the httptest listener port, which differs
// between runs, so error text stays golden-stable.
var localAddrPattern = regexp.MustCompile(`127\.0\.0\.1:\d+`)

func recordFromStreamPart(part fantasy.StreamPart) PartRecord {
	record := PartRecord{
		Type:     string(part.Type),
		ID:       part.ID,
		Metadata: metadataRecords(part.ProviderMetadata),
	}
	switch part.Type {
	case fantasy.StreamPartTypeTextDelta, fantasy.StreamPartTypeReasoningDelta:
		record.Delta = part.Delta
	case fantasy.StreamPartTypeToolInputDelta:
		record.Delta = part.Delta
		record.Input = part.ToolCallInput
	case fantasy.StreamPartTypeToolInputStart, fantasy.StreamPartTypeToolCall:
		record.Name = part.ToolCallName
		record.Input = part.ToolCallInput
	case fantasy.StreamPartTypeFinish:
		record.Reason = string(part.FinishReason)
		record.Usage = usageRecord(part.Usage)
	case fantasy.StreamPartTypeSource:
		record.Metadata = map[string]json.RawMessage{
			"source_type": rawString(string(part.SourceType)),
			"url":         rawString(part.URL),
			"title":       rawString(part.Title),
		}
	case fantasy.StreamPartTypeWarnings:
		for _, warning := range part.Warnings {
			if warning.Message != "" {
				record.Warnings = append(record.Warnings, warning.Message)
				continue
			}
			record.Warnings = append(record.Warnings, string(warning.Type))
		}
	case fantasy.StreamPartTypeError:
		if part.Error != nil {
			record.Error = localAddrPattern.ReplaceAllString(part.Error.Error(), "127.0.0.1:0")
		}
	}
	return record
}

// recordsFromContent converts response content parts (agent steps, Generate
// responses) into PartRecords using the same record format as stream parts.
func recordsFromContent(content fantasy.ResponseContent) []PartRecord {
	records := []PartRecord{}
	for _, c := range content {
		switch c.GetType() {
		case fantasy.ContentTypeText:
			if text, ok := fantasy.AsContentType[fantasy.TextContent](c); ok {
				records = append(records, PartRecord{
					Type:     string(fantasy.ContentTypeText),
					Delta:    text.Text,
					Metadata: metadataRecords(text.ProviderMetadata),
				})
			}
		case fantasy.ContentTypeReasoning:
			if reasoning, ok := fantasy.AsContentType[fantasy.ReasoningContent](c); ok {
				records = append(records, PartRecord{
					Type:     string(fantasy.ContentTypeReasoning),
					Delta:    reasoning.Text,
					Metadata: metadataRecords(reasoning.ProviderMetadata),
				})
			}
		case fantasy.ContentTypeToolCall:
			if call, ok := fantasy.AsContentType[fantasy.ToolCallContent](c); ok {
				records = append(records, PartRecord{
					Type:     string(fantasy.ContentTypeToolCall),
					ID:       call.ToolCallID,
					Name:     call.ToolName,
					Input:    call.Input,
					Metadata: metadataRecords(call.ProviderMetadata),
				})
			}
		case fantasy.ContentTypeToolResult:
			if result, ok := fantasy.AsContentType[fantasy.ToolResultContent](c); ok {
				record := PartRecord{
					Type:     string(fantasy.ContentTypeToolResult),
					ID:       result.ToolCallID,
					Name:     result.ToolName,
					Metadata: metadataRecords(result.ProviderMetadata),
				}
				if text, ok := result.Result.(fantasy.ToolResultOutputContentText); ok {
					record.Input = text.Text
				}
				records = append(records, record)
			}
		}
	}
	return records
}

func rawString(s string) json.RawMessage {
	raw, err := json.Marshal(s)
	if err != nil {
		return json.RawMessage(`""`)
	}
	return raw
}

func usageRecord(usage fantasy.Usage) *UsageRecord {
	if usage == (fantasy.Usage{}) {
		return nil
	}
	return &UsageRecord{
		InputTokens:         usage.InputTokens,
		OutputTokens:        usage.OutputTokens,
		TotalTokens:         usage.TotalTokens,
		ReasoningTokens:     usage.ReasoningTokens,
		CacheCreationTokens: usage.CacheCreationTokens,
		CacheReadTokens:     usage.CacheReadTokens,
	}
}

func metadataRecords(metadata fantasy.ProviderMetadata) map[string]json.RawMessage {
	if len(metadata) == 0 {
		return nil
	}
	out := make(map[string]json.RawMessage, len(metadata))
	for key, value := range metadata {
		raw, err := json.Marshal(value)
		if err != nil {
			continue
		}
		out[key] = raw
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
