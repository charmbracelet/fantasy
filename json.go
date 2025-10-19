package fantasy

import (
	"encoding/json"
	"fmt"
)

// toolResultOutputJSON is a helper type for JSON serialization of ToolResultOutputContent
type toolResultOutputJSON struct {
	Type string          `json:"type"`
	Data json.RawMessage `json:"data"`
}

// messagePartJSON is a helper type for JSON serialization of MessagePart
type messagePartJSON struct {
	Type string          `json:"type"`
	Data json.RawMessage `json:"data"`
}

// MarshalJSON implements custom JSON marshaling for Message
func (m Message) MarshalJSON() ([]byte, error) {
	type Alias Message

	// Convert MessagePart slice to a JSON-friendly format
	contentJSON := make([]messagePartJSON, len(m.Content))
	for i, part := range m.Content {
		partData, err := json.Marshal(part)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal message part %d: %w", i, err)
		}
		contentJSON[i] = messagePartJSON{
			Type: string(part.GetType()),
			Data: partData,
		}
	}

	return json.Marshal(&struct {
		Role            MessageRole       `json:"role"`
		Content         []messagePartJSON `json:"content"`
		ProviderOptions ProviderOptions   `json:"provider_options"`
	}{
		Role:            m.Role,
		Content:         contentJSON,
		ProviderOptions: m.ProviderOptions,
	})
}

// UnmarshalJSON implements custom JSON unmarshaling for Message
func (m *Message) UnmarshalJSON(data []byte) error {
	aux := &struct {
		Role            MessageRole       `json:"role"`
		Content         []messagePartJSON `json:"content"`
		ProviderOptions ProviderOptions   `json:"provider_options"`
	}{}

	if err := json.Unmarshal(data, aux); err != nil {
		return err
	}

	m.Role = aux.Role
	m.ProviderOptions = aux.ProviderOptions
	m.Content = make([]MessagePart, len(aux.Content))

	for i, partJSON := range aux.Content {
		var part MessagePart
		var err error

		switch ContentType(partJSON.Type) {
		case ContentTypeText:
			var tp TextPart
			err = json.Unmarshal(partJSON.Data, &tp)
			part = tp
		case ContentTypeReasoning:
			var rp ReasoningPart
			err = json.Unmarshal(partJSON.Data, &rp)
			part = rp
		case ContentTypeFile:
			var fp FilePart
			err = json.Unmarshal(partJSON.Data, &fp)
			part = fp
		case ContentTypeToolCall:
			var tcp ToolCallPart
			err = json.Unmarshal(partJSON.Data, &tcp)
			part = tcp
		case ContentTypeToolResult:
			var trp ToolResultPart
			err = json.Unmarshal(partJSON.Data, &trp)
			part = trp
		default:
			return fmt.Errorf("unknown message part type: %s", partJSON.Type)
		}

		if err != nil {
			return fmt.Errorf("failed to unmarshal message part %d of type %s: %w", i, partJSON.Type, err)
		}

		m.Content[i] = part
	}

	return nil
}

// MarshalJSON implements custom JSON marshaling for ToolResultPart
func (t ToolResultPart) MarshalJSON() ([]byte, error) {
	outputData, err := json.Marshal(t.Output)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal tool result output: %w", err)
	}

	return json.Marshal(&struct {
		ToolCallID      string               `json:"tool_call_id"`
		Output          toolResultOutputJSON `json:"output"`
		ProviderOptions ProviderOptions      `json:"provider_options"`
	}{
		ToolCallID: t.ToolCallID,
		Output: toolResultOutputJSON{
			Type: string(t.Output.GetType()),
			Data: outputData,
		},
		ProviderOptions: t.ProviderOptions,
	})
}

// UnmarshalJSON implements custom JSON unmarshaling for ToolResultPart
func (t *ToolResultPart) UnmarshalJSON(data []byte) error {
	aux := &struct {
		ToolCallID      string               `json:"tool_call_id"`
		Output          toolResultOutputJSON `json:"output"`
		ProviderOptions ProviderOptions      `json:"provider_options"`
	}{}

	if err := json.Unmarshal(data, aux); err != nil {
		return err
	}

	t.ToolCallID = aux.ToolCallID
	t.ProviderOptions = aux.ProviderOptions

	var output ToolResultOutputContent
	var err error

	switch ToolResultContentType(aux.Output.Type) {
	case ToolResultContentTypeText:
		var textOutput ToolResultOutputContentText
		err = json.Unmarshal(aux.Output.Data, &textOutput)
		output = textOutput
	case ToolResultContentTypeError:
		var errorOutput ToolResultOutputContentError
		err = json.Unmarshal(aux.Output.Data, &errorOutput)
		output = errorOutput
	case ToolResultContentTypeMedia:
		var mediaOutput ToolResultOutputContentMedia
		err = json.Unmarshal(aux.Output.Data, &mediaOutput)
		output = mediaOutput
	default:
		return fmt.Errorf("unknown tool result output type: %s", aux.Output.Type)
	}

	if err != nil {
		return fmt.Errorf("failed to unmarshal tool result output: %w", err)
	}

	t.Output = output
	return nil
}

// MarshalJSON implements custom JSON marshaling for ToolResultOutputContentError
func (t ToolResultOutputContentError) MarshalJSON() ([]byte, error) {
	var errorStr string
	if t.Error != nil {
		errorStr = t.Error.Error()
	}

	return json.Marshal(&struct {
		Error string `json:"error"`
	}{
		Error: errorStr,
	})
}

// UnmarshalJSON implements custom JSON unmarshaling for ToolResultOutputContentError
func (t *ToolResultOutputContentError) UnmarshalJSON(data []byte) error {
	aux := &struct {
		Error string `json:"error"`
	}{}

	if err := json.Unmarshal(data, aux); err != nil {
		return err
	}

	if aux.Error != "" {
		t.Error = fmt.Errorf("%s", aux.Error)
	}

	return nil
}
