package openai

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/openai-go"

	"charm.land/fantasy"
)

// ToolCacheTypeFunc returns the prompt-cache hint to attach to the tool result
// at index i of msg, or "" for none. Providers without prompt caching pass nil.
//
// See anthropic.ToolResultCacheType for the Anthropic-style implementation.
type ToolCacheTypeFunc func(msg fantasy.Message, i int) string

// ToolMessages converts a tool-role prompt message into the chat-completions
// messages answering the assistant's tool_calls.
//
// Media results come back in deferred rather than messages: they convert to a
// user message, which cannot sit inside the run of tool messages. Hand them to
// a ToolRunBuffer.
func ToolMessages(msg fantasy.Message, cacheType ToolCacheTypeFunc) (messages, deferred []openai.ChatCompletionMessageParamUnion, warnings []fantasy.CallWarning) {
	warn := func(format string, args ...any) {
		warnings = append(warnings, fantasy.CallWarning{
			Type:    fantasy.CallWarningTypeOther,
			Message: fmt.Sprintf(format, args...),
		})
	}

	for i, c := range msg.Content {
		if c.GetType() != fantasy.ContentTypeToolResult {
			warn("tool message can only have tool result content")
			continue
		}
		part, ok := fantasy.AsContentType[fantasy.ToolResultPart](c)
		if !ok {
			warn("tool message result part does not have the right type")
			continue
		}

		var hint string
		if cacheType != nil {
			hint = cacheType(msg, i)
		}

		switch part.Output.GetType() {
		case fantasy.ToolResultContentTypeText:
			output, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentText](part.Output)
			if !ok {
				warn("tool result output does not have the right type")
				continue
			}
			messages = append(messages, textToolMessage(output.Text, part.ToolCallID, hint))
		case fantasy.ToolResultContentTypeError:
			output, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentError](part.Output)
			if !ok {
				warn("tool result output does not have the right type")
				continue
			}
			messages = append(messages, textToolMessage(output.Error.Error(), part.ToolCallID, hint))
		case fantasy.ToolResultContentTypeMedia:
			output, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentMedia](part.Output)
			if !ok {
				warn("tool result output does not have the right type")
				continue
			}
			tool, media, mediaWarnings := ToolResultMediaMessages(output, part.ToolCallID)
			tagToolCacheControl(tool.OfTool, hint)
			messages = append(messages, tool)
			deferred = append(deferred, media...)
			warnings = append(warnings, mediaWarnings...)
		default:
			// Skipping silently would leave the assistant's tool_call
			// unanswered, which strict backends reject.
			warn("tool result output type %q not supported", part.Output.GetType())
		}
	}
	return messages, deferred, warnings
}

// textToolMessage builds a tool message, tagged for prompt caching when the
// provider asked for a hint.
func textToolMessage(text, toolCallID, cacheType string) openai.ChatCompletionMessageParamUnion {
	msg := openai.ToolMessage(text, toolCallID)
	tagToolCacheControl(msg.OfTool, cacheType)
	return msg
}

// tagToolCacheControl marks a tool message for Anthropic-style prompt caching.
//
// It takes the tool param rather than the message union that wraps it: the
// union is not the value that gets marshalled, so setting the extra field
// there compiles and silently drops the hint.
func tagToolCacheControl(msg *openai.ChatCompletionToolMessageParam, cacheType string) {
	if msg == nil || cacheType == "" {
		return
	}
	msg.SetExtraFields(map[string]any{
		"cache_control": map[string]string{"type": cacheType},
	})
}

// ToolResultMediaMessages maps a tool-result media output to the chat
// completions messages that convey it: a text tool message that keeps the
// tool_call/tool_result pairing valid, plus the user messages holding the image
// or audio itself, which a tool message cannot carry.
//
// The two are returned separately because the tool message must stay inside the
// run of tool messages while the media has to land after it. Unsupported media
// types return only the tool message, plus a warning.
func ToolResultMediaMessages(output fantasy.ToolResultOutputContentMedia, toolCallID string) (openai.ChatCompletionMessageParamUnion, []openai.ChatCompletionMessageParamUnion, []fantasy.CallWarning) {
	mediaPart, warning, emit := toolResultMediaUserPart(output)

	placeholder := output.Text
	if placeholder == "" && emit {
		placeholder = fmt.Sprintf("The tool returned %s content; see the following user message.", output.MediaType)
	} else if placeholder == "" {
		placeholder = fmt.Sprintf("The tool returned %s content, which cannot be displayed.", output.MediaType)
	}
	toolMessage := openai.ToolMessage(placeholder, toolCallID)

	if warning != nil {
		return toolMessage, nil, []fantasy.CallWarning{*warning}
	}
	if !emit {
		return toolMessage, nil, nil
	}
	return toolMessage, []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage([]openai.ChatCompletionContentPartUnionParam{mediaPart}),
	}, nil
}

// toolResultMediaUserPart maps a tool-result media output to a user content
// part, reporting whether the caller should emit it.
func toolResultMediaUserPart(output fantasy.ToolResultOutputContentMedia) (openai.ChatCompletionContentPartUnionParam, *fantasy.CallWarning, bool) {
	switch {
	case strings.HasPrefix(output.MediaType, "image/"):
		data := "data:" + output.MediaType + ";base64," + output.Data
		imageBlock := openai.ChatCompletionContentPartImageParam{
			ImageURL: openai.ChatCompletionContentPartImageImageURLParam{URL: data},
		}
		return openai.ChatCompletionContentPartUnionParam{OfImageURL: &imageBlock}, nil, true
	case output.MediaType == "audio/wav", output.MediaType == "audio/mpeg", output.MediaType == "audio/mp3":
		format := "wav"
		if output.MediaType != "audio/wav" {
			format = "mp3"
		}
		audioBlock := openai.ChatCompletionContentPartInputAudioParam{
			InputAudio: openai.ChatCompletionContentPartInputAudioInputAudioParam{
				Data:   output.Data,
				Format: format,
			},
		}
		return openai.ChatCompletionContentPartUnionParam{OfInputAudio: &audioBlock}, nil, true
	default:
		return openai.ChatCompletionContentPartUnionParam{}, &fantasy.CallWarning{
			Type:    fantasy.CallWarningTypeOther,
			Message: fmt.Sprintf("tool result media type %s not supported, sending text placeholder only", output.MediaType),
		}, false
	}
}

// ToolRunBuffer holds synthetic user messages carrying tool-result media back
// until the current run of tool messages ends.
//
// Chat-completions validators require every tool message answering an
// assistant's tool_calls to follow that assistant message with only other tool
// messages in between, so emitting media where it is produced splits the run
// and every tool_call_id after the split reads as unanswered.
type ToolRunBuffer struct {
	deferred []openai.ChatCompletionMessageParamUnion
}

// Role reports that a prompt message of the given role is about to be
// converted, flushing any deferred media when that role ends a tool run.
// It returns messages with the flush applied.
func (b *ToolRunBuffer) Role(role fantasy.MessageRole, messages []openai.ChatCompletionMessageParamUnion) []openai.ChatCompletionMessageParamUnion {
	if role == fantasy.MessageRoleTool {
		return messages
	}
	return b.flush(messages)
}

// Defer holds media messages until the current tool run ends.
func (b *ToolRunBuffer) Defer(messages ...openai.ChatCompletionMessageParamUnion) {
	b.deferred = append(b.deferred, messages...)
}

// Close flushes anything still deferred when the prompt ends on a tool run.
func (b *ToolRunBuffer) Close(messages []openai.ChatCompletionMessageParamUnion) []openai.ChatCompletionMessageParamUnion {
	return b.flush(messages)
}

func (b *ToolRunBuffer) flush(messages []openai.ChatCompletionMessageParamUnion) []openai.ChatCompletionMessageParamUnion {
	if len(b.deferred) == 0 {
		return messages
	}
	messages = append(messages, b.deferred...)
	b.deferred = nil
	return messages
}
