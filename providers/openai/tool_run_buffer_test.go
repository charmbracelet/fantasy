package openai

import (
	"testing"

	openai "github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/require"

	"charm.land/fantasy"
)

// ToolRunBuffer owns one rule: a run of tool messages answering an assistant's
// tool_calls must stay contiguous, so synthetic media user messages land after
// the run ends. All four chat-completions converters route through it, which
// is why it is tested here rather than four times over.

// label renders a message as a short role tag so the tests can assert on
// ordering without caring about content.
func label(m openai.ChatCompletionMessageParamUnion) string {
	switch {
	case m.OfTool != nil:
		return "tool:" + m.OfTool.ToolCallID
	case m.OfAssistant != nil:
		return "assistant"
	case m.OfUser != nil:
		return "user"
	default:
		return "other"
	}
}

func labels(msgs []openai.ChatCompletionMessageParamUnion) []string {
	out := make([]string, 0, len(msgs))
	for _, m := range msgs {
		out = append(out, label(m))
	}
	return out
}

// step is one prompt message being converted: its role, what the converter
// emits inline, and what it defers.
type step struct {
	role     fantasy.MessageRole
	emit     []openai.ChatCompletionMessageParamUnion
	deferred []openai.ChatCompletionMessageParamUnion
}

// run drives the buffer the way every converter's loop does.
func run(steps []step) []openai.ChatCompletionMessageParamUnion {
	var buf ToolRunBuffer
	var messages []openai.ChatCompletionMessageParamUnion
	for _, s := range steps {
		messages = buf.Role(s.role, messages)
		messages = append(messages, s.emit...)
		buf.Defer(s.deferred...)
	}
	return buf.Close(messages)
}

func assistant(text string) []openai.ChatCompletionMessageParamUnion {
	return []openai.ChatCompletionMessageParamUnion{openai.AssistantMessage(text)}
}

func tool(id string) []openai.ChatCompletionMessageParamUnion {
	return []openai.ChatCompletionMessageParamUnion{openai.ToolMessage("result", id)}
}

func user(text string) []openai.ChatCompletionMessageParamUnion {
	return []openai.ChatCompletionMessageParamUnion{openai.UserMessage(text)}
}

func TestToolRunBuffer(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		steps []step
		want  []string
	}{
		{
			// Nothing deferred: the buffer must not reorder anything.
			name: "passthrough",
			steps: []step{
				{role: fantasy.MessageRoleUser, emit: user("hi")},
				{role: fantasy.MessageRoleAssistant, emit: assistant("hello")},
			},
			want: []string{"user", "assistant"},
		},
		{
			// A parallel batch: both tool messages stay contiguous and the
			// media lands after the run, before the next user turn.
			name: "parallel batch keeps tool run contiguous",
			steps: []step{
				{role: fantasy.MessageRoleAssistant, emit: assistant("on it")},
				{role: fantasy.MessageRoleTool, emit: tool("c1"), deferred: user("media1")},
				{role: fantasy.MessageRoleTool, emit: tool("c2"), deferred: user("media2")},
				{role: fantasy.MessageRoleUser, emit: user("next")},
			},
			want: []string{"assistant", "tool:c1", "tool:c2", "user", "user", "user"},
		},
		{
			// A text result in the same batch still emits in place.
			name: "mixed batch keeps text inline",
			steps: []step{
				{role: fantasy.MessageRoleAssistant, emit: assistant("on it")},
				{role: fantasy.MessageRoleTool, emit: tool("img"), deferred: user("media")},
				{role: fantasy.MessageRoleTool, emit: tool("txt")},
				{role: fantasy.MessageRoleUser, emit: user("next")},
			},
			want: []string{"assistant", "tool:img", "tool:txt", "user", "user"},
		},
		{
			// The run ends at an assistant message too, not just a user one.
			name: "flush before assistant",
			steps: []step{
				{role: fantasy.MessageRoleAssistant, emit: assistant("on it")},
				{role: fantasy.MessageRoleTool, emit: tool("c1"), deferred: user("media")},
				{role: fantasy.MessageRoleAssistant, emit: assistant("done")},
			},
			want: []string{"assistant", "tool:c1", "user", "assistant"},
		},
		{
			// When the prompt ends on a tool run, Close flushes rather than
			// dropping the media.
			name: "flush at end",
			steps: []step{
				{role: fantasy.MessageRoleAssistant, emit: assistant("on it")},
				{role: fantasy.MessageRoleTool, emit: tool("c1"), deferred: user("media")},
			},
			want: []string{"assistant", "tool:c1", "user"},
		},
		{
			// Two assistant turns. The second turn's tool message must not be
			// hoisted ahead of the first turn's trailing messages: this is the
			// case a blind post-pass over the finished slice gets wrong.
			name: "two turns keep their own runs",
			steps: []step{
				{role: fantasy.MessageRoleAssistant, emit: assistant("first")},
				{role: fantasy.MessageRoleTool, emit: tool("c1"), deferred: user("media")},
				{role: fantasy.MessageRoleAssistant, emit: assistant("second")},
				{role: fantasy.MessageRoleTool, emit: tool("c2")},
				{role: fantasy.MessageRoleUser, emit: user("final")},
			},
			want: []string{"assistant", "tool:c1", "user", "assistant", "tool:c2", "user"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			require.Equal(t, tt.want, labels(run(tt.steps)))
		})
	}
}
