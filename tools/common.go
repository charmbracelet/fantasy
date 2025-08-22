package tools

import "errors"

type Permission struct {
	ToolCallID  string
	ToolName    string
	Path        string
	Action      string
	Description string
	Params      any
}
type PermissionAsk = func(ask Permission) bool

var ErrorPermissionDenied = errors.New("permission denied")
