package fantasy

import (
	_ "embed"
	"strings"
)

//go:embed version.txt
var version string

// Version is the SDK version, read from version.txt.
var Version = strings.TrimSpace(version)
