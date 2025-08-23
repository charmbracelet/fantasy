package tools

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"
)

type (
	sessionIDContextKey string
	messageIDContextKey string
)

const (
	MaxOutputLength  = 30000
	MaxReadSize      = 250 * 1024
	DefaultReadLimit = 2000
	MaxLineLength    = 2000

	SessionIDContextKey sessionIDContextKey = "session_id"
	MessageIDContextKey messageIDContextKey = "message_id"
)

func truncateOutput(content string) string {
	if len(content) <= MaxOutputLength {
		return content
	}

	halfLength := MaxOutputLength / 2
	start := content[:halfLength]
	end := content[len(content)-halfLength:]

	truncatedLinesCount := countLines(content[halfLength : len(content)-halfLength])
	return fmt.Sprintf("%s\n\n... [%d lines truncated] ...\n\n%s", start, truncatedLinesCount, end)
}

func countLines(s string) int {
	if s == "" {
		return 0
	}
	return len(strings.Split(s, "\n"))
}

func GetContextValues(ctx context.Context) (string, string) {
	sessionID := ctx.Value(SessionIDContextKey)
	messageID := ctx.Value(MessageIDContextKey)
	if sessionID == nil {
		return "", ""
	}
	if messageID == nil {
		return sessionID.(string), ""
	}
	return sessionID.(string), messageID.(string)
}

// File record to track when files were read/written
type fileRecord struct {
	path      string
	readTime  time.Time
	writeTime time.Time
}

var (
	fileRecords     = make(map[string]fileRecord)
	fileRecordMutex sync.RWMutex
)

func recordFileRead(path string) {
	fileRecordMutex.Lock()
	defer fileRecordMutex.Unlock()

	record, exists := fileRecords[path]
	if !exists {
		record = fileRecord{path: path}
	}
	record.readTime = time.Now()
	fileRecords[path] = record
}

func getLastReadTime(path string) time.Time {
	fileRecordMutex.RLock()
	defer fileRecordMutex.RUnlock()

	record, exists := fileRecords[path]
	if !exists {
		return time.Time{}
	}
	return record.readTime
}

func recordFileWrite(path string) {
	fileRecordMutex.Lock()
	defer fileRecordMutex.Unlock()

	record, exists := fileRecords[path]
	if !exists {
		record = fileRecord{path: path}
	}
	record.writeTime = time.Now()
	fileRecords[path] = record
}
