package providertests

import (
	"bytes"
	"io"
	"net/http"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func newRecorder(t *testing.T) *recorder.Recorder {
	cassetteName := filepath.Join("testdata", t.Name())

	r, err := recorder.New(
		cassetteName,
		recorder.WithMode(recorder.ModeRecordOnce),
		recorder.WithMatcher(customMatcher(t)),
		recorder.WithHook(hookRemoveHeaders, recorder.AfterCaptureHook),
	)
	if err != nil {
		t.Fatalf("recorder: failed to create recorder: %v", err)
	}

	t.Cleanup(func() {
		if err := r.Stop(); err != nil {
			t.Errorf("recorder: failed to stop recorder: %v", err)
		}
	})

	return r
}

func customMatcher(t *testing.T) recorder.MatcherFunc {
	return func(r *http.Request, i cassette.Request) bool {
		if r.Body == nil || r.Body == http.NoBody {
			return cassette.DefaultMatcher(r, i)
		}

		var reqBody []byte
		var err error
		reqBody, err = io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("recorder: failed to read request body")
		}
		r.Body.Close()
		r.Body = io.NopCloser(bytes.NewBuffer(reqBody))

		return r.Method == i.Method && r.URL.String() == i.URL && string(reqBody) == i.Body
	}
}

var headersToKeep = map[string]struct{}{
	"accept":       {},
	"content-type": {},
	"user-agent":   {},
}

func hookRemoveHeaders(i *cassette.Interaction) error {
	for k := range i.Request.Headers {
		if _, ok := headersToKeep[strings.ToLower(k)]; !ok {
			delete(i.Request.Headers, k)
		}
	}
	for k := range i.Response.Headers {
		if _, ok := headersToKeep[strings.ToLower(k)]; !ok {
			delete(i.Response.Headers, k)
		}
	}
	return nil
}
