package kronk

import (
	"context"
	"errors"
	"regexp"
	"strconv"
	"strings"

	"charm.land/fantasy"
	krn "github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
)

var kronkContextPattern = regexp.MustCompile(`input tokens \[(\d+)\] exceed context window \[(\d+)\]`)

func toProviderErr(err error) error {
	if err == nil || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return err
	}

	if errors.Is(err, krn.ErrAdmissionTimeout) {
		return &fantasy.ProviderError{
			Title:          "kronk model busy",
			Message:        err.Error(),
			Cause:          err,
			TransientError: true,
		}
	}

	if isInvalidRequest(err) {
		providerErr := &fantasy.ProviderError{
			Title:   "invalid kronk request",
			Message: err.Error(),
			Cause:   err,
		}
		parseContextTooLargeError(err.Error(), providerErr)
		return providerErr
	}

	if strings.Contains(err.Error(), "context window is full") {
		return &fantasy.ProviderError{
			Title:              "kronk context window exceeded",
			Message:            err.Error(),
			Cause:              err,
			ContextTooLargeErr: true,
		}
	}

	if transportErr := fantasy.WrapTransportError(err); transportErr != err {
		return transportErr
	}

	return &fantasy.ProviderError{
		Title:   "kronk request failed",
		Message: err.Error(),
		Cause:   err,
	}
}

func toOperationErr(title string, err error) error {
	mappedErr := toProviderErr(err)
	var providerErr *fantasy.ProviderError
	if errors.As(mappedErr, &providerErr) && providerErr.Title == "kronk request failed" {
		providerErr.Title = title
	}
	return mappedErr
}

func isInvalidRequest(err error) bool {
	return errors.Is(err, model.ErrInvalidRequest) ||
		errors.Is(err, model.ErrMessagesMissing) ||
		errors.Is(err, model.ErrMessagesInvalid) ||
		errors.Is(err, model.ErrFileInputsUnsupported)
}

func parseContextTooLargeError(message string, providerErr *fantasy.ProviderError) {
	matches := kronkContextPattern.FindStringSubmatch(message)
	if matches == nil {
		return
	}

	providerErr.Title = "kronk context window exceeded"
	providerErr.ContextTooLargeErr = true
	providerErr.ContextUsedTokens, _ = strconv.Atoi(matches[1])
	providerErr.ContextMaxTokens, _ = strconv.Atoi(matches[2])
}
