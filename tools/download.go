package tools

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/charmbracelet/crush/internal/ai"
	"github.com/charmbracelet/crush/internal/permission"
)

type DownloadParams struct {
	URL      string `json:"url" description:"The URL to download from"`
	FilePath string `json:"file_path" description:"The local file path where the downloaded content should be saved"`
	Timeout  int    `json:"timeout,omitempty" description:"Optional timeout in seconds (max 600)"`
}

type DownloadPermissionsParams struct {
	URL      string `json:"url"`
	FilePath string `json:"file_path"`
	Timeout  int    `json:"timeout,omitempty"`
}

const (
	DownloadToolName = "download"
)

func NewDownloadTool(permissions permission.Service, workingDir string) ai.AgentTool {
	client := &http.Client{
		Timeout: 5 * time.Minute, // Default 5 minute timeout for downloads
		Transport: &http.Transport{
			MaxIdleConns:        100,
			MaxIdleConnsPerHost: 10,
			IdleConnTimeout:     90 * time.Second,
		},
	}
	return ai.NewTypedToolFunc(
		DownloadToolName,
		`Downloads binary data from a URL and saves it to a local file.

WHEN TO USE THIS TOOL:
- Use when you need to download files, images, or other binary data from URLs
- Helpful for downloading assets, documents, or any file type
- Useful for saving remote content locally for processing or storage

HOW TO USE:
- Provide the URL to download from
- Specify the local file path where the content should be saved
- Optionally set a timeout for the request

FEATURES:
- Downloads any file type (binary or text)
- Automatically creates parent directories if they don't exist
- Handles large files efficiently with streaming
- Sets reasonable timeouts to prevent hanging
- Validates input parameters before making requests

LIMITATIONS:
- Maximum file size is 100MB
- Only supports HTTP and HTTPS protocols
- Cannot handle authentication or cookies
- Some websites may block automated requests
- Will overwrite existing files without warning

TIPS:
- Use absolute paths or paths relative to the working directory
- Set appropriate timeouts for large files or slow connections`,
		func(ctx context.Context, params DownloadParams, call ai.ToolCall) (ai.ToolResponse, error) {
			if params.URL == "" {
				return ai.NewTextErrorResponse("URL parameter is required"), nil
			}

			if params.FilePath == "" {
				return ai.NewTextErrorResponse("file_path parameter is required"), nil
			}

			if !strings.HasPrefix(params.URL, "http://") && !strings.HasPrefix(params.URL, "https://") {
				return ai.NewTextErrorResponse("URL must start with http:// or https://"), nil
			}

			// Convert relative path to absolute path
			var filePath string
			if filepath.IsAbs(params.FilePath) {
				filePath = params.FilePath
			} else {
				filePath = filepath.Join(workingDir, params.FilePath)
			}

			sessionID, messageID := GetContextValues(ctx)
			if sessionID == "" || messageID == "" {
				return ai.ToolResponse{}, fmt.Errorf("session ID and message ID are required for the download tool")
			}
			granted := permissions.Request(
				permission.CreatePermissionRequest{
					SessionID:   sessionID,
					Path:        filePath,
					ToolName:    DownloadToolName,
					Action:      "download",
					Description: fmt.Sprintf("Download file from URL: %s to %s", params.URL, filePath),
					Params:      DownloadPermissionsParams(params),
				},
			)

			if !granted {
				return ai.ToolResponse{}, permission.ErrorPermissionDenied
			}

			// Handle timeout with context
			requestCtx := ctx
			if params.Timeout > 0 {
				maxTimeout := 600 // 10 minutes
				if params.Timeout > maxTimeout {
					params.Timeout = maxTimeout
				}
				var cancel context.CancelFunc
				requestCtx, cancel = context.WithTimeout(ctx, time.Duration(params.Timeout)*time.Second)
				defer cancel()
			}

			req, err := http.NewRequestWithContext(requestCtx, "GET", params.URL, nil)
			if err != nil {
				return ai.ToolResponse{}, fmt.Errorf("failed to create request: %w", err)
			}

			req.Header.Set("User-Agent", "crush/1.0")

			resp, err := client.Do(req)
			if err != nil {
				return ai.ToolResponse{}, fmt.Errorf("failed to download from URL: %w", err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK {
				return ai.NewTextErrorResponse(fmt.Sprintf("Request failed with status code: %d", resp.StatusCode)), nil
			}

			// Check content length if available
			maxSize := int64(100 * 1024 * 1024) // 100MB
			if resp.ContentLength > maxSize {
				return ai.NewTextErrorResponse(fmt.Sprintf("File too large: %d bytes (max %d bytes)", resp.ContentLength, maxSize)), nil
			}

			// Create parent directories if they don't exist
			if err := os.MkdirAll(filepath.Dir(filePath), 0o755); err != nil {
				return ai.ToolResponse{}, fmt.Errorf("failed to create parent directories: %w", err)
			}

			// Create the output file
			outFile, err := os.Create(filePath)
			if err != nil {
				return ai.ToolResponse{}, fmt.Errorf("failed to create output file: %w", err)
			}
			defer outFile.Close()

			// Copy data with size limit
			limitedReader := io.LimitReader(resp.Body, maxSize)
			bytesWritten, err := io.Copy(outFile, limitedReader)
			if err != nil {
				return ai.ToolResponse{}, fmt.Errorf("failed to write file: %w", err)
			}

			// Check if we hit the size limit
			if bytesWritten == maxSize {
				// Clean up the file since it might be incomplete
				os.Remove(filePath)
				return ai.NewTextErrorResponse(fmt.Sprintf("File too large: exceeded %d bytes limit", maxSize)), nil
			}

			contentType := resp.Header.Get("Content-Type")
			responseMsg := fmt.Sprintf("Successfully downloaded %d bytes to %s", bytesWritten, filePath)
			if contentType != "" {
				responseMsg += fmt.Sprintf(" (Content-Type: %s)", contentType)
			}

			return ai.NewTextResponse(responseMsg), nil
		})
}
