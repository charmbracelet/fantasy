package bedrock

// This file ensures AWS SDK Bedrock Runtime dependency is retained in go.mod
// until the Nova implementation is complete. It will be removed once nova.go
// is implemented with actual usage of these imports.

import (
	_ "github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)
