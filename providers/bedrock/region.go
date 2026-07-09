package bedrock

import (
	"os"
	"strings"
)

// applyRegionPrefix adds the region prefix to the model ID if not already present.
// It extracts the first two characters from the region string to create the prefix.
// If the region is invalid (empty or less than 2 characters), it defaults to "us.".
// If the model ID already has a region prefix, it returns the model ID unchanged.
func applyRegionPrefix(modelID, region string) string {
	// Default to "us." if region is invalid
	if len(region) < 2 {
		region = "us-east-1"
	}

	// Extract region prefix (first two characters + ".")
	prefix := region[:2] + "."

	// Check if already prefixed to avoid duplication
	if strings.HasPrefix(modelID, prefix) {
		return modelID
	}

	// Check if it has any other region prefix (e.g., "eu.", "ap.", etc.)
	// Region prefixes are always 2 letters followed by a dot
	if len(modelID) >= 3 && modelID[2] == '.' {
		// Check if the first two characters are lowercase letters (region code pattern)
		firstTwo := modelID[:2]
		if isLowercaseLetters(firstTwo) {
			// Already has a region prefix, don't add another
			return modelID
		}
	}

	return prefix + modelID
}

// isLowercaseLetters checks if a string contains only lowercase letters
func isLowercaseLetters(s string) bool {
	for _, c := range s {
		if c < 'a' || c > 'z' {
			return false
		}
	}
	return true
}

// getRegionFromEnv reads the AWS_REGION environment variable.
// This is a helper function for tests and can be used when region is not provided.
func getRegionFromEnv() string {
	region := os.Getenv("AWS_REGION")
	if region == "" {
		return "us-east-1"
	}
	return region
}
