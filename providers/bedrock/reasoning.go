package bedrock

import (
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

func reasoningTextFromBlock(block types.ReasoningContentBlock) (string, *ReasoningOptionMetadata) {
	switch content := block.(type) {
	case *types.ReasoningContentBlockMemberReasoningText:
		text := ""
		if content.Value.Text != nil {
			text = *content.Value.Text
		}
		return text, nil
	case *types.ReasoningContentBlockMemberRedactedContent:
		return "", &ReasoningOptionMetadata{
			RedactedData: string(content.Value),
		}
	default:
		return "", nil
	}
}
