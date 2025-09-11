package ai

func FloatOption(f float64) *float64 {
	return &f
}

func BoolOption(b bool) *bool {
	return &b
}

func StringOption(s string) *string {
	return &s
}

func IntOption(i int64) *int64 {
	return &i
}
