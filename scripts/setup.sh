#!/bin/bash

# Setup script for Fantasy AI SDK development environment
set -e

echo "🚀 Setting up Fantasy AI SDK development environment..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_status() {
    local status=$1
    local message=$2
    case $status in
        "OK") echo -e "${GREEN}✓ $message${NC}" ;;
        "WARN") echo -e "${YELLOW}⚠ $message${NC}" ;;
        "FAIL") echo -e "${RED}✗ $message${NC}" ;;
    esac
}

# Install golangci-lint v2
echo "📦 Installing golangci-lint v2.0.0..."
if command -v go &> /dev/null; then
    go install github.com/golangci/golangci-lint/cmd/golangci-lint@v2.0.0
    print_status "OK" "golangci-lint v2.0.0 installed"
else
    print_status "FAIL" "Go not found in PATH"
    exit 1
fi

# Install task CLI
echo "📦 Installing task CLI..."
go install github.com/go-task/task/v3/cmd/task@latest
print_status "OK" "task CLI installed"

# Install additional development tools
echo "📦 Installing development tools..."
go install golang.org/x/tools/cmd/goimports@latest
go install github.com/securego/gosec/v2/cmd/gosec@latest
print_status "OK" "Development tools installed"

# Set up pre-commit hook
echo "🔧 Setting up pre-commit hook..."
if [ -f "scripts/pre-commit.sh" ]; then
    cp scripts/pre-commit.sh .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit
    print_status "OK" "Pre-commit hook installed"
else
    print_status "WARN" "Pre-commit script not found at scripts/pre-commit.sh"
fi

# Verify installations
echo ""
echo "🔍 Verifying installations..."

if command -v golangci-lint &> /dev/null; then
    VERSION=$(golangci-lint version | head -n1)
    print_status "OK" "golangci-lint: $VERSION"
else
    print_status "FAIL" "golangci-lint not found"
fi

if command -v task &> /dev/null; then
    TASK_VERSION=$(task --version 2>/dev/null || echo "unknown")
    print_status "OK" "task: $TASK_VERSION"
else
    print_status "FAIL" "task not found"
fi

if command -v goimports &> /dev/null; then
    print_status "OK" "goimports installed"
else
    print_status "WARN" "goimports not found"
fi

if command -v gosec &> /dev/null; then
    print_status "OK" "gosec installed"
else
    print_status "WARN" "gosec not found"
fi

echo ""
print_status "OK" "Setup completed!"
echo ""
echo "🎯 Quick start commands:"
echo "  task ci-verify      # Run complete CI verification"
echo "  task test           # Run all tests"
echo "  task lint           # Run linter"
echo "  task build          # Build project"
echo "  task -l             # List all available tasks"
echo ""
echo "💡 Pre-commit hook is now active - it will run checks before each commit"