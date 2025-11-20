#!/bin/bash

# Pre-commit hook for Fantasy AI SDK
# This script helps prevent CI failures by running key checks locally

set -e

echo "🔍 Running pre-commit checks..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "OK") echo -e "${GREEN}✓ $message${NC}" ;;
        "WARN") echo -e "${YELLOW}⚠ $message${NC}" ;;
        "FAIL") echo -e "${RED}✗ $message${NC}" ;;
    esac
}

# Check if we're in a Go project
if [ ! -f "go.mod" ]; then
    print_status "WARN" "Not in a Go project (no go.mod found)"
    exit 0
fi

echo "📦 Checking Go modules..."
go mod download
go mod verify

# Check if go mod tidy would make changes
echo "🧹 Checking if go.mod/go.sum need updates..."
if ! go mod tidy -diff > /dev/null 2>&1; then
    echo "Running go mod tidy..."
    go mod tidy
    
    if [ -n "$(git status --porcelain go.mod go.sum)" ]; then
        print_status "WARN" "go.mod or go.sum changed:"
        git diff --stat go.mod go.sum
        
        read -p "Commit these changes? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git add go.mod go.sum
            git commit -m "chore: update go.sum after dependency changes"
            print_status "OK" "Committed go.mod/go.sum changes"
        else
            print_status "WARN" "Please commit go.mod/go.sum changes manually"
        fi
    fi
else
    print_status "OK" "go.mod and go.sum are in sync"
fi

echo "🔨 Building project..."
if go build -v ./...; then
    print_status "OK" "Build successful"
else
    print_status "FAIL" "Build failed"
    exit 1
fi

echo "🧪 Running tests..."
mkdir -p test-results
if go test -v ./... -count=1 > test-results/pre-commit-test.log 2>&1; then
    print_status "OK" "All tests passed - see test-results/pre-commit-test.log for details"
else
    echo "❌ Test failures:"
    cat test-results/pre-commit-test.log
    print_status "FAIL" "Tests failed"
    exit 1
fi

# Check if golangci-lint is available
if command -v golangci-lint &> /dev/null; then
    echo "🔍 Running linter..."
    mkdir -p test-results
    if golangci-lint run --timeout=5m > test-results/lint.log 2>&1; then
        print_status "OK" "Linting passed - see test-results/lint.log for details"
    else
        echo "⚠️ Linting issues found:"
        cat test-results/lint.log
        print_status "WARN" "Linting found issues (see output above)"
    fi
else
    print_status "WARN" "golangci-lint not found - install with: go install github.com/golangci/golangci-lint/cmd/golangci-lint@v1.64.8"
fi

echo ""
print_status "OK" "Pre-commit checks completed!"
echo "💡 Tip: Use 'git commit --no-verify' to skip these checks if needed"