#!/bin/bash

# FFGL DX11/Metal Plugin Clean Script
# Cleans build artifacts and distribution files

set -e

VERBOSE=false
DRY_RUN=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --dry-run              Show what would be cleaned without actually removing"
    echo "  --verbose              Enable verbose output"
    echo "  --help                 Show this help message"
    echo ""
    echo "This script cleans:"
    echo "  - Cargo target directory"
    echo "  - Distribution directory (dist/)"
    echo "  - Cargo lock file"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_dry_run() { echo -e "${YELLOW}[DRY RUN]${NC} $1"; }

remove_path() {
    local path=$1
    local description=$2

    if [ ! -e "$path" ]; then
        if [ "$VERBOSE" = true ]; then
            log_info "Already clean: $description ($path)"
        fi
        return 0
    fi

    if [ "$DRY_RUN" = true ]; then
        log_dry_run "Would remove: $description ($path)"
        return 0
    fi

    if [ "$VERBOSE" = true ]; then
        log_info "Removing: $description ($path)"
    fi

    if rm -rf "$path"; then
        log_success "Cleaned: $description"
    else
        log_error "Failed to remove: $description ($path)"
        return 1
    fi
}

main() {
    log_info "FFGL DX11/Metal Plugin Clean"

    local success_count=0
    local total_count=0

    local items=(
        "target:Workspace target directory"
        "dist:Distribution directory"
        "Cargo.lock:Workspace lock file"
    )

    for item in "${items[@]}"; do
        local path="${item%%:*}"
        local description="${item##*:}"

        total_count=$((total_count + 1))
        if remove_path "$path" "$description"; then
            success_count=$((success_count + 1))
        fi
    done

    echo ""
    if [ "$DRY_RUN" = true ]; then
        log_info "Dry run complete. $total_count item(s) would be cleaned."
    else
        log_info "Clean complete: $success_count/$total_count successful"

        if [ $success_count -eq $total_count ]; then
            log_success "All items cleaned successfully!"
        elif [ $success_count -gt 0 ]; then
            log_warning "Some items could not be cleaned."
        else
            log_error "No items were cleaned."
            exit 1
        fi
    fi
}

main
