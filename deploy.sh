#!/bin/bash

# FFGL DX11/Metal Plugin Deployment Script
# Deploys plugins to system directories for various VJ software

set -e

# Default values
PLUGIN="all"
ARCH="current"
SOURCE_DIR=""
DRY_RUN=false
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

ALL_PLUGINS="passthrough invert blur kitchen-sink"

# Display name for a plugin key
get_plugin_display() {
    case "$1" in
        passthrough)  echo "Passthrough"  ;;
        invert)       echo "Invert"       ;;
        blur)         echo "Blur"         ;;
        kitchen-sink) echo "KitchenSink"  ;;
        *) return 1 ;;
    esac
}

# Print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --plugin PLUGIN        Plugin to deploy: passthrough, invert, blur, kitchen-sink, all (default: all)"
    echo "  --arch ARCH            Architecture: arm64, x86_64, universal, current (default: current)"
    echo "  --source-dir DIR       Source directory (default: auto-detect from dist/)"
    echo "  --dry-run              Show what would be deployed without actually copying"
    echo "  --verbose              Enable verbose output"
    echo "  --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Deploy all plugins for current arch"
    echo "  $0 --plugin blur --arch universal    # Deploy blur plugin universal binary"
    echo "  $0 --dry-run                         # Show deployment plan without copying"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --plugin)
            PLUGIN="$2"
            shift 2
            ;;
        --arch)
            ARCH="$2"
            shift 2
            ;;
        --source-dir)
            SOURCE_DIR="$2"
            shift 2
            ;;
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

# Logging functions
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_dry_run() { echo -e "${YELLOW}[DRY RUN]${NC} $1"; }

# Detect current platform
detect_platform() {
    case "$(uname -s)" in
        Darwin*)            echo "macos"   ;;
        CYGWIN*|MINGW32*|MINGW64*|MSYS*) echo "windows" ;;
        *)
            log_error "Unsupported platform: $(uname -s)"
            exit 1
            ;;
    esac
}

# Detect current architecture
detect_arch() {
    case "$(uname -m)" in
        arm64|aarch64) echo "arm64"  ;;
        x86_64|amd64)  echo "x86_64" ;;
        *)
            log_error "Unsupported architecture: $(uname -m)"
            exit 1
            ;;
    esac
}

# Get deployment directories for macOS
get_macos_deploy_dirs() {
    local dirs=()

    # Resolume Arena Extra Effects (primary target)
    local resolume_extra="$HOME/Documents/Resolume Arena/Extra Effects"
    if [ -d "$HOME/Documents/Resolume Arena" ] || mkdir -p "$resolume_extra" 2>/dev/null; then
        dirs+=("$resolume_extra")
    fi

    # VDMX
    if [ -d "/Library/Graphics/FreeFrame" ]; then
        dirs+=("/Library/Graphics/FreeFrame")
    fi

    # User FreeFrame directory
    local user_ff="$HOME/Library/Graphics/FreeFrame"
    if [ -d "$user_ff" ] || mkdir -p "$user_ff" 2>/dev/null; then
        dirs+=("$user_ff")
    fi

    # MadMapper
    if [ -d "/Applications/MadMapper.app/Contents/Resources/FFGL" ]; then
        dirs+=("/Applications/MadMapper.app/Contents/Resources/FFGL")
    fi

    printf '%s\n' "${dirs[@]}"
}

# Get deployment directories for Windows
get_windows_deploy_dirs() {
    local dirs=()

    local program_files_dirs=(
        "/c/Program Files/Resolume Arena/FFGL"
        "/c/Program Files/Resolume Avenue/FFGL"
        "/c/Program Files (x86)/Resolume Arena/FFGL"
        "/c/Program Files (x86)/Resolume Avenue/FFGL"
    )

    for dir in "${program_files_dirs[@]}"; do
        if [ -d "$dir" ]; then
            dirs+=("$dir")
        fi
    done

    if [ -n "$APPDATA" ]; then
        local user_ff="$APPDATA/FreeFrame"
        if [ -d "$user_ff" ] || mkdir -p "$user_ff" 2>/dev/null; then
            dirs+=("$user_ff")
        fi
    fi

    printf '%s\n' "${dirs[@]}"
}

# Get deployment directories for current platform
get_deploy_dirs() {
    local platform
    platform=$(detect_platform)
    case "$platform" in
        macos)   get_macos_deploy_dirs   ;;
        windows) get_windows_deploy_dirs ;;
    esac
}

# Find source directory
find_source_dir() {
    local platform
    platform=$(detect_platform)
    local arch=$1

    if [ -n "$SOURCE_DIR" ]; then
        if [ -d "$SOURCE_DIR" ]; then
            echo "$SOURCE_DIR"
            return 0
        else
            log_error "Specified source directory does not exist: $SOURCE_DIR"
            exit 1
        fi
    fi

    local dist_dir="dist/$platform/$arch"
    if [ -d "$dist_dir" ]; then
        echo "$dist_dir"
        return 0
    fi

    # Try universal on macOS if current arch not found
    if [ "$platform" = "macos" ] && [ -d "dist/$platform/universal" ]; then
        log_info "Using universal binary (current arch $arch not found)"
        echo "dist/$platform/universal"
        return 0
    fi

    log_error "No built plugins found. Run ./build.sh first."
    log_error "Looked for: $dist_dir"
    if [ "$platform" = "macos" ]; then
        log_error "Also looked for: dist/$platform/universal"
    fi
    exit 1
}

# Deploy a single plugin
deploy_plugin() {
    local plugin_name=$1
    local source_path=$2
    local dest_dir=$3

    if [ ! -e "$source_path" ]; then
        log_warning "Plugin not found: $source_path"
        return 1
    fi

    local dest_path="$dest_dir/$(basename "$source_path")"

    if [ "$DRY_RUN" = true ]; then
        log_dry_run "Would deploy: $source_path -> $dest_path"
        return 0
    fi

    if [ "$VERBOSE" = true ]; then
        log_info "Deploying: $source_path -> $dest_path"
    fi

    if ! mkdir -p "$dest_dir" 2>/dev/null; then
        log_error "Cannot create directory: $dest_dir (permission denied?)"
        return 1
    fi

    if [ -d "$source_path" ]; then
        if cp -R "$source_path" "$dest_dir/"; then
            log_success "Deployed $plugin_name bundle to $dest_dir"
        else
            log_error "Failed to deploy $plugin_name bundle to $dest_dir"
            return 1
        fi
    else
        if cp "$source_path" "$dest_path"; then
            log_success "Deployed $plugin_name to $dest_dir"
        else
            log_error "Failed to deploy $plugin_name to $dest_dir"
            return 1
        fi
    fi
}

# Main deployment function
main() {
    log_info "FFGL DX11/Metal Plugin Deployment"

    local platform
    platform=$(detect_platform)

    if [ "$ARCH" = "current" ]; then
        ARCH=$(detect_arch)
        log_info "Detected architecture: $ARCH"
    fi

    local source_dir
    source_dir=$(find_source_dir "$ARCH")
    log_info "Source directory: $source_dir"

    # Get deployment directories
    local deploy_dirs=()
    while IFS= read -r dir; do
        [ -n "$dir" ] && deploy_dirs+=("$dir")
    done < <(get_deploy_dirs)

    if [ ${#deploy_dirs[@]} -eq 0 ]; then
        log_warning "No VJ software installation directories found"
        log_info "You may need to manually copy plugins from: $source_dir"
        exit 0
    fi

    log_info "Found ${#deploy_dirs[@]} deployment target(s):"
    for dir in "${deploy_dirs[@]}"; do
        log_info "  - $dir"
    done

    # Determine plugins to deploy
    local plugins=()
    if [ "$PLUGIN" = "all" ]; then
        for p in $ALL_PLUGINS; do
            plugins+=("$p")
        done
    else
        plugins=("$PLUGIN")
    fi

    # Deploy each plugin to each directory
    local success_count=0
    local total_count=0

    for plugin in "${plugins[@]}"; do
        local display_name
        display_name=$(get_plugin_display "$plugin")

        local bundle_path="$source_dir/${display_name}.bundle"
        local dylib_path="$source_dir/${display_name}.dylib"
        local dll_path="$source_dir/${display_name}.dll"

        for deploy_dir in "${deploy_dirs[@]}"; do
            local plugin_path=""

            if [ "$platform" = "macos" ]; then
                if [[ "$deploy_dir" == *"Resolume"* ]]; then
                    # Resolume prefers bundles
                    if [ -d "$bundle_path" ]; then
                        plugin_path="$bundle_path"
                    elif [ -f "$dylib_path" ]; then
                        plugin_path="$dylib_path"
                    fi
                else
                    # VDMX, FreeFrame, MadMapper prefer dylibs
                    if [ -f "$dylib_path" ]; then
                        plugin_path="$dylib_path"
                    elif [ -d "$bundle_path" ]; then
                        plugin_path="$bundle_path"
                    fi
                fi
            else
                plugin_path="$dll_path"
            fi

            if [ -z "$plugin_path" ] || [ ! -e "$plugin_path" ]; then
                log_warning "No suitable plugin found for $plugin in $deploy_dir"
                continue
            fi

            total_count=$((total_count + 1))
            if deploy_plugin "$plugin" "$plugin_path" "$deploy_dir"; then
                success_count=$((success_count + 1))
            fi
        done
    done

    # Summary
    echo ""
    if [ "$DRY_RUN" = true ]; then
        log_info "Dry run complete. $total_count deployment(s) would be performed."
    else
        log_info "Deployment complete: $success_count/$total_count successful"

        if [ $success_count -eq $total_count ]; then
            log_success "All plugins deployed successfully!"
        elif [ $success_count -gt 0 ]; then
            log_warning "Some deployments failed. Check permissions and paths."
        else
            log_error "All deployments failed. Check permissions and paths."
            exit 1
        fi
    fi
}

main
