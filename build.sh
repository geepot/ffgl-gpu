#!/bin/bash

# FFGL Plugin Cross-Platform Build Script (unified WGSL → Metal/GLSL)
# Builds plugins for macOS (arm64/x86_64/universal) and Windows (x86_64)

set -e

# Default values
PLATFORM="current"
ARCH="current"
PLUGIN="all"
PROFILE="release"
TOOLCHAIN="msvc"
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ── Plugin registry (Bash 3.2-compatible) ────────────────────────────────────
ALL_PLUGINS="passthrough invert blur kitchen-sink"

# Cargo package name for a plugin key
get_plugin_package() {
    case "$1" in
        passthrough)  echo "ffgl-passthrough"  ;;
        invert)       echo "ffgl-invert"       ;;
        blur)         echo "ffgl-blur"         ;;
        kitchen-sink) echo "ffgl-kitchen-sink" ;;
        *) return 1 ;;
    esac
}

# Cargo output filename stem (hyphens become underscores)
get_plugin_stem() {
    case "$1" in
        passthrough)  echo "ffgl_passthrough"  ;;
        invert)       echo "ffgl_invert"       ;;
        blur)         echo "ffgl_blur"         ;;
        kitchen-sink) echo "ffgl_kitchen_sink" ;;
        *) return 1 ;;
    esac
}

# Display / distribution name
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
    echo "  --platform PLATFORM    Target platform: macos, windows, all, current (default: current)"
    echo "  --arch ARCH            Target architecture: arm64, x86_64, universal, current (default: current)"
    echo "  --plugin PLUGIN        Plugin to build: passthrough, invert, blur, kitchen-sink, all (default: all)"
    echo "  --profile PROFILE      Build profile: debug, release (default: release)"
    echo "  --toolchain TOOLCHAIN  Windows toolchain: msvc, gnu (default: msvc)"
    echo "  --verbose              Enable verbose output"
    echo "  --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                          # Build for current platform/arch, all plugins"
    echo "  $0 --platform macos --arch universal       # Build universal macOS binaries"
    echo "  $0 --platform windows --plugin blur        # Build blur plugin for Windows (MSVC)"
    echo "  $0 --platform windows --toolchain gnu      # Build for Windows using MinGW"
    echo "  $0 --platform all --plugin passthrough     # Build passthrough plugin for all platforms"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --arch)
            ARCH="$2"
            shift 2
            ;;
        --plugin)
            PLUGIN="$2"
            shift 2
            ;;
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --toolchain)
            TOOLCHAIN="$2"
            shift 2
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
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect current platform
detect_platform() {
    case "$(uname -s)" in
        Darwin*)
            echo "macos"
            ;;
        CYGWIN*|MINGW32*|MINGW64*|MSYS*)
            echo "windows"
            ;;
        *)
            log_error "Unsupported platform: $(uname -s)"
            exit 1
            ;;
    esac
}

# Detect current architecture
detect_arch() {
    case "$(uname -m)" in
        arm64|aarch64)
            echo "arm64"
            ;;
        x86_64|amd64)
            echo "x86_64"
            ;;
        *)
            log_error "Unsupported architecture: $(uname -m)"
            exit 1
            ;;
    esac
}

# Get Rust target for platform/arch combination
get_rust_target() {
    local platform=$1
    local arch=$2
    local toolchain=${3:-"msvc"}

    case "$platform-$arch-$toolchain" in
        macos-arm64-*)
            echo "aarch64-apple-darwin"
            ;;
        macos-x86_64-*)
            echo "x86_64-apple-darwin"
            ;;
        windows-x86_64-msvc)
            echo "x86_64-pc-windows-msvc"
            ;;
        windows-x86_64-gnu)
            echo "x86_64-pc-windows-gnu"
            ;;
        *)
            log_error "Unsupported platform-arch-toolchain combination: $platform-$arch-$toolchain"
            exit 1
            ;;
    esac
}

# Get library extension for platform
get_lib_extension() {
    local platform=$1
    case "$platform" in
        macos)
            echo "dylib"
            ;;
        windows)
            echo "dll"
            ;;
        *)
            log_error "Unknown platform: $platform"
            exit 1
            ;;
    esac
}

# Get library prefix for platform
get_lib_prefix() {
    local platform=$1
    case "$platform" in
        macos)
            echo "lib"
            ;;
        windows)
            echo ""
            ;;
        *)
            log_error "Unknown platform: $platform"
            exit 1
            ;;
    esac
}

# Install Rust target if not already installed
ensure_rust_target() {
    local target=$1
    if ! rustup target list --installed | grep -q "^$target$"; then
        log_info "Installing Rust target: $target"
        rustup target add "$target"
    fi
}

# Create bundle for macOS
create_bundle() {
    local plugin_key=$1
    local platform=$2
    local arch=$3

    if [ "$platform" != "macos" ]; then
        return 0
    fi

    local display_name
    display_name=$(get_plugin_display "$plugin_key")
    local dist_dir="dist/$platform/$arch"
    local dylib_file="$dist_dir/$display_name.dylib"
    local bundle_dir="$dist_dir/$display_name.bundle"
    local bundle_executable="$bundle_dir/Contents/MacOS/$display_name"

    if [ ! -f "$dylib_file" ]; then
        log_warning "Dylib not found for bundle creation: $dylib_file"
        return 1
    fi

    log_info "Creating bundle for $display_name ($arch)"

    mkdir -p "$bundle_dir/Contents/MacOS"
    cp "$dylib_file" "$bundle_executable"

    log_success "Created bundle -> $bundle_dir"
}

# Copy built plugin artifacts to dist directory
copy_plugin_artifacts() {
    local plugin_key=$1
    local target=$2
    local platform=$3
    local arch=$4

    local lib_prefix
    lib_prefix=$(get_lib_prefix "$platform")
    local lib_ext
    lib_ext=$(get_lib_extension "$platform")
    local profile_dir
    if [ "$PROFILE" = "debug" ]; then
        profile_dir="debug"
    else
        profile_dir="$PROFILE"
    fi

    local stem
    stem=$(get_plugin_stem "$plugin_key")
    local display_name
    display_name=$(get_plugin_display "$plugin_key")
    local source_lib="target/$target/$profile_dir/${lib_prefix}${stem}.$lib_ext"
    local dist_dir="dist/$platform/$arch"
    local dest_lib="$dist_dir/$display_name.$lib_ext"

    mkdir -p "$dist_dir"

    if [ -f "$source_lib" ]; then
        cp "$source_lib" "$dest_lib"
        log_success "Built $plugin_key -> $dest_lib"

        create_bundle "$plugin_key" "$platform" "$arch"
    else
        log_error "Built library not found: $source_lib"
        return 1
    fi
}

# Create universal binary on macOS
create_universal_binary() {
    local plugin_key=$1
    local lib_ext="dylib"
    local display_name
    display_name=$(get_plugin_display "$plugin_key")

    local arm64_lib="dist/macos/arm64/$display_name.$lib_ext"
    local x86_64_lib="dist/macos/x86_64/$display_name.$lib_ext"
    local universal_dir="dist/macos/universal"
    local universal_lib="$universal_dir/$display_name.$lib_ext"

    if [ -f "$arm64_lib" ] && [ -f "$x86_64_lib" ]; then
        mkdir -p "$universal_dir"
        log_info "Creating universal binary for $display_name"
        lipo -create "$arm64_lib" "$x86_64_lib" -output "$universal_lib"
        log_success "Created universal binary -> $universal_lib"

        create_bundle "$plugin_key" "macos" "universal"
    else
        log_warning "Cannot create universal binary for $display_name: missing arm64 or x86_64 build"
        if [ ! -f "$arm64_lib" ]; then
            log_warning "Missing: $arm64_lib"
        fi
        if [ ! -f "$x86_64_lib" ]; then
            log_warning "Missing: $x86_64_lib"
        fi
    fi
}

# Main build function
main() {
    log_info "FFGL Plugin Build System (WGSL transpiled)"
    log_info "Platform: $PLATFORM, Architecture: $ARCH, Plugin: $PLUGIN, Profile: $PROFILE, Toolchain: $TOOLCHAIN"

    # Resolve current platform/arch
    if [ "$PLATFORM" = "current" ]; then
        PLATFORM=$(detect_platform)
        log_info "Detected platform: $PLATFORM"
    fi

    if [ "$ARCH" = "current" ]; then
        ARCH=$(detect_arch)
        log_info "Detected architecture: $ARCH"
    fi

    # Clean previous dist directory
    if [ -d "dist" ]; then
        log_info "Cleaning previous build artifacts"
        rm -rf dist
    fi

    # Determine platforms to build
    local platforms=()
    if [ "$PLATFORM" = "all" ]; then
        platforms=("macos" "windows")
    else
        platforms=("$PLATFORM")
    fi

    # Determine plugins to build
    local plugins=()
    if [ "$PLUGIN" = "all" ]; then
        for p in $ALL_PLUGINS; do
            plugins+=("$p")
        done
    else
        plugins=("$PLUGIN")
    fi

    # Validate plugin names
    for plugin in "${plugins[@]}"; do
        if ! get_plugin_package "$plugin" >/dev/null 2>&1; then
            log_error "Unknown plugin: $plugin"
            log_error "Valid plugins: $ALL_PLUGINS"
            exit 1
        fi
    done

    # Set LIBCLANG_PATH per-platform (required by bindgen for ffgl-core FFI generation)
    if [ -z "$LIBCLANG_PATH" ]; then
        local current_os
        current_os=$(uname -s)
        case "$current_os" in
            Darwin*)
                if [ -d "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib" ]; then
                    export LIBCLANG_PATH="/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib"
                elif [ -d "/Library/Developer/CommandLineTools/usr/lib" ]; then
                    export LIBCLANG_PATH="/Library/Developer/CommandLineTools/usr/lib"
                else
                    log_warning "Could not find libclang.dylib — bindgen may fail. Set LIBCLANG_PATH manually."
                fi
                ;;
            CYGWIN*|MINGW32*|MINGW64*|MSYS*)
                local vs_found=""
                for vs_ver in 18 2022 17 2019; do
                    for vs_ed in Community Professional Enterprise BuildTools; do
                        local vs_try="C:\\Program Files\\Microsoft Visual Studio\\${vs_ver}\\${vs_ed}\\VC\\Tools\\Llvm\\x64\\bin"
                        if [ -d "$vs_try" ] 2>/dev/null; then
                            vs_found="$vs_try"
                            break 2
                        fi
                    done
                done
                if [ -z "$vs_found" ] && [ -d "C:\\Program Files\\LLVM\\bin" ] 2>/dev/null; then
                    vs_found="C:\\Program Files\\LLVM\\bin"
                fi
                if [ -n "$vs_found" ]; then
                    export LIBCLANG_PATH="$vs_found"
                else
                    log_warning "Could not find libclang on Windows — bindgen may fail. Set LIBCLANG_PATH manually."
                fi
                ;;
        esac
        if [ -n "$LIBCLANG_PATH" ]; then
            log_info "LIBCLANG_PATH=$LIBCLANG_PATH"
        fi
    fi

    # Build for each platform
    for platform in "${platforms[@]}"; do
        log_info "Building for platform: $platform"

        # Determine architectures for this platform
        local architectures=()
        if [ "$platform" = "macos" ]; then
            if [ "$ARCH" = "universal" ]; then
                architectures=("arm64" "x86_64")
            else
                architectures=("$ARCH")
            fi
        elif [ "$platform" = "windows" ]; then
            if [ "$ARCH" = "universal" ]; then
                log_warning "Universal binaries not supported on Windows, using x86_64"
                architectures=("x86_64")
            elif [ "$ARCH" = "current" ] || [ "$ARCH" = "arm64" ]; then
                log_warning "Windows ARM64 not supported, using x86_64"
                architectures=("x86_64")
            else
                architectures=("$ARCH")
            fi
        fi

        # Build for each architecture
        for arch in "${architectures[@]}"; do
            local target
            target=$(get_rust_target "$platform" "$arch" "$TOOLCHAIN")

            ensure_rust_target "$target"

            # Build all plugins in a single cargo invocation for maximum parallelism
            local package_args=""
            for plugin in "${plugins[@]}"; do
                package_args="$package_args -p $(get_plugin_package "$plugin")"
            done

            log_info "Building plugins (${plugins[*]}) for $target ($platform-$arch)"

            local build_cmd
            if [ "$PROFILE" = "debug" ]; then
                build_cmd="cargo build --target $target $package_args"
            else
                build_cmd="cargo build --profile $PROFILE --target $target $package_args"
            fi

            if [ "$VERBOSE" = true ]; then
                build_cmd="$build_cmd --verbose"
            fi

            if ! $build_cmd; then
                log_error "Failed to build plugins for $target"
                exit 1
            fi

            # Copy built artifacts to dist
            for plugin in "${plugins[@]}"; do
                copy_plugin_artifacts "$plugin" "$target" "$platform" "$arch"
            done
        done

        # Create universal binaries if requested and on macOS
        if [ "$platform" = "macos" ] && [ "$ARCH" = "universal" ]; then
            for plugin in "${plugins[@]}"; do
                create_universal_binary "$plugin"
            done
        fi
    done

    log_success "Build complete! Check the dist/ directory for outputs."

    # Show what was built
    if [ -d "dist" ]; then
        echo ""
        log_info "Built artifacts:"
        echo ""
        log_info "Libraries:"
        find dist -name "*.dylib" -o -name "*.dll" | sort
        echo ""
        log_info "Bundles:"
        find dist -name "*.bundle" -type d | sort
        if [ ! "$(find dist -name "*.bundle" -type d)" ]; then
            echo "  (none - bundles are only created on macOS)"
        fi
    fi
}

# Run main function
main
