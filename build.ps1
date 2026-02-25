# FFGL DX11/Metal Plugin Windows Build Script
# Mirrors build.sh for Windows (PowerShell)

param(
    [string]$Platform  = "windows",
    [string]$Arch      = "x86_64",
    [string]$Plugin    = "all",
    [string]$Profile   = "release",
    [string]$Toolchain = "msvc",
    [switch]$Verbose,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

function Write-Info    { param($msg) Write-Host "[INFO] $msg"    -ForegroundColor Cyan    }
function Write-Success { param($msg) Write-Host "[SUCCESS] $msg" -ForegroundColor Green   }
function Write-Warn    { param($msg) Write-Host "[WARNING] $msg" -ForegroundColor Yellow  }
function Write-Err     { param($msg) Write-Host "[ERROR] $msg"   -ForegroundColor Red     }

# ── Plugin registry ──────────────────────────────────────────────────────────
$PluginRegistry = @{
    "passthrough"  = @{ Package = "ffgl-passthrough";  Stem = "ffgl_passthrough";  Display = "Passthrough"  }
    "invert"       = @{ Package = "ffgl-invert";       Stem = "ffgl_invert";       Display = "Invert"       }
    "blur"         = @{ Package = "ffgl-blur";         Stem = "ffgl_blur";         Display = "Blur"         }
    "kitchen-sink" = @{ Package = "ffgl-kitchen-sink"; Stem = "ffgl_kitchen_sink"; Display = "KitchenSink"  }
}

function Show-Usage {
    Write-Host @"
Usage: .\build.ps1 [OPTIONS]

Options:
  -Plugin    PLUGIN     Plugin to build: passthrough, invert, blur, kitchen-sink, all (default: all)
  -Profile   PROFILE    Build profile: debug, release (default: release)
  -Toolchain TOOLCHAIN  Windows toolchain: msvc, gnu (default: msvc)
  -Verbose              Enable verbose output
  -Help                 Show this help message

Examples:
  .\build.ps1                                    # Build all plugins (release, MSVC)
  .\build.ps1 -Plugin blur                       # Build only the blur plugin
  .\build.ps1 -Profile debug                     # Debug build
  .\build.ps1 -Plugin passthrough -Verbose       # Verbose passthrough plugin build
  .\build.ps1 -Toolchain gnu                     # Build using MinGW toolchain
"@
}

function Get-RustTarget {
    param([string]$Toolchain)
    switch ($Toolchain) {
        "msvc" { return "x86_64-pc-windows-msvc" }
        "gnu"  { return "x86_64-pc-windows-gnu"  }
        default {
            Write-Err "Unknown toolchain: $Toolchain (use 'msvc' or 'gnu')"
            exit 1
        }
    }
}

function Ensure-RustTarget {
    param([string]$Target)
    $installed = rustup target list --installed
    if ($installed -notcontains $Target) {
        Write-Info "Installing Rust target: $Target"
        rustup target add $Target
        if ($LASTEXITCODE -ne 0) { Write-Err "Failed to install target $Target"; exit 1 }
    }
}

function Copy-PluginArtifacts {
    param([string]$PluginKey, [string]$Target, [string]$ProfileDir)

    $reg = $PluginRegistry[$PluginKey]
    $sourceDll = "target\$Target\$ProfileDir\$($reg.Stem).dll"
    $distDir   = "dist\windows\x86_64"
    $destDll   = "$distDir\$($reg.Display).dll"

    New-Item -ItemType Directory -Force -Path $distDir | Out-Null

    if (Test-Path $sourceDll) {
        Copy-Item $sourceDll $destDll -Force
        Write-Success "Built $PluginKey -> $destDll"
    } else {
        Write-Err "Built library not found: $sourceDll"
        exit 1
    }
}

# --- Main ---

if ($Help) { Show-Usage; exit 0 }

# Validate parameters
if ($Platform -notin @("windows", "current")) {
    Write-Warn "This script only supports Windows builds. Ignoring -Platform '$Platform'."
}
if ($Arch -notin @("x86_64", "current")) {
    Write-Warn "Only x86_64 is supported on Windows. Ignoring -Arch '$Arch'."
}

$target = Get-RustTarget $Toolchain
$profileDir = if ($Profile -eq "debug") { "debug" } else { $Profile }

$plugins = if ($Plugin -eq "all") { @("passthrough", "invert", "blur", "kitchen-sink") } else { @($Plugin) }

# Validate plugin names
foreach ($p in $plugins) {
    if (-not $PluginRegistry.ContainsKey($p)) {
        Write-Err "Unknown plugin: $p. Valid plugins: $($PluginRegistry.Keys -join ', ')"
        exit 1
    }
}

Write-Info "FFGL DX11/Metal Plugin Build System"
Write-Info "Target: $target | Plugins: $($plugins -join ', ') | Profile: $Profile"

# Set LIBCLANG_PATH if not already set (required by bindgen for ffgl-core FFI generation)
if (-not $env:LIBCLANG_PATH) {
    $vsEditions = @("Community", "Professional", "Enterprise", "BuildTools")
    $vsVersions = @("18", "2022", "17", "2019")
    $vsClangPaths = foreach ($ver in $vsVersions) {
        foreach ($ed in $vsEditions) {
            "C:\Program Files\Microsoft Visual Studio\$ver\$ed\VC\Tools\Llvm\x64\bin"
        }
    }
    $vsClangPaths += "C:\Program Files\LLVM\bin"

    $found = $vsClangPaths | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($found) {
        $env:LIBCLANG_PATH = $found
        Write-Info "LIBCLANG_PATH=$env:LIBCLANG_PATH"
    } else {
        Write-Warn "Could not find libclang on Windows - bindgen may fail. Set LIBCLANG_PATH manually."
    }
}

# Clean dist
if (Test-Path "dist") {
    Write-Info "Cleaning previous build artifacts"
    Remove-Item -Recurse -Force "dist"
}

# Ensure Rust target is installed
Ensure-RustTarget $target

# Build all plugins in one cargo invocation
$packageArgs = $plugins | ForEach-Object { "-p"; $PluginRegistry[$_].Package }
$buildArgs = @("build", "--target", $target) + $packageArgs
if ($Profile -ne "debug") { $buildArgs += @("--profile", $Profile) }
if ($Verbose)             { $buildArgs += "--verbose" }

Write-Info "Running: cargo $($buildArgs -join ' ')"
& cargo @buildArgs
if ($LASTEXITCODE -ne 0) {
    Write-Err "cargo build failed"
    exit 1
}

# Copy artifacts to dist/
foreach ($p in $plugins) {
    Copy-PluginArtifacts $p $target $profileDir
}

Write-Success "Build complete! Check the dist\ directory for outputs."
Write-Host ""
Write-Info "Built artifacts:"
Get-ChildItem -Recurse dist -Filter "*.dll" | ForEach-Object { Write-Host "  $($_.FullName)" }
