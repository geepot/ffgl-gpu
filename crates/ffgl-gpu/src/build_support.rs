//! Shader compilation helpers for consumer `build.rs` scripts.
//!
//! These functions compile Metal (`.metal` -> `.metallib`) and HLSL
//! (`.hlsl` -> `.cso`) shaders at build time and emit the appropriate
//! `cargo:rerun-if-changed` directives.
//!
//! # Usage
//!
//! In your plugin crate's `build.rs`:
//!
//! ```rust,ignore
//! // macOS
//! ffgl_gpu::build_support::compile_metal_shaders(
//!     std::path::Path::new("src/shaders"),
//! );
//!
//! // Windows
//! ffgl_gpu::build_support::compile_hlsl_shaders(
//!     std::path::Path::new("src/shaders"),
//!     &[
//!         ffgl_gpu::build_support::HlslEntry {
//!             file: "compute.hlsl",
//!             entry_point: "main_cs",
//!             target: "cs_5_0",
//!         },
//!     ],
//! );
//! ```
//!
//! Then in your Rust source, load the compiled shaders:
//!
//! ```rust,ignore
//! // Metal
//! let metallib = ffgl_gpu::include_metallib!();
//!
//! // HLSL
//! let compute_shader = ffgl_gpu::include_hlsl_shader!("compute");
//! ```

use std::path::Path;

/// Compile Metal shaders from a directory.
///
/// Scans `shader_dir` for `.metal` files, compiles each to `.air` via
/// `xcrun -sdk macosx metal`, then links all `.air` files into a single
/// `shaders.metallib` in `OUT_DIR`.
///
/// Emits `cargo:rerun-if-changed` for each `.metal` and `.h` file found in
/// the shader directory.
#[cfg(target_os = "macos")]
pub fn compile_metal_shaders(shader_dir: &Path) {
    use std::process::Command;

    let out_dir = std::env::var("OUT_DIR").unwrap();

    // Collect all .metal files
    let metal_files: Vec<_> = match std::fs::read_dir(shader_dir) {
        Ok(entries) => entries
            .filter_map(|e| {
                let path = e.ok()?.path();
                if path.extension().is_some_and(|ext| ext == "metal") {
                    Some(path)
                } else {
                    None
                }
            })
            .collect(),
        Err(_) => {
            println!(
                "cargo:warning=No Metal shader directory found at {shader_dir:?}, \
                 skipping shader compilation"
            );
            return;
        }
    };

    if metal_files.is_empty() {
        println!(
            "cargo:warning=No .metal files found in {shader_dir:?}, \
             skipping shader compilation"
        );
        return;
    }

    // Compile each .metal to .air
    let mut air_files = Vec::new();
    for metal_file in &metal_files {
        let stem = metal_file.file_stem().unwrap().to_str().unwrap();
        let air_file = format!("{out_dir}/{stem}.air");

        let status = Command::new("xcrun")
            .args([
                "-sdk",
                "macosx",
                "metal",
                "-std=macos-metal2.0",
                "-mmacos-version-min=13.0",
                "-c",
                metal_file.to_str().unwrap(),
                "-I",
                shader_dir.to_str().unwrap(),
                "-o",
                &air_file,
            ])
            .status()
            .expect("Failed to run xcrun metal compiler. Is Xcode installed?");
        assert!(
            status.success(),
            "Metal shader compilation failed for {metal_file:?}"
        );
        air_files.push(air_file);
    }

    // Link all .air into a single .metallib
    let metallib_path = format!("{out_dir}/shaders.metallib");
    let mut cmd = Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metallib"]);
    for air in &air_files {
        cmd.arg(air);
    }
    cmd.args(["-o", &metallib_path]);
    let status = cmd.status().expect("Failed to run xcrun metallib linker");
    assert!(status.success(), "Metal library linking failed");

    // Re-run if shaders change
    for metal_file in &metal_files {
        println!("cargo:rerun-if-changed={}", metal_file.display());
    }
    // Also re-run if any header files change
    if let Ok(entries) = std::fs::read_dir(shader_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "h") {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
}

/// An HLSL shader entry point to compile.
pub struct HlslEntry {
    /// HLSL source file name (relative to the shader directory).
    pub file: &'static str,
    /// Entry point function name.
    pub entry_point: &'static str,
    /// Shader target profile (e.g. `"cs_5_0"`, `"vs_5_0"`, `"ps_5_0"`).
    pub target: &'static str,
}

/// Compile HLSL shaders from a directory.
///
/// Each [`HlslEntry`] specifies a source file, entry point, and target
/// profile. The compiled shader object (`.cso`) is written to `OUT_DIR` with
/// the entry point as the file name.
///
/// Emits `cargo:rerun-if-changed` for each `.hlsl` and `.hlsli` file found
/// in the shader directory.
#[cfg(target_os = "windows")]
pub fn compile_hlsl_shaders(shader_dir: &Path, entries: &[HlslEntry]) {
    use std::process::Command;

    if !shader_dir.is_dir() {
        println!(
            "cargo:warning=No HLSL shader directory found at {shader_dir:?}, \
             skipping shader compilation"
        );
        return;
    }

    let out_dir = std::env::var("OUT_DIR").unwrap();

    let fxc =
        find_fxc().expect("Could not find fxc.exe. Install Windows SDK or add fxc.exe to PATH.");

    for entry in entries {
        let input_path = shader_dir.join(entry.file);
        let output_path = format!("{out_dir}/{}.cso", entry.entry_point);

        let status = Command::new(&fxc)
            .args([
                "/T",
                entry.target,
                "/E",
                entry.entry_point,
                "/I",
                shader_dir.to_str().unwrap(),
                "/Fo",
                &output_path,
                "/nologo",
                "/O3",
                input_path.to_str().unwrap(),
            ])
            .status()
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to run fxc.exe for {}:{}: {e}",
                    entry.file, entry.entry_point
                )
            });
        assert!(
            status.success(),
            "HLSL compilation failed for {}:{}",
            entry.file,
            entry.entry_point
        );
    }

    // Re-run if any shader source changes
    if let Ok(entries) = std::fs::read_dir(shader_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path
                .extension()
                .is_some_and(|e| e == "hlsl" || e == "hlsli")
            {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
}

/// Find fxc.exe: check PATH first, then scan Windows SDK directories.
#[cfg(target_os = "windows")]
fn find_fxc() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;

    // Check if fxc.exe is already on PATH
    if let Ok(output) = std::process::Command::new("where")
        .arg("fxc.exe")
        .output()
    {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout);
            if let Some(line) = path.lines().next() {
                let p = PathBuf::from(line.trim());
                if p.exists() {
                    return Some(p);
                }
            }
        }
    }

    // Scan Windows SDK directories
    let program_files = std::env::var("ProgramFiles(x86)")
        .unwrap_or_else(|_| r"C:\Program Files (x86)".to_string());
    let sdk_base = PathBuf::from(program_files).join(r"Windows Kits\10\bin");

    if let Ok(entries) = std::fs::read_dir(&sdk_base) {
        let mut versions: Vec<_> = entries
            .filter_map(|e| {
                let path = e.ok()?.path();
                if path.is_dir() {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();
        // Sort descending to prefer newest SDK
        versions.sort_by(|a, b| b.cmp(a));

        for version_dir in versions {
            let fxc_path = version_dir.join("x64").join("fxc.exe");
            if fxc_path.exists() {
                return Some(fxc_path);
            }
        }
    }

    None
}

/// Load embedded Metal shader library compiled by
/// [`compile_metal_shaders`].
///
/// Expands to `include_bytes!(concat!(env!("OUT_DIR"), "/shaders.metallib"))`.
#[macro_export]
macro_rules! include_metallib {
    () => {
        include_bytes!(concat!(env!("OUT_DIR"), "/shaders.metallib"))
    };
}

/// Load an embedded HLSL compiled shader object (`.cso`) compiled by
/// [`compile_hlsl_shaders`].
///
/// The `$name` argument is the entry point name used during compilation.
///
/// Expands to `include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".cso"))`.
#[macro_export]
macro_rules! include_hlsl_shader {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".cso"))
    };
}
