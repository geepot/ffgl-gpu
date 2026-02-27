//! Shader compilation helpers for consumer `build.rs` scripts.
//!
//! These functions compile shaders at build time and emit the appropriate
//! `cargo:rerun-if-changed` directives.
//!
//! # Usage
//!
//! Write shaders in WGSL and use [`compile_wgsl_shaders`] in your `build.rs`:
//!
//! ```rust,ignore
//! use ffgl_gpu::build_support::{compile_wgsl_shaders, WgslEntry, ShaderStage};
//!
//! fn main() {
//!     compile_wgsl_shaders(
//!         std::path::Path::new("shaders"),
//!         &[WgslEntry {
//!             file: "passthrough.wgsl",
//!             entry_point: "passthrough",
//!             stage: ShaderStage::Compute,
//!         }],
//!     );
//! }
//! ```
//!
//! Then in your Rust source, load the compiled shaders:
//!
//! ```rust,ignore
//! // macOS: embedded metallib
//! let metallib = ffgl_gpu::include_metallib!();
//!
//! // Windows: embedded GLSL source
//! let glsl = ffgl_gpu::include_glsl_shader!("passthrough");
//! ```

use std::path::Path;

// ---------------------------------------------------------------------------
// WGSL transpilation (cross-platform)
// ---------------------------------------------------------------------------

/// Shader stage for WGSL entry points.
#[derive(Debug, Clone, Copy)]
pub enum ShaderStage {
    Compute,
    Vertex,
    Fragment,
}

impl ShaderStage {
    fn to_naga(self) -> naga::ShaderStage {
        match self {
            ShaderStage::Compute => naga::ShaderStage::Compute,
            ShaderStage::Vertex => naga::ShaderStage::Vertex,
            ShaderStage::Fragment => naga::ShaderStage::Fragment,
        }
    }
}

/// A WGSL shader entry point to compile.
pub struct WgslEntry {
    /// WGSL source file name (relative to the shader directory).
    pub file: &'static str,
    /// Entry point function name in the WGSL source.
    pub entry_point: &'static str,
    /// Shader stage.
    pub stage: ShaderStage,
}

/// Compile WGSL shaders via naga transpilation.
///
/// Parses each WGSL source file, validates it, then:
/// - **macOS:** Transpiles to Metal Shading Language, compiles `.metal` → `.air`
///   → `.metallib` via `xcrun`.
/// - **Windows:** Transpiles to GLSL 4.60, writes `<entry_point>.glsl` to
///   `OUT_DIR`.
///
/// Emits `cargo:rerun-if-changed` for each `.wgsl` file.
pub fn compile_wgsl_shaders(shader_dir: &Path, entries: &[WgslEntry]) {
    let out_dir = std::env::var("OUT_DIR").unwrap();

    // Group entries by source file to avoid re-parsing
    let mut files_to_entries: std::collections::BTreeMap<&str, Vec<&WgslEntry>> =
        std::collections::BTreeMap::new();
    for entry in entries {
        files_to_entries
            .entry(entry.file)
            .or_default()
            .push(entry);
    }

    #[cfg(target_os = "macos")]
    let mut metal_files: Vec<std::path::PathBuf> = Vec::new();

    for (file, file_entries) in &files_to_entries {
        let wgsl_path = shader_dir.join(file);
        let source = std::fs::read_to_string(&wgsl_path).unwrap_or_else(|e| {
            panic!("Failed to read WGSL source {wgsl_path:?}: {e}");
        });

        // Parse WGSL
        let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|e| {
            panic!("Failed to parse WGSL {wgsl_path:?}: {e}");
        });

        // Validate
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| {
            panic!("WGSL validation failed for {wgsl_path:?}: {e}");
        });

        // --- macOS: Transpile to Metal, write .metal file ---
        #[cfg(target_os = "macos")]
        {
            // Build binding map: auto-assign Metal indices per resource type.
            // Textures get sequential [[texture(N)]], buffers get [[buffer(N)]],
            // samplers get [[sampler(N)]].
            let binding_map = build_msl_binding_map(&module);

            let mut per_entry_point_map = std::collections::BTreeMap::new();
            for entry in file_entries.iter() {
                per_entry_point_map.insert(
                    entry.entry_point.to_string(),
                    naga::back::msl::EntryPointResources {
                        resources: binding_map.clone(),
                        push_constant_buffer: None,
                        sizes_buffer: None,
                    },
                );
            }

            let (msl_source, _) = naga::back::msl::write_string(
                &module,
                &info,
                &naga::back::msl::Options {
                    lang_version: (2, 0),
                    per_entry_point_map,
                    fake_missing_bindings: false,
                    ..Default::default()
                },
                &naga::back::msl::PipelineOptions::default(),
            )
            .unwrap_or_else(|e| {
                panic!("MSL transpilation failed for {wgsl_path:?}: {e}");
            });

            let stem = std::path::Path::new(file)
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap();
            let metal_path = format!("{out_dir}/{stem}.metal");
            std::fs::write(&metal_path, &msl_source).unwrap_or_else(|e| {
                panic!("Failed to write {metal_path}: {e}");
            });
            metal_files.push(std::path::PathBuf::from(metal_path));
        }

        // --- Windows: Transpile each entry point to GLSL 4.60 ---
        #[cfg(target_os = "windows")]
        {
            for entry in file_entries {
                let mut glsl_source = String::new();
                let mut writer = naga::back::glsl::Writer::new(
                    &mut glsl_source,
                    &module,
                    &info,
                    &naga::back::glsl::Options {
                        version: naga::back::glsl::Version::Desktop(460),
                        writer_flags: naga::back::glsl::WriterFlags::empty(),
                        binding_map: Default::default(),
                        zero_initialize_workgroup_memory: true,
                    },
                    &naga::back::glsl::PipelineOptions {
                        shader_stage: entry.stage.to_naga(),
                        entry_point: entry.entry_point.to_string(),
                        multiview: None,
                    },
                    naga::proc::BoundsCheckPolicies::default(),
                )
                .unwrap_or_else(|e| {
                    panic!(
                        "GLSL writer init failed for {}:{}: {e}",
                        file, entry.entry_point
                    );
                });

                writer.write().unwrap_or_else(|e| {
                    panic!(
                        "GLSL transpilation failed for {}:{}: {e}",
                        file, entry.entry_point
                    );
                });

                let glsl_path = format!("{out_dir}/{}.glsl", entry.entry_point);
                std::fs::write(&glsl_path, &glsl_source).unwrap_or_else(|e| {
                    panic!("Failed to write {glsl_path}: {e}");
                });
            }
        }

        // Suppress unused variable warning when neither Mac nor Windows
        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        let _ = file_entries;
    }

    // --- macOS: Compile .metal → .air → .metallib ---
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;

        if metal_files.is_empty() {
            return;
        }

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

        let metallib_path = format!("{out_dir}/shaders.metallib");
        let mut cmd = Command::new("xcrun");
        cmd.args(["-sdk", "macosx", "metallib"]);
        for air in &air_files {
            cmd.arg(air);
        }
        cmd.args(["-o", &metallib_path]);
        let status = cmd.status().expect("Failed to run xcrun metallib linker");
        assert!(status.success(), "Metal library linking failed");
    }

    // Re-run if any WGSL source changes
    if let Ok(dir_entries) = std::fs::read_dir(shader_dir) {
        for dir_entry in dir_entries.flatten() {
            let path = dir_entry.path();
            if path.extension().is_some_and(|ext| ext == "wgsl") {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
}

/// Build a Metal binding map from a parsed WGSL module.
///
/// Inspects global variables for resource bindings and assigns sequential
/// Metal indices per resource type: textures get `[[texture(N)]]`, buffers
/// get `[[buffer(N)]]`, and samplers get `[[sampler(N)]]`.
#[cfg(target_os = "macos")]
fn build_msl_binding_map(
    module: &naga::Module,
) -> std::collections::BTreeMap<naga::ResourceBinding, naga::back::msl::BindTarget> {
    let mut map = std::collections::BTreeMap::new();
    let mut texture_slot: u8 = 0;
    let mut buffer_slot: u8 = 0;
    let mut sampler_slot: u8 = 0;

    // Collect and sort by (group, binding) for deterministic ordering
    let mut bindings: Vec<_> = module
        .global_variables
        .iter()
        .filter_map(|(_, var)| {
            let rb = var.binding.as_ref()?;
            Some((rb.clone(), var.space, module.types[var.ty].inner.clone()))
        })
        .collect();
    bindings.sort_by_key(|(rb, _, _)| (rb.group, rb.binding));

    for (rb, _space, inner) in bindings {
        let target = match inner {
            naga::TypeInner::Image { .. } => {
                let slot = texture_slot;
                texture_slot += 1;
                naga::back::msl::BindTarget {
                    texture: Some(slot),
                    ..Default::default()
                }
            }
            naga::TypeInner::Sampler { .. } => {
                let slot = sampler_slot;
                sampler_slot += 1;
                naga::back::msl::BindTarget {
                    sampler: Some(naga::back::msl::BindSamplerTarget::Resource(slot)),
                    ..Default::default()
                }
            }
            _ => {
                // Uniform/storage buffers and other resources
                let slot = buffer_slot;
                buffer_slot += 1;
                naga::back::msl::BindTarget {
                    buffer: Some(slot),
                    ..Default::default()
                }
            }
        };
        map.insert(rb, target);
    }

    map
}

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

/// Load an embedded GLSL shader source transpiled from WGSL by
/// [`compile_wgsl_shaders`].
///
/// The `$name` argument is the entry point name used during compilation.
///
/// Expands to `include_str!(concat!(env!("OUT_DIR"), "/", $name, ".glsl"))`.
#[macro_export]
macro_rules! include_glsl_shader {
    ($name:literal) => {
        include_str!(concat!(env!("OUT_DIR"), "/", $name, ".glsl"))
    };
}
