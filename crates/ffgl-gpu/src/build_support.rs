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

        // --- Transpile each entry point to GLSL 4.60 ---
        // Always generated (pure Rust, no platform dependency) so cross-compilation works.
        //
        // Build GLSL binding map: WGSL (group, binding) → GL binding point.
        // Identity mapping so naga emits `layout(binding = N)` qualifiers,
        // allowing dispatch code to bind to matching texture/image units.
        let glsl_binding_map = build_glsl_binding_map(&module);

        for entry in file_entries.iter() {
            let mut glsl_source = String::new();
            let glsl_options = naga::back::glsl::Options {
                version: naga::back::glsl::Version::Desktop(460),
                writer_flags: naga::back::glsl::WriterFlags::empty(),
                binding_map: glsl_binding_map.clone(),
                zero_initialize_workgroup_memory: true,
            };
            let pipeline_options = naga::back::glsl::PipelineOptions {
                shader_stage: entry.stage.to_naga(),
                entry_point: entry.entry_point.to_string(),
                multiview: None,
            };
            let mut writer = naga::back::glsl::Writer::new(
                &mut glsl_source,
                &module,
                &info,
                &glsl_options,
                &pipeline_options,
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

/// Build a GLSL binding map from a parsed WGSL module.
///
/// Identity mapping: each WGSL `@group(G) @binding(B)` gets GL binding point B.
/// This causes naga to emit `layout(binding = B)` qualifiers in GLSL, so the
/// dispatch code can bind textures/images to matching units without name lookups.
fn build_glsl_binding_map(
    module: &naga::Module,
) -> std::collections::BTreeMap<naga::ResourceBinding, u8> {
    let mut map = std::collections::BTreeMap::new();
    for (_, var) in module.global_variables.iter() {
        if let Some(rb) = &var.binding {
            map.insert(rb.clone(), rb.binding as u8);
        }
    }
    map
}

/// Load embedded Metal shader library compiled by
/// [`compile_wgsl_shaders`].
///
/// Expands to `include_bytes!(concat!(env!("OUT_DIR"), "/shaders.metallib"))`.
#[macro_export]
macro_rules! include_metallib {
    () => {
        include_bytes!(concat!(env!("OUT_DIR"), "/shaders.metallib"))
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
