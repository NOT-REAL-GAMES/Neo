use std::{
    any::Any,
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Path, PathBuf},
    sync::Arc,
};

use cudarc::{
    driver::{CudaContext, DriverError, sys},
    nvrtc::{Ptx, compile_ptx, result as nvrtc_result},
};

use crate::{Context, RuntimeError};

pub fn nvrtc_available() -> bool {
    RuntimeDiagnostics::collect().nvrtc_loadable
}

pub(crate) fn compile_cuda_image_checked(
    ctx: &Context,
    cuda_source: &str,
    diagnostics: &RuntimeDiagnostics,
) -> Result<Ptx, RuntimeError> {
    match compile_cubin_for_context_checked(ctx, cuda_source, diagnostics) {
        Ok(cubin) => return Ok(Ptx::from_binary(cubin)),
        Err(err) => {
            let _ = err;
        }
    }
    compile_ptx_checked(cuda_source, diagnostics)
}

fn compile_ptx_checked(
    cuda_source: &str,
    diagnostics: &RuntimeDiagnostics,
) -> Result<Ptx, RuntimeError> {
    let panic_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = catch_unwind(AssertUnwindSafe(|| compile_ptx(cuda_source)));
    std::panic::set_hook(panic_hook);
    result
        .map_err(|payload| RuntimeError::Nvrtc(nvrtc_panic_help(payload, diagnostics)))?
        .map_err(|err| RuntimeError::Nvrtc(err.to_string()))
}

fn compile_cubin_for_context_checked(
    ctx: &Context,
    cuda_source: &str,
    diagnostics: &RuntimeDiagnostics,
) -> Result<Vec<u8>, RuntimeError> {
    let (major, minor) = ctx.inner.compute_capability()?;
    let arch = format!("sm_{major}{minor}");
    compile_cubin_checked(cuda_source, &arch, diagnostics)
}

fn compile_cubin_checked(
    cuda_source: &str,
    arch: &str,
    diagnostics: &RuntimeDiagnostics,
) -> Result<Vec<u8>, RuntimeError> {
    let panic_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = catch_unwind(AssertUnwindSafe(|| compile_cubin(cuda_source, arch)));
    std::panic::set_hook(panic_hook);
    result.map_err(|payload| RuntimeError::Nvrtc(nvrtc_panic_help(payload, diagnostics)))?
}

fn compile_cubin(cuda_source: &str, arch: &str) -> Result<Vec<u8>, RuntimeError> {
    use std::ffi::{CStr, CString};

    let src =
        CString::new(cuda_source.as_bytes()).expect("CUDA source cannot contain null terminators");
    let program = nvrtc_result::create_program(src.as_c_str(), None)
        .map_err(|err| RuntimeError::Nvrtc(err.to_string()))?;
    let options = vec![format!("--gpu-architecture={arch}")];
    let compile_result = unsafe { nvrtc_result::compile_program(program, &options) };
    if let Err(err) = compile_result {
        let log = unsafe { nvrtc_result::get_program_log(program) }
            .ok()
            .map(|raw| {
                unsafe { CStr::from_ptr(raw.as_ptr()) }
                    .to_string_lossy()
                    .to_string()
            })
            .unwrap_or_default();
        unsafe {
            let _ = nvrtc_result::destroy_program(program);
        }
        return Err(RuntimeError::Nvrtc(format!(
            "native CUBIN compile failed for {arch}: {err}\n{log}"
        )));
    }
    let cubin = unsafe { nvrtc_get_cubin(program) };
    unsafe {
        let _ = nvrtc_result::destroy_program(program);
    }
    cubin
}

unsafe fn nvrtc_get_cubin(
    program: cudarc::nvrtc::sys::nvrtcProgram,
) -> Result<Vec<u8>, RuntimeError> {
    let mut size = 0usize;
    unsafe { cudarc::nvrtc::sys::nvrtcGetCUBINSize(program, &mut size).result() }
        .map_err(|err| RuntimeError::Nvrtc(err.to_string()))?;
    let mut cubin = vec![0u8; size];
    unsafe { cudarc::nvrtc::sys::nvrtcGetCUBIN(program, cubin.as_mut_ptr().cast()).result() }
        .map_err(|err| RuntimeError::Nvrtc(err.to_string()))?;
    Ok(cubin)
}

pub(crate) fn load_cuda_module_checked(
    ctx: &Context,
    image: Ptx,
) -> Result<Arc<cudarc::driver::CudaModule>, RuntimeError> {
    ctx.inner
        .load_module(image)
        .map_err(|err| unsupported_ptx_error(err).unwrap_or(RuntimeError::Driver(err)))
}

fn unsupported_ptx_error(err: DriverError) -> Option<RuntimeError> {
    (err.0 == sys::CUresult::CUDA_ERROR_UNSUPPORTED_PTX_VERSION).then(|| {
        RuntimeError::Nvrtc(
            "CUDA driver rejected the compiled PTX because it was produced by a newer CUDA Toolkit than this driver supports. Update the NVIDIA driver, install a CUDA Toolkit matching the driver's reported CUDA version, or use Neo's native CUBIN path for the current GPU.".to_string(),
        )
    })
}

fn nvrtc_panic_help(payload: Box<dyn Any + Send>, diagnostics: &RuntimeDiagnostics) -> String {
    let panic_message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&'static str>().copied())
        .unwrap_or("cudarc panicked while loading NVRTC");
    format!("{panic_message}\n\n{}", diagnostics.nvrtc_loader_help())
}

#[cfg(feature = "cuda-12060")]
pub(crate) fn expected_cuda_build_label() -> &'static str {
    "CUDA 12.6"
}

#[cfg(feature = "cuda-13000")]
pub(crate) fn expected_cuda_build_label() -> &'static str {
    "CUDA 13"
}

#[cfg(windows)]
fn expected_cuda_path_env_keys() -> &'static [&'static str] {
    #[cfg(feature = "cuda-12060")]
    {
        &["CUDA_PATH_V12_6", "CUDA_PATH_V12_60"]
    }
    #[cfg(feature = "cuda-13000")]
    {
        &["CUDA_PATH_V13_0"]
    }
}

#[cfg(windows)]
fn active_cuda_path_env_prefix() -> &'static str {
    #[cfg(feature = "cuda-12060")]
    {
        "CUDA_PATH_V12_"
    }
    #[cfg(feature = "cuda-13000")]
    {
        "CUDA_PATH_V13_"
    }
}

#[cfg(windows)]
fn compatible_nvrtc_names() -> &'static [&'static str] {
    #[cfg(feature = "cuda-12060")]
    {
        &[
            "nvrtc64_120_0.dll",
            "nvrtc64_120.dll",
            "nvrtc64_12.dll",
            "nvrtc64.dll",
            "nvrtc.dll",
        ]
    }
    #[cfg(feature = "cuda-13000")]
    {
        &[
            "nvrtc64_130_0.dll",
            "nvrtc64_130.dll",
            "nvrtc64_13.dll",
            "nvrtc64.dll",
            "nvrtc.dll",
        ]
    }
}

#[cfg(windows)]
fn nvrtc_candidates() -> Vec<PathBuf> {
    let names = compatible_nvrtc_names();
    let mut dirs = Vec::new();

    for key in expected_cuda_path_env_keys() {
        if let Some(root) = std::env::var_os(key) {
            push_cuda_root_bin_dirs(&mut dirs, PathBuf::from(root));
        }
    }
    for key in ["CUDA_PATH", "CUDA_HOME"] {
        if let Some(root) = std::env::var_os(key) {
            push_cuda_root_bin_dirs(&mut dirs, PathBuf::from(root));
        }
    }

    let mut active_versioned_roots = std::env::vars_os()
        .filter_map(|(key, root)| {
            let key = key.to_string_lossy();
            let is_exact = expected_cuda_path_env_keys()
                .iter()
                .any(|expected| key == *expected);
            (key.starts_with(active_cuda_path_env_prefix()) && !is_exact)
                .then(|| PathBuf::from(root))
        })
        .collect::<Vec<_>>();
    active_versioned_roots.sort_by(|left, right| right.cmp(left));
    for root in active_versioned_roots {
        push_cuda_root_bin_dirs(&mut dirs, root);
    }

    let mut toolkit_dirs = cuda_toolkit_bin_dirs();
    toolkit_dirs.sort_by(|left, right| right.cmp(left));
    for dir in toolkit_dirs {
        push_unique_path(&mut dirs, dir);
    }
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            push_unique_path(&mut dirs, dir);
        }
    }
    for dir in nvidia_app_nvrtc_dirs() {
        push_unique_path(&mut dirs, dir);
    }

    dirs.into_iter()
        .flat_map(|dir| names.iter().map(move |name| dir.join(name)))
        .collect()
}

#[cfg(windows)]
fn push_unique_path(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !paths.iter().any(|existing| existing == &path) {
        paths.push(path);
    }
}

#[cfg(windows)]
pub(crate) fn push_cuda_root_bin_dirs(paths: &mut Vec<PathBuf>, root: PathBuf) {
    let bin = root.join("bin");
    push_unique_path(paths, bin.join("x64"));
    push_unique_path(paths, bin);
}

#[cfg(windows)]
pub(crate) fn compatible_nvrtc_candidate(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let path_text = path.to_string_lossy().to_ascii_lowercase();
    #[cfg(feature = "cuda-12060")]
    if path_text.contains("\\cuda\\v13") || path_text.contains("/cuda/v13") {
        return false;
    }
    #[cfg(feature = "cuda-13000")]
    if path_text.contains("\\cuda\\v12") || path_text.contains("/cuda/v12") {
        return false;
    }
    compatible_nvrtc_names()
        .iter()
        .any(|compatible| name.eq_ignore_ascii_case(compatible))
}

#[cfg(not(windows))]
pub(crate) fn compatible_nvrtc_candidate(_path: &Path) -> bool {
    true
}

#[cfg(not(windows))]
fn nvrtc_candidates() -> Vec<PathBuf> {
    let names = [
        "libnvrtc.so",
        "libnvrtc.so.13",
        "libnvrtc.so.12",
        "libnvrtc.so.11",
        "libnvrtc.dylib",
    ];
    let mut dirs = vec![
        PathBuf::from("/usr/lib"),
        PathBuf::from("/usr/local/cuda/lib64"),
        PathBuf::from("/usr/local/cuda/lib"),
    ];
    if let Some(path) = std::env::var_os("LD_LIBRARY_PATH") {
        dirs.extend(std::env::split_paths(&path));
    }
    dirs.into_iter()
        .flat_map(|dir| names.iter().map(move |name| dir.join(name)))
        .collect()
}

fn expected_nvrtc_library_hint() -> String {
    #[cfg(windows)]
    {
        compatible_nvrtc_names().join(", ")
    }
    #[cfg(not(windows))]
    {
        format!(
            "an NVRTC shared library compatible with {}",
            expected_cuda_build_label()
        )
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeDiagnostics {
    pub cuda_driver_available: bool,
    pub cuda_driver_error: Option<String>,
    pub nvrtc_candidates: Vec<PathBuf>,
    pub nvrtc_found: Vec<PathBuf>,
    pub nvrtc_compatible: Vec<PathBuf>,
    pub nvrtc_loadable: bool,
}

impl RuntimeDiagnostics {
    pub fn collect() -> Self {
        let (cuda_driver_available, cuda_driver_error) = match CudaContext::new(0) {
            Ok(ctx) => {
                drop(ctx);
                (true, None)
            }
            Err(err) => (false, Some(format!("{err:?}"))),
        };
        let nvrtc_candidates = nvrtc_candidates();
        let nvrtc_found = nvrtc_candidates
            .iter()
            .filter(|candidate| candidate.exists())
            .cloned()
            .collect::<Vec<_>>();
        let nvrtc_compatible = nvrtc_found
            .iter()
            .filter(|candidate| compatible_nvrtc_candidate(candidate))
            .cloned()
            .collect::<Vec<_>>();
        let nvrtc_loadable = !nvrtc_compatible.is_empty();
        Self {
            cuda_driver_available,
            cuda_driver_error,
            nvrtc_candidates,
            nvrtc_found,
            nvrtc_compatible,
            nvrtc_loadable,
        }
    }

    pub fn nvrtc_help(&self) -> String {
        if !self.nvrtc_compatible.is_empty() {
            return format!(
                "NVRTC was found, but the dynamic loader could not use it.\n\n{}",
                self.nvrtc_loader_help()
            );
        }
        if !self.nvrtc_found.is_empty() {
            return format!(
                "NVRTC was found, but not an NVRTC compatible with this Neo {} build.\n\n{}",
                expected_cuda_build_label(),
                self.nvrtc_loader_help()
            );
        }
        format!(
            "NVRTC shared library was not found. Install the NVIDIA CUDA Toolkit for {} or add the directory containing {} to PATH.",
            expected_cuda_build_label(),
            expected_nvrtc_library_hint()
        )
    }

    pub fn nvrtc_loader_help(&self) -> String {
        if let Some(found) = self.nvrtc_compatible.first() {
            return format!(
                "Neo found NVRTC at {} for this Neo {} build and tried to register {} with the process DLL loader. If this still fails, launch Neo from a shell where that CUDA bin directory is on PATH, or set the matching CUDA_PATH/CUDA_PATH_V* environment variable to the CUDA Toolkit root before starting Neo.",
                found.display(),
                expected_cuda_build_label(),
                found.parent().unwrap_or_else(|| Path::new("")).display()
            );
        }
        if let Some(found) = self.nvrtc_found.first() {
            let checked = self
                .nvrtc_candidates
                .iter()
                .filter(|candidate| compatible_nvrtc_candidate(candidate))
                .take(16)
                .map(|candidate| format!("  - {}", candidate.display()))
                .collect::<Vec<_>>()
                .join("\n");
            let checked = if checked.is_empty() {
                format!(
                    "  - no {}-compatible candidate names were generated",
                    expected_cuda_build_label()
                )
            } else {
                checked
            };
            return format!(
                "Neo found NVRTC at {}, but this Neo {} build expects compatible NVRTC names such as {} from the matching CUDA Toolkit.\nSet the matching CUDA_PATH_V* or CUDA_PATH to your CUDA Toolkit root before starting Neo.\nChecked compatible candidates:\n{}",
                found.display(),
                expected_cuda_build_label(),
                expected_nvrtc_library_hint(),
                checked
            );
        }
        let checked = self
            .nvrtc_candidates
            .iter()
            .filter(|candidate| compatible_nvrtc_candidate(candidate))
            .take(16)
            .map(|candidate| format!("  - {}", candidate.display()))
            .collect::<Vec<_>>()
            .join("\n");
        if checked.is_empty() {
            format!(
                "Neo did not generate any NVRTC candidate paths compatible with this Neo {} build. Set the matching CUDA_PATH_V* or CUDA_PATH to your CUDA Toolkit root before starting Neo.",
                expected_cuda_build_label()
            )
        } else {
            format!(
                "Neo could not find an NVRTC DLL compatible with this Neo {} build. Set the matching CUDA_PATH_V* or CUDA_PATH to your CUDA Toolkit root before starting Neo.\nChecked compatible candidates:\n{checked}",
                expected_cuda_build_label()
            )
        }
    }
}

#[cfg(windows)]
pub(crate) fn configure_nvrtc_search_path(diagnostics: &RuntimeDiagnostics) {
    let Some(dir) = diagnostics
        .nvrtc_compatible
        .first()
        .and_then(|path| path.parent())
    else {
        return;
    };

    register_windows_dll_directory(dir);

    let Some(current_path) = std::env::var_os("PATH") else {
        // SAFETY: Neo is single-threaded at the point this is called by the CLI/runtime setup.
        unsafe {
            std::env::set_var("PATH", dir);
        }
        return;
    };

    let paths = std::env::split_paths(&current_path).collect::<Vec<_>>();
    if paths.iter().any(|path| path == dir) {
        return;
    }
    let mut new_paths = vec![dir.to_path_buf()];
    new_paths.extend(paths);
    if let Ok(joined) = std::env::join_paths(new_paths) {
        // SAFETY: Neo updates the process DLL search path before NVRTC is loaded.
        unsafe {
            std::env::set_var("PATH", joined);
        }
    }
}

#[cfg(windows)]
fn register_windows_dll_directory(dir: &Path) {
    use std::os::windows::ffi::OsStrExt;
    use windows::Win32::System::LibraryLoader::SetDllDirectoryW;
    use windows::core::PCWSTR;

    let wide = dir
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    unsafe {
        let _ = SetDllDirectoryW(PCWSTR(wide.as_ptr()));
    }
}

#[cfg(not(windows))]
pub(crate) fn configure_nvrtc_search_path(_diagnostics: &RuntimeDiagnostics) {}

#[cfg(windows)]
fn cuda_toolkit_bin_dirs() -> Vec<PathBuf> {
    let root = Path::new(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .flat_map(|entry| {
            let mut dirs = Vec::new();
            push_cuda_root_bin_dirs(&mut dirs, entry.path());
            dirs
        })
        .filter(|path| path.is_dir())
        .collect()
}

#[cfg(windows)]
fn nvidia_app_nvrtc_dirs() -> Vec<PathBuf> {
    [
        r"C:\Program Files\NVIDIA Corporation\NVIDIA Audio Effects SDK",
        r"C:\Program Files\Blackmagic Design\DaVinci Resolve",
    ]
    .into_iter()
    .map(PathBuf::from)
    .filter(|path| path.is_dir())
    .collect()
}
