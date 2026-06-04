#[cfg(windows)]
fn configure_nvrtc_search_path(diagnostics: &RuntimeDiagnostics) {
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
fn configure_nvrtc_search_path(_diagnostics: &RuntimeDiagnostics) {}

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
