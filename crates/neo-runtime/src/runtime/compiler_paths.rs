#[cfg(feature = "cuda-12060")]
fn expected_cuda_build_label() -> &'static str {
    "CUDA 12.6"
}

#[cfg(feature = "cuda-13000")]
fn expected_cuda_build_label() -> &'static str {
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
fn push_cuda_root_bin_dirs(paths: &mut Vec<PathBuf>, root: PathBuf) {
    let bin = root.join("bin");
    push_unique_path(paths, bin.join("x64"));
    push_unique_path(paths, bin);
}

#[cfg(windows)]
fn compatible_nvrtc_candidate(path: &Path) -> bool {
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
fn compatible_nvrtc_candidate(_path: &Path) -> bool {
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

