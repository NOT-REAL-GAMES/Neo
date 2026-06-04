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
