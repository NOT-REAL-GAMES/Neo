#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DrawBackendSelection {
    DrawExecution,
    DrawExecutionDrawAll,
    CudaTiled,
}

impl DrawBackendSelection {
    fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "primary-neo" | "cuda-tiled" | "cuda" => Ok(Self::CudaTiled),
            "draw-execution" | "hardware-raster" | "raster" | "raster-culled" => {
                Ok(Self::DrawExecution)
            }
            "hardware-raster-draw-all"
            | "hardware-raster-baseline"
            | "raster-draw-all"
            | "raster-baseline" => Ok(Self::DrawExecutionDrawAll),
            _ => anyhow::bail!(
                "unknown draw backend `{value}`; expected primary-neo, cuda-tiled, draw-execution, hardware-raster, hardware-raster-draw-all, or compatibility aliases cuda, raster, raster-culled, raster-draw-all, raster-baseline"
            ),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::CudaTiled => "cuda-tiled",
            Self::DrawExecution => "draw-execution",
            Self::DrawExecutionDrawAll => "hardware-raster-draw-all",
        }
    }

    fn defaults(self) -> Vec<String> {
        match self {
            Self::DrawExecution => draw_execution_defaults(),
            Self::DrawExecutionDrawAll => draw_execution_draw_all_defaults(),
            Self::CudaTiled => cuda_tiled_defaults(),
        }
    }
}

fn main() -> anyhow::Result<()> {
    let (draw_backend, passthrough) = parse_wrapper_args(std::env::args().skip(1))?;
    let mut args = draw_backend.defaults();
    args.extend(passthrough);
    neo_live_window::run_from_args(args)
}

fn parse_wrapper_args(
    args: impl IntoIterator<Item = String>,
) -> anyhow::Result<(DrawBackendSelection, Vec<String>)> {
    let mut draw_backend = DrawBackendSelection::CudaTiled;
    let mut passthrough = Vec::new();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        if arg == "--draw-backend" || arg == "--renderer" {
            let value = args
                .next()
                .ok_or_else(|| anyhow::anyhow!("{arg} requires a draw backend name"))?;
            let selected = DrawBackendSelection::parse(&value)?;
            let _selected_label = selected.label();
            draw_backend = selected;
        } else {
            passthrough.push(arg);
        }
    }
    Ok((draw_backend, passthrough))
}

fn draw_execution_defaults() -> Vec<String> {
    let mut args = draw_execution_draw_all_defaults();
    replace_arg_value(&mut args, "--draw-policy", "compute-culled");
    replace_arg_value(&mut args, "--draw-depth", "auto");
    args.extend([
        "--cull-order".to_string(),
        "stable-dense".to_string(),
        "--visibility".to_string(),
        "projected-size".to_string(),
        "--min-projected-pixels".to_string(),
        "0.85".to_string(),
    ]);
    args
}

fn draw_execution_draw_all_defaults() -> Vec<String> {
    [
        "examples/stress-quads/hardware_raster.neo",
        "--title",
        "Neo 3D Quad Stress (Draw Execution)",
        "--width",
        "3440",
        "--height",
        "1369",
        "--mode",
        "draw-stress",
        "--presenter",
        "d3d12-interop",
        "--kernel-target-fps",
        "1000",
        "--present-target-fps",
        "1000",
        "--max-inflight",
        "8",
        "--present-ring",
        "8",
        "--instance-grid",
        "256x256x128",
        "--draw-policy",
        "draw-all",
        "--draw-depth",
        "off",
        "--no-hot-reload",
        "--interop-fallback",
        "fail",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn replace_arg_value(args: &mut [String], flag: &str, value: &str) {
    if let Some(index) = args.iter().position(|arg| arg == flag)
        && let Some(slot) = args.get_mut(index + 1)
    {
        *slot = value.to_string();
    }
}

fn cuda_tiled_defaults() -> Vec<String> {
    [
        "examples/stress-quads/three_d_instances.neo",
        "--title",
        "Neo 3D Quad Stress (CUDA)",
        "--width",
        "3440",
        "--height",
        "1369",
        "--mode",
        "instance-stress",
        "--presenter",
        "d3d12-interop",
        "--kernel-target-fps",
        "1000",
        "--present-target-fps",
        "1000",
        "--max-inflight",
        "8",
        "--present-ring",
        "8",
        "--instance-grid",
        "256x256x128",
        "--instance-stress-variant",
        "tiled",
        "--instance-layout",
        "aosoa32",
        "--instance-debug-view",
        "off",
        "--no-hot-reload",
        "--interop-fallback",
        "fail",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_primary_neo_draw_backend() {
        let (draw_backend, passthrough) = parse_wrapper_args(std::iter::empty()).unwrap();
        assert_eq!(draw_backend, DrawBackendSelection::CudaTiled);
        assert_eq!(draw_backend.label(), "cuda-tiled");
        assert!(passthrough.is_empty());
        let args = draw_backend.defaults();
        assert!(args.contains(&"examples/stress-quads/three_d_instances.neo".to_string()));
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--mode", "instance-stress"])
        );
    }

    #[test]
    fn accepts_cuda_tiled_draw_backend_escape_hatch() {
        let (draw_backend, passthrough) = parse_wrapper_args(
            ["--draw-backend", "cuda-tiled"]
                .into_iter()
                .map(String::from),
        )
        .unwrap();
        assert_eq!(draw_backend, DrawBackendSelection::CudaTiled);
        assert_eq!(draw_backend.label(), "cuda-tiled");
        assert!(passthrough.is_empty());
        let args = draw_backend.defaults();
        assert!(args.contains(&"examples/stress-quads/three_d_instances.neo".to_string()));
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--mode", "instance-stress"])
        );
    }

    #[test]
    fn accepts_primary_neo_draw_backend_alias() {
        let (draw_backend, passthrough) = parse_wrapper_args(
            ["--draw-backend", "primary-neo"]
                .into_iter()
                .map(String::from),
        )
        .unwrap();
        assert_eq!(draw_backend, DrawBackendSelection::CudaTiled);
        assert!(passthrough.is_empty());
    }

    #[test]
    fn accepts_legacy_renderer_flag_and_cuda_alias() {
        let (draw_backend, passthrough) =
            parse_wrapper_args(["--renderer", "cuda"].into_iter().map(String::from)).unwrap();
        assert_eq!(draw_backend, DrawBackendSelection::CudaTiled);
        assert!(passthrough.is_empty());
    }

    #[test]
    fn forwards_live_window_overrides_after_defaults() {
        let (draw_backend, passthrough) = parse_wrapper_args(
            [
                "--draw-backend",
                "hardware-raster",
                "--width",
                "1280",
                "--height",
                "720",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();
        assert_eq!(draw_backend, DrawBackendSelection::DrawExecution);
        assert_eq!(draw_backend.label(), "draw-execution");
        assert_eq!(passthrough, ["--width", "1280", "--height", "720"]);
    }

    #[test]
    fn draw_execution_defaults_to_compute_culled_projected_visibility() {
        let args = draw_execution_defaults();
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--title", "Neo 3D Quad Stress (Draw Execution)"])
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--mode", "draw-stress"])
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--draw-policy", "compute-culled"])
        );
        assert!(args.windows(2).any(|pair| pair == ["--draw-depth", "auto"]));
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--cull-order", "stable-dense"])
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--visibility", "projected-size"])
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--min-projected-pixels", "0.85"])
        );
    }

    #[test]
    fn accepts_raster_culled_alias_for_optimized_draw_execution_path() {
        let (draw_backend, passthrough) = parse_wrapper_args(
            ["--draw-backend", "raster-culled"]
                .into_iter()
                .map(String::from),
        )
        .unwrap();
        assert_eq!(draw_backend, DrawBackendSelection::DrawExecution);
        assert!(passthrough.is_empty());
    }

    #[test]
    fn accepts_draw_execution_alias_for_optimized_draw_execution_path() {
        let (draw_backend, passthrough) = parse_wrapper_args(
            ["--draw-backend", "draw-execution"]
                .into_iter()
                .map(String::from),
        )
        .unwrap();
        assert_eq!(draw_backend, DrawBackendSelection::DrawExecution);
        assert_eq!(draw_backend.label(), "draw-execution");
        assert!(passthrough.is_empty());
    }

    #[test]
    fn accepts_explicit_hardware_raster_draw_all_baseline() {
        let (draw_backend, passthrough) = parse_wrapper_args(
            ["--draw-backend", "hardware-raster-draw-all"]
                .into_iter()
                .map(String::from),
        )
        .unwrap();
        assert_eq!(draw_backend, DrawBackendSelection::DrawExecutionDrawAll);
        assert_eq!(draw_backend.label(), "hardware-raster-draw-all");
        assert!(passthrough.is_empty());
        let args = draw_backend.defaults();
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--draw-policy", "draw-all"])
        );
        assert!(args.windows(2).any(|pair| pair == ["--draw-depth", "off"]));
        assert!(!args.iter().any(|arg| arg == "--visibility"));
    }

    #[test]
    fn rejects_unknown_draw_backend() {
        let err = parse_wrapper_args(["--draw-backend", "software"].into_iter().map(String::from))
            .unwrap_err()
            .to_string();
        assert!(err.contains(
            "expected primary-neo, cuda-tiled, draw-execution, hardware-raster, hardware-raster-draw-all"
        ));
    }

    #[test]
    fn missing_draw_backend_value_reports_the_flag_name() {
        let err = parse_wrapper_args(["--draw-backend"].into_iter().map(String::from))
            .unwrap_err()
            .to_string();
        assert!(err.contains("--draw-backend requires a draw backend name"));
    }
}
