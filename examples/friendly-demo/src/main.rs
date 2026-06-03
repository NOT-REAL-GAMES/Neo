use neo_app::{GeometryBuilder, InstanceGrid, InteropFallback, NeoApp, Presenter, TargetSpec};

fn main() -> anyhow::Result<()> {
    let app = friendly_demo_app();

    let _draw_graph = app.draw_graph()?;
    app.run()
}

fn friendly_demo_app() -> NeoApp {
    NeoApp::new()
        .window("Neo Friendly Demo", 1280, 720)
        .presenter(Presenter::D3d12Interop)
        .interop_fallback(InteropFallback::Fail)
        .geometry_stream("quad", GeometryBuilder::quad().colored())
        .instance_stream_aosoa32("instances", InstanceGrid::new(128, 128, 64))
        .cuda_tiled_material_kernel(
            "quad-material",
            "instance_raster",
            "examples/stress-quads/three_d_instances.neo",
        )
        .draw_cuda_tiled(
            "main",
            "quad",
            "instances",
            "quad-material",
            TargetSpec::window(),
        )
        .target_fps(240.0)
        .max_inflight(8)
        .present_ring(8)
        .hot_reload(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use neo_app::DrawPolicy;

    #[test]
    fn friendly_demo_uses_modern_cuda_tiled_draw_contract() {
        let app = friendly_demo_app();
        let draw = app.draw_spec("main").unwrap();
        assert_eq!(draw.geometry.mesh, "quad");
        assert_eq!(draw.instances.as_ref().unwrap().name, "instances");
        assert_eq!(draw.material, "quad-material");
        assert_eq!(draw.target, TargetSpec::window());
        assert_eq!(draw.policy, DrawPolicy::CudaTiled);

        let cuda_plan = app.cuda_run_plan().unwrap().unwrap();
        assert_eq!(cuda_plan.entrypoint, "instance_raster");
        assert_eq!(cuda_plan.instances.grid, InstanceGrid::new(128, 128, 64));
        assert!(app.draw_execution_run_plan().unwrap().is_none());

        let args = app.live_window_args();
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--mode", "instance-stress"])
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--instance-stress-variant", "tiled"])
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--instance-grid", "128x128x64"])
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--instance-layout", "aosoa32"])
        );
        assert!(!args.iter().any(|arg| arg == "--raster-draw-policy"));
    }
}
