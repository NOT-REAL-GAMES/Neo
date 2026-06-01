use neo_app::{InteropFallback, MeshBuilder, NeoApp, Presenter};

fn main() -> anyhow::Result<()> {
    NeoApp::new()
        .window("Neo Friendly Demo", 1280, 720)
        .presenter(Presenter::D3d12Interop)
        .interop_fallback(InteropFallback::Fail)
        .kernel("image", "examples/live-window/live.neo")
        .kernel("raster", "examples/mesh-buffer/raster.neo")
        .mesh("quad", MeshBuilder::quad().colored())
        .target_fps(240.0)
        .max_inflight(8)
        .present_ring(8)
        .hot_reload(true)
        .run()
}
