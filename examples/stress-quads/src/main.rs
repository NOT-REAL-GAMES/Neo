use neo_app::{InteropFallback, NeoApp, Presenter};

fn main() -> anyhow::Result<()> {
    NeoApp::new()
        .window("Neo Quad Stress", 3440, 1369)
        .presenter(Presenter::D3d12Interop)
        .interop_fallback(InteropFallback::Fail)
        .kernel("image", "examples/stress-quads/million_quads.neo")
        .target_fps(240.0)
        .max_inflight(8)
        .present_ring(8)
        .hot_reload(false)
        .run()
}
