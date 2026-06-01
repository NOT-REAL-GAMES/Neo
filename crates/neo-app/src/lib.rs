use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use anyhow::{Result, bail};
use neo_runtime::{
    Context, IndexFormat, MeshBuffer, MeshBufferDesc, PrimitiveTopology, VertexAttribute,
    VertexFormat, VertexLayout, VertexSemantic,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Presenter {
    D3d12Interop,
    D3d12Upload,
    D3d11Upload,
    Gdi,
}

impl Presenter {
    fn live_window_name(self) -> &'static str {
        match self {
            Self::D3d12Interop => "d3d12-interop",
            Self::D3d12Upload => "d3d12",
            Self::D3d11Upload => "d3d11",
            Self::Gdi => "gdi",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteropFallback {
    NoInterop,
    Fail,
}

impl InteropFallback {
    fn live_window_name(self) -> &'static str {
        match self {
            Self::NoInterop => "no-interop",
            Self::Fail => "fail",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderPolicy {
    Auto,
    ForceRender,
    PauseWhenEmpty,
}

impl RenderPolicy {
    fn live_window_name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::ForceRender => "force-render",
            Self::PauseWhenEmpty => "pause-when-empty",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FpsCap {
    pub kernel: Option<f32>,
    pub present: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelSpec {
    pub entrypoint: String,
    pub path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct MeshSpec {
    pub name: String,
    pub source: MeshSource,
}

#[derive(Debug, Clone)]
pub enum MeshSource {
    Builder(MeshBuilder),
}

#[derive(Debug, Clone)]
pub struct NeoAppConfig {
    pub window: WindowConfig,
    pub presenter: Presenter,
    pub interop_fallback: InteropFallback,
    pub fps: FpsCap,
    pub hot_reload: bool,
    pub max_inflight: u32,
    pub present_ring: usize,
    pub render_policy: RenderPolicy,
}

pub struct NeoApp {
    config: NeoAppConfig,
    kernels: BTreeMap<String, KernelSpec>,
    meshes: BTreeMap<String, MeshSpec>,
    context: Option<Context>,
}

pub struct NeoAppParts {
    pub config: NeoAppConfig,
    pub kernels: BTreeMap<String, KernelSpec>,
    pub meshes: BTreeMap<String, MeshSpec>,
    pub context: Option<Context>,
}

impl Default for NeoApp {
    fn default() -> Self {
        Self::new()
    }
}

impl NeoApp {
    pub fn new() -> Self {
        Self {
            config: NeoAppConfig {
                window: WindowConfig {
                    title: "Neo".to_string(),
                    width: 960,
                    height: 540,
                },
                presenter: Presenter::D3d12Interop,
                interop_fallback: InteropFallback::NoInterop,
                fps: FpsCap {
                    kernel: None,
                    present: None,
                },
                hot_reload: true,
                max_inflight: 8,
                present_ring: 8,
                render_policy: RenderPolicy::Auto,
            },
            kernels: BTreeMap::new(),
            meshes: BTreeMap::new(),
            context: None,
        }
    }

    pub fn window(mut self, title: impl Into<String>, width: u32, height: u32) -> Self {
        self.config.window = WindowConfig {
            title: title.into(),
            width,
            height,
        };
        self
    }

    pub fn presenter(mut self, presenter: Presenter) -> Self {
        self.config.presenter = presenter;
        self
    }

    pub fn interop_fallback(mut self, fallback: InteropFallback) -> Self {
        self.config.interop_fallback = fallback;
        self
    }

    pub fn kernel(mut self, entrypoint: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        let entrypoint = entrypoint.into();
        self.kernels.insert(
            entrypoint.clone(),
            KernelSpec {
                entrypoint,
                path: path.into(),
            },
        );
        self
    }

    pub fn mesh(mut self, name: impl Into<String>, mesh: impl Into<MeshSource>) -> Self {
        let name = name.into();
        self.meshes.insert(
            name.clone(),
            MeshSpec {
                name,
                source: mesh.into(),
            },
        );
        self
    }

    pub fn target_fps(mut self, fps: f32) -> Self {
        self.config.fps.kernel = Some(fps);
        self.config.fps.present = Some(fps);
        self
    }

    pub fn kernel_target_fps(mut self, fps: f32) -> Self {
        self.config.fps.kernel = Some(fps);
        self
    }

    pub fn present_target_fps(mut self, fps: f32) -> Self {
        self.config.fps.present = Some(fps);
        self
    }

    pub fn hot_reload(mut self, enabled: bool) -> Self {
        self.config.hot_reload = enabled;
        self
    }

    pub fn max_inflight(mut self, max_inflight: u32) -> Self {
        self.config.max_inflight = max_inflight;
        self
    }

    pub fn present_ring(mut self, present_ring: usize) -> Self {
        self.config.present_ring = present_ring;
        self
    }

    pub fn render_policy(mut self, policy: RenderPolicy) -> Self {
        self.config.render_policy = policy;
        self
    }

    pub fn validate(&self) -> Result<()> {
        if self.config.window.width == 0 || self.config.window.height == 0 {
            bail!("window width and height must be greater than zero");
        }
        if self.config.max_inflight == 0 {
            bail!("max_inflight must be greater than zero");
        }
        if self.config.present_ring == 0 {
            bail!("present_ring must be greater than zero");
        }
        for fps in [self.config.fps.kernel, self.config.fps.present]
            .into_iter()
            .flatten()
        {
            if !fps.is_finite() || fps <= 0.0 {
                bail!("FPS caps must be finite and greater than zero");
            }
        }
        Ok(())
    }

    pub fn context(&mut self) -> Result<&Context> {
        if self.context.is_none() {
            self.context = Some(Context::new_default_device()?);
        }
        Ok(self.context.as_ref().expect("context was just initialized"))
    }

    pub fn context_mut(&mut self) -> Result<&mut Context> {
        if self.context.is_none() {
            self.context = Some(Context::new_default_device()?);
        }
        Ok(self.context.as_mut().expect("context was just initialized"))
    }

    pub fn kernel_spec(&self, entrypoint: &str) -> Option<&KernelSpec> {
        self.kernels.get(entrypoint)
    }

    pub fn mesh_spec(&self, name: &str) -> Option<&MeshSpec> {
        self.meshes.get(name)
    }

    pub fn presenter_kind(&self) -> Presenter {
        self.config.presenter
    }

    pub fn config(&self) -> &NeoAppConfig {
        &self.config
    }

    pub fn into_parts(self) -> NeoAppParts {
        NeoAppParts {
            config: self.config,
            kernels: self.kernels,
            meshes: self.meshes,
            context: self.context,
        }
    }

    pub fn run(self) -> Result<()> {
        let mut app = self;
        app.validate()?;
        let _friendly_meshes = app.build_meshes()?;
        neo_live_window::run_from_args(app.live_window_args())
    }

    pub fn build_meshes(&mut self) -> Result<BTreeMap<String, MeshBuffer>> {
        let builders: Vec<(String, MeshBuilder)> = self
            .meshes
            .iter()
            .map(|(name, spec)| match &spec.source {
                MeshSource::Builder(builder) => (name.clone(), builder.clone()),
            })
            .collect();
        if builders.is_empty() {
            return Ok(BTreeMap::new());
        }
        let context = self.context()?;
        builders
            .into_iter()
            .map(|(name, builder)| builder.build(context).map(|mesh| (name, mesh)))
            .collect()
    }

    pub fn live_window_args(&self) -> Vec<String> {
        let mesh_mode = self.meshes.contains_key("quad") && self.kernels.contains_key("raster");
        let source = if mesh_mode {
            self.kernels.get("raster").map(|kernel| kernel.path.clone())
        } else {
            self.kernels
                .get("image")
                .or_else(|| self.kernels.values().next())
                .map(|kernel| kernel.path.clone())
        }
        .unwrap_or_else(|| Path::new("examples/live-window/live.neo").to_path_buf());

        let mut args = vec![
            source.display().to_string(),
            "--title".to_string(),
            self.config.window.title.clone(),
            "--width".to_string(),
            self.config.window.width.to_string(),
            "--height".to_string(),
            self.config.window.height.to_string(),
            "--presenter".to_string(),
            self.config.presenter.live_window_name().to_string(),
            "--interop-fallback".to_string(),
            self.config.interop_fallback.live_window_name().to_string(),
            "--max-inflight".to_string(),
            self.config.max_inflight.to_string(),
            "--present-ring".to_string(),
            self.config.present_ring.to_string(),
            "--render-policy".to_string(),
            self.config.render_policy.live_window_name().to_string(),
        ];
        args.push("--mode".to_string());
        args.push(if mesh_mode {
            "mesh-demo".to_string()
        } else {
            "kernel-throughput".to_string()
        });
        if let Some(fps) = self.config.fps.kernel {
            args.push("--kernel-target-fps".to_string());
            args.push(fps.to_string());
        }
        if let Some(fps) = self.config.fps.present {
            args.push("--present-target-fps".to_string());
            args.push(fps.to_string());
        }
        args.push(if self.config.hot_reload {
            "--hot-reload".to_string()
        } else {
            "--no-hot-reload".to_string()
        });
        args
    }
}

#[derive(Debug, Clone)]
pub struct MeshBuilder {
    vertices: Vec<FriendlyVertex>,
    indices: Vec<u16>,
    layout: VertexLayout,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
struct FriendlyVertex {
    position: [f32; 3],
    color_bgra: u32,
}

impl MeshBuilder {
    pub fn quad() -> Self {
        Self {
            vertices: vec![
                FriendlyVertex {
                    position: [-0.65, -0.65, 0.0],
                    color_bgra: 0xffff_4040,
                },
                FriendlyVertex {
                    position: [0.65, -0.65, 0.0],
                    color_bgra: 0xff40_ff40,
                },
                FriendlyVertex {
                    position: [0.65, 0.65, 0.0],
                    color_bgra: 0xff40_40ff,
                },
                FriendlyVertex {
                    position: [-0.65, 0.65, 0.0],
                    color_bgra: 0xffff_ff40,
                },
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
            layout: friendly_vertex_layout(),
        }
    }

    pub fn cube() -> Self {
        let mut mesh = Self::quad();
        mesh.vertices = vec![
            FriendlyVertex {
                position: [-0.5, -0.5, -0.5],
                color_bgra: 0xffff_4040,
            },
            FriendlyVertex {
                position: [0.5, -0.5, -0.5],
                color_bgra: 0xff40_ff40,
            },
            FriendlyVertex {
                position: [0.5, 0.5, -0.5],
                color_bgra: 0xff40_40ff,
            },
            FriendlyVertex {
                position: [-0.5, 0.5, -0.5],
                color_bgra: 0xffff_ff40,
            },
            FriendlyVertex {
                position: [-0.5, -0.5, 0.5],
                color_bgra: 0xffff_40ff,
            },
            FriendlyVertex {
                position: [0.5, -0.5, 0.5],
                color_bgra: 0xff40_ffff,
            },
            FriendlyVertex {
                position: [0.5, 0.5, 0.5],
                color_bgra: 0xffffffff,
            },
            FriendlyVertex {
                position: [-0.5, 0.5, 0.5],
                color_bgra: 0xff80_80ff,
            },
        ];
        mesh.indices = vec![
            0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7,
            3, 3, 7, 4, 3, 4, 0,
        ];
        mesh
    }

    pub fn colored(self) -> Self {
        self
    }

    pub fn vertices(mut self, vertices: &[[f32; 3]]) -> Self {
        self.vertices = vertices
            .iter()
            .enumerate()
            .map(|(idx, position)| FriendlyVertex {
                position: *position,
                color_bgra: default_color(idx),
            })
            .collect();
        self
    }

    pub fn indices_u16(mut self, indices: &[u16]) -> Self {
        self.indices = indices.to_vec();
        self
    }

    pub fn layout(mut self, layout: VertexLayout) -> Self {
        self.layout = layout;
        self
    }

    pub fn desc(&self) -> MeshBufferDesc {
        MeshBufferDesc {
            vertex_count: self.vertices.len() as u32,
            vertex_layout: self.layout.clone(),
            index_format: if self.indices.is_empty() {
                IndexFormat::None
            } else {
                IndexFormat::U16
            },
            index_count: self.indices.len() as u32,
            topology: PrimitiveTopology::TriangleList,
        }
    }

    pub fn build(&self, context: &Context) -> Result<MeshBuffer> {
        MeshBuffer::upload_typed(context, self.desc(), &self.vertices, &self.indices)
            .map_err(Into::into)
    }
}

impl From<MeshBuilder> for MeshSource {
    fn from(value: MeshBuilder) -> Self {
        Self::Builder(value)
    }
}

fn friendly_vertex_layout() -> VertexLayout {
    VertexLayout {
        stride: std::mem::size_of::<FriendlyVertex>() as u32,
        attributes: vec![
            VertexAttribute {
                semantic: VertexSemantic::Position,
                format: VertexFormat::F32x3,
                offset: 0,
            },
            VertexAttribute {
                semantic: VertexSemantic::Color0,
                format: VertexFormat::U8x4Unorm,
                offset: 12,
            },
        ],
    }
}

fn default_color(idx: usize) -> u32 {
    const COLORS: [u32; 6] = [
        0xffff_4040,
        0xff40_ff40,
        0xff40_40ff,
        0xffff_ff40,
        0xffff_40ff,
        0xff40_ffff,
    ];
    COLORS[idx % COLORS.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_defaults_are_stable() {
        let app = NeoApp::new();
        assert_eq!(app.config().window.width, 960);
        assert_eq!(app.config().window.height, 540);
        assert_eq!(app.presenter_kind(), Presenter::D3d12Interop);
        assert!(app.config().hot_reload);
        assert_eq!(app.config().max_inflight, 8);
        assert_eq!(app.config().present_ring, 8);
        assert_eq!(app.config().render_policy, RenderPolicy::Auto);
    }

    #[test]
    fn invalid_sizes_and_fps_are_rejected() {
        let err = NeoApp::new().window("Neo", 0, 720).validate().unwrap_err();
        assert!(err.to_string().contains("width and height"));
        let err = NeoApp::new().target_fps(f32::NAN).validate().unwrap_err();
        assert!(err.to_string().contains("FPS caps"));
        let err = NeoApp::new().present_ring(0).validate().unwrap_err();
        assert!(err.to_string().contains("present_ring"));
    }

    #[test]
    fn presenter_fallback_policy_maps_to_live_window_args() {
        let args = NeoApp::new()
            .presenter(Presenter::D3d12Upload)
            .interop_fallback(InteropFallback::Fail)
            .render_policy(RenderPolicy::ForceRender)
            .kernel("image", "examples/live-window/live.neo")
            .live_window_args();
        assert!(args.windows(2).any(|pair| pair == ["--presenter", "d3d12"]));
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--interop-fallback", "fail"])
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--render-policy", "force-render"])
        );
    }

    #[test]
    fn mesh_builder_quad_matches_manual_runtime_layout() {
        let desc = MeshBuilder::quad().colored().desc();
        assert_eq!(desc.vertex_count, 4);
        assert_eq!(desc.index_count, 6);
        assert_eq!(desc.index_format, IndexFormat::U16);
        assert_eq!(desc.vertex_layout, friendly_vertex_layout());
    }

    #[test]
    fn fluent_builder_selects_mesh_demo_mode() {
        let args = NeoApp::new()
            .kernel("image", "examples/live-window/live.neo")
            .kernel("raster", "examples/mesh-buffer/raster.neo")
            .mesh("quad", MeshBuilder::quad().colored())
            .target_fps(240.0)
            .hot_reload(false)
            .live_window_args();
        assert!(args.windows(2).any(|pair| pair == ["--mode", "mesh-demo"]));
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--kernel-target-fps", "240"])
        );
        assert!(args.iter().any(|arg| arg == "--no-hot-reload"));
    }

    #[test]
    fn fluent_builder_selects_image_throughput_for_quad_stress() {
        let args = NeoApp::new()
            .kernel("image", "examples/stress-quads/million_quads.neo")
            .target_fps(240.0)
            .hot_reload(false)
            .live_window_args();
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--mode", "kernel-throughput"])
        );
        assert!(args.windows(2).any(|pair| pair == ["--title", "Neo"]));
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--present-target-fps", "240"])
        );
    }

    #[test]
    fn escape_hatch_into_parts_exposes_configuration() {
        let app = NeoApp::new()
            .kernel("image", "examples/live-window/live.neo")
            .mesh("quad", MeshBuilder::quad());
        let parts = app.into_parts();
        assert_eq!(parts.config.presenter, Presenter::D3d12Interop);
        assert!(parts.kernels.contains_key("image"));
        assert!(parts.meshes.contains_key("quad"));
    }

    #[test]
    fn context_escape_hatch_skips_without_cuda() {
        let mut app = NeoApp::new();
        match app.context() {
            Ok(ctx) => ctx.synchronize().unwrap(),
            Err(err) => eprintln!("skipping context escape hatch test without CUDA: {err}"),
        }
    }

    #[test]
    fn build_meshes_uses_runtime_meshbuffer_when_cuda_exists() {
        let mut app = NeoApp::new().mesh("quad", MeshBuilder::quad());
        match app.build_meshes() {
            Ok(meshes) => {
                let mesh = meshes.get("quad").unwrap();
                assert_eq!(mesh.desc().vertex_count, 4);
            }
            Err(err) => eprintln!("skipping runtime mesh build test without CUDA: {err}"),
        }
    }
}
