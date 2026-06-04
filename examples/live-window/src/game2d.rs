use std::{collections::BTreeMap, path::PathBuf};

use super::*;

const WHITE_TEXTURE_NAME: &str = "__neo_white";
const PACKED_SPRITE_BYTES: usize = 36;
const DEFAULT_GAME_FPS: f32 = 60.0;
const AUTO_GPU_AREA_NUMERATOR: u64 = 1;
const AUTO_GPU_AREA_DENOMINATOR: u64 = 4;
const AUTO_GPU_SPRITE_THRESHOLD: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameRenderer {
    Auto,
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DirtyRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl DirtyRect {
    fn full(width: u32, height: u32) -> Self {
        Self {
            x: 0,
            y: 0,
            width,
            height,
        }
    }

    fn from_sprite(sprite: &Sprite, width: u32, height: u32) -> Option<Self> {
        if sprite.rect.width <= 0.0 || sprite.rect.height <= 0.0 {
            return None;
        }
        let min_x = sprite.rect.x.floor().max(0.0) as i32;
        let min_y = sprite.rect.y.floor().max(0.0) as i32;
        let max_x = (sprite.rect.x + sprite.rect.width).ceil().min(width as f32) as i32;
        let max_y = (sprite.rect.y + sprite.rect.height)
            .ceil()
            .min(height as f32) as i32;
        if min_x >= max_x || min_y >= max_y {
            return None;
        }
        Some(Self {
            x: min_x as u32,
            y: min_y as u32,
            width: (max_x - min_x) as u32,
            height: (max_y - min_y) as u32,
        })
    }

    fn end_x(self) -> u32 {
        self.x + self.width
    }

    fn end_y(self) -> u32 {
        self.y + self.height
    }

    fn area(self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    fn is_full(self, width: u32, height: u32) -> bool {
        self.x == 0 && self.y == 0 && self.width == width && self.height == height
    }

    fn union(self, other: Self) -> Self {
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        let end_x = self.end_x().max(other.end_x());
        let end_y = self.end_y().max(other.end_y());
        Self {
            x,
            y,
            width: end_x - x,
            height: end_y - y,
        }
    }
}

impl Rect {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub const BLACK: Self = Self::rgba(0, 0, 0, 255);
    pub const WHITE: Self = Self::rgba(255, 255, 255, 255);

    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self::rgba(r, g, b, 255)
    }

    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    fn bgra(self) -> [u8; 4] {
        [self.b, self.g, self.r, self.a]
    }

    fn bgra_u32(self) -> u32 {
        u32::from(self.b)
            | (u32::from(self.g) << 8)
            | (u32::from(self.r) << 16)
            | (u32::from(self.a) << 24)
    }

    fn rgba_u32(self) -> u32 {
        u32::from(self.r)
            | (u32::from(self.g) << 8)
            | (u32::from(self.b) << 16)
            | (u32::from(self.a) << 24)
    }
}

#[derive(Debug, Clone, PartialEq)]
enum SpriteTexture {
    Image(String),
    Solid,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sprite {
    texture: SpriteTexture,
    rect: Rect,
    tint: Color,
}

impl Sprite {
    pub fn image(name: impl Into<String>) -> Self {
        Self {
            texture: SpriteTexture::Image(name.into()),
            rect: Rect::new(0.0, 0.0, 32.0, 32.0),
            tint: Color::WHITE,
        }
    }

    pub fn solid(color: Color) -> Self {
        Self {
            texture: SpriteTexture::Solid,
            rect: Rect::new(0.0, 0.0, 32.0, 32.0),
            tint: color,
        }
    }

    pub fn at(mut self, x: f32, y: f32) -> Self {
        self.rect.x = x;
        self.rect.y = y;
        self
    }

    pub fn size(mut self, width: f32, height: f32) -> Self {
        self.rect.width = width;
        self.rect.height = height;
        self
    }

    pub fn rect(mut self, x: f32, y: f32, width: f32, height: f32) -> Self {
        self.rect = Rect::new(x, y, width, height);
        self
    }

    pub fn tint(mut self, color: Color) -> Self {
        self.tint = color;
        self
    }

    pub fn bounds(&self) -> Rect {
        self.rect
    }

    pub fn color(&self) -> Color {
        self.tint
    }

    fn texture_name(&self) -> &str {
        match &self.texture {
            SpriteTexture::Image(name) => name,
            SpriteTexture::Solid => WHITE_TEXTURE_NAME,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameTextureSpec {
    pub name: String,
    pub path: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GameWindowConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GameConfig {
    pub window: GameWindowConfig,
    pub renderer: GameRenderer,
    pub target_fps: Option<f32>,
    pub max_frames: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct NeoGame {
    config: GameConfig,
    textures: Vec<GameTextureSpec>,
}

impl Default for NeoGame {
    fn default() -> Self {
        Self::new()
    }
}

impl NeoGame {
    pub fn new() -> Self {
        Self {
            config: GameConfig {
                window: GameWindowConfig {
                    title: "Neo Game".to_string(),
                    width: DEFAULT_WIDTH,
                    height: DEFAULT_HEIGHT,
                },
                renderer: GameRenderer::Auto,
                target_fps: Some(DEFAULT_GAME_FPS),
                max_frames: None,
            },
            textures: Vec::new(),
        }
    }

    pub fn window(mut self, title: impl Into<String>, width: u32, height: u32) -> Self {
        self.config.window = GameWindowConfig {
            title: title.into(),
            width,
            height,
        };
        self
    }

    pub fn texture(mut self, name: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        let name = name.into();
        self.textures.push(GameTextureSpec {
            name,
            path: path.into(),
        });
        self
    }

    pub fn renderer(mut self, renderer: GameRenderer) -> Self {
        self.config.renderer = renderer;
        self
    }

    pub fn target_fps(mut self, fps: f32) -> Self {
        self.config.target_fps = Some(fps);
        self
    }

    pub fn uncapped(mut self) -> Self {
        self.config.target_fps = None;
        self
    }

    pub fn max_frames(mut self, frames: u32) -> Self {
        self.config.max_frames = Some(frames);
        self
    }

    pub fn config(&self) -> &GameConfig {
        &self.config
    }

    pub fn texture_specs(&self) -> &[GameTextureSpec] {
        &self.textures
    }

    pub fn validate(&self) -> Result<()> {
        if self.config.window.width == 0 || self.config.window.height == 0 {
            bail!("game window width and height must be greater than zero");
        }
        if let Some(fps) = self.config.target_fps
            && (!fps.is_finite() || fps <= 0.0)
        {
            bail!("game target_fps must be finite and greater than zero");
        }
        if self
            .textures
            .iter()
            .any(|texture| texture.name == WHITE_TEXTURE_NAME)
        {
            bail!("texture name `{WHITE_TEXTURE_NAME}` is reserved");
        }
        Ok(())
    }

    pub fn run<F>(self, callback: F) -> Result<()>
    where
        F: 'static + FnMut(&mut GameFrame) -> Result<()>,
    {
        run_game(self, callback)
    }

    pub fn run_for_frames<F>(&self, frames: u32, mut callback: F) -> Result<Vec<u8>>
    where
        F: FnMut(&mut GameFrame) -> Result<()>,
    {
        self.validate()?;
        let atlas = TextureAtlas::from_specs(&self.textures)?;
        let mut renderer = CpuGameRenderer::default();
        let mut frame = GameFrame::new(self.config.window.width, self.config.window.height);
        let fixed_delta = self
            .config
            .target_fps
            .map(|fps| 1.0 / fps)
            .unwrap_or(1.0 / 60.0);

        for frame_index in 0..frames {
            frame.begin(
                self.config.window.width,
                self.config.window.height,
                frame_index as f32 * fixed_delta,
                fixed_delta,
                frame_index,
            );
            callback(&mut frame)?;
            renderer.render(&atlas, &frame)?;
        }
        Ok(renderer.pixels)
    }
}

pub struct GameFrame {
    width: u32,
    height: u32,
    time_seconds: f32,
    delta_seconds: f32,
    frame_index: u32,
    clear_color: Color,
    sprites: Vec<Sprite>,
}

impl GameFrame {
    fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            time_seconds: 0.0,
            delta_seconds: 0.0,
            frame_index: 0,
            clear_color: Color::BLACK,
            sprites: Vec::new(),
        }
    }

    fn begin(
        &mut self,
        width: u32,
        height: u32,
        time_seconds: f32,
        delta_seconds: f32,
        frame_index: u32,
    ) {
        self.width = width;
        self.height = height;
        self.time_seconds = time_seconds;
        self.delta_seconds = delta_seconds;
        self.frame_index = frame_index;
        self.clear_color = Color::BLACK;
        self.sprites.clear();
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn time_seconds(&self) -> f32 {
        self.time_seconds
    }

    pub fn delta_seconds(&self) -> f32 {
        self.delta_seconds
    }

    pub fn frame_index(&self) -> u32 {
        self.frame_index
    }

    pub fn clear(&mut self, color: Color) {
        self.clear_color = color;
    }

    pub fn draw(&mut self, sprite: Sprite) {
        self.sprites.push(sprite);
    }

    pub fn sprites(&self) -> &[Sprite] {
        &self.sprites
    }
}

#[derive(Debug, Clone, Copy)]
struct AtlasRegion {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

struct LoadedTexture {
    name: String,
    width: u32,
    height: u32,
    bgra: Vec<u8>,
}

struct TextureAtlas {
    width: u32,
    height: u32,
    pixels_bgra: Vec<u8>,
    regions: BTreeMap<String, AtlasRegion>,
}

impl TextureAtlas {
    fn from_specs(specs: &[GameTextureSpec]) -> Result<Self> {
        let mut textures = vec![LoadedTexture {
            name: WHITE_TEXTURE_NAME.to_string(),
            width: 1,
            height: 1,
            bgra: vec![255, 255, 255, 255],
        }];
        for spec in specs {
            textures.push(load_texture(spec)?);
        }
        Self::pack(textures)
    }

    fn pack(textures: Vec<LoadedTexture>) -> Result<Self> {
        let total_area: u64 = textures
            .iter()
            .map(|texture| u64::from(texture.width) * u64::from(texture.height))
            .sum();
        let max_width = textures
            .iter()
            .map(|texture| texture.width)
            .max()
            .unwrap_or(1);
        let sqrt_area = (total_area as f64).sqrt().ceil() as u32;
        let atlas_width = next_power_of_two_u32(max_width.max(sqrt_area).max(16))?;
        let mut x = 0u32;
        let mut y = 0u32;
        let mut shelf_height = 0u32;
        let mut placements = Vec::with_capacity(textures.len());

        for texture in &textures {
            if texture.width == 0 || texture.height == 0 {
                bail!("texture `{}` has zero width or height", texture.name);
            }
            if x != 0 && x.saturating_add(texture.width) > atlas_width {
                y = y
                    .checked_add(shelf_height)
                    .context("texture atlas height overflow")?;
                x = 0;
                shelf_height = 0;
            }
            placements.push(AtlasRegion {
                x,
                y,
                width: texture.width,
                height: texture.height,
            });
            x = x
                .checked_add(texture.width)
                .context("texture atlas width overflow")?;
            shelf_height = shelf_height.max(texture.height);
        }
        let atlas_height = next_power_of_two_u32(
            y.checked_add(shelf_height)
                .context("texture atlas height overflow")?
                .max(1),
        )?;
        let byte_len = frame_byte_len(atlas_width, atlas_height)?;
        let mut pixels_bgra = vec![0; byte_len];
        let mut regions = BTreeMap::new();

        for (texture, region) in textures.into_iter().zip(placements) {
            copy_texture_into_atlas(&mut pixels_bgra, atlas_width, &texture, region);
            regions.insert(texture.name, region);
        }
        Ok(Self {
            width: atlas_width,
            height: atlas_height,
            pixels_bgra,
            regions,
        })
    }

    fn region(&self, name: &str) -> Result<AtlasRegion> {
        self.regions
            .get(name)
            .copied()
            .ok_or_else(|| anyhow!("sprite texture `{name}` was not registered"))
    }
}

fn next_power_of_two_u32(value: u32) -> Result<u32> {
    value
        .checked_next_power_of_two()
        .ok_or_else(|| anyhow!("texture atlas dimensions overflowed u32"))
}

fn load_texture(spec: &GameTextureSpec) -> Result<LoadedTexture> {
    let image = image::ImageReader::open(&spec.path)
        .with_context(|| format!("failed to open texture `{}`", spec.path.display()))?
        .decode()
        .with_context(|| format!("failed to decode texture `{}`", spec.path.display()))?
        .to_rgba8();
    let (width, height) = image.dimensions();
    let mut bgra = Vec::with_capacity(image.as_raw().len());
    for rgba in image.as_raw().chunks_exact(4) {
        bgra.extend_from_slice(&[rgba[2], rgba[1], rgba[0], rgba[3]]);
    }
    Ok(LoadedTexture {
        name: spec.name.clone(),
        width,
        height,
        bgra,
    })
}

fn copy_texture_into_atlas(
    atlas: &mut [u8],
    atlas_width: u32,
    texture: &LoadedTexture,
    region: AtlasRegion,
) {
    let atlas_pitch = atlas_width as usize * 4;
    let texture_pitch = texture.width as usize * 4;
    for row in 0..texture.height as usize {
        let src_start = row * texture_pitch;
        let dst_start = (region.y as usize + row) * atlas_pitch + region.x as usize * 4;
        atlas[dst_start..dst_start + texture_pitch]
            .copy_from_slice(&texture.bgra[src_start..src_start + texture_pitch]);
    }
}

#[derive(Default)]
struct CpuGameRenderer {
    pixels: Vec<u8>,
}

impl CpuGameRenderer {
    fn render<'a>(&'a mut self, atlas: &TextureAtlas, frame: &GameFrame) -> Result<&'a [u8]> {
        let dirty = DirtyRect::full(frame.width, frame.height);
        self.render_dirty(atlas, frame, Some(dirty))
    }

    fn render_dirty<'a>(
        &'a mut self,
        atlas: &TextureAtlas,
        frame: &GameFrame,
        mut dirty: Option<DirtyRect>,
    ) -> Result<&'a [u8]> {
        let byte_len = frame_byte_len(frame.width, frame.height)?;
        if self.pixels.len() != byte_len {
            self.pixels.resize(byte_len, 0);
            dirty = Some(DirtyRect::full(frame.width, frame.height));
        }
        validate_sprite_textures(atlas, &frame.sprites)?;
        let Some(dirty) = dirty else {
            return Ok(&self.pixels);
        };
        if dirty.width == 0 || dirty.height == 0 {
            return Ok(&self.pixels);
        }
        fill_bgra_rect(&mut self.pixels, frame.width, dirty, frame.clear_color);
        render_cpu_sprites(
            &mut self.pixels,
            frame.width,
            frame.height,
            atlas,
            &frame.sprites,
            dirty,
        )?;
        Ok(&self.pixels)
    }
}

fn fill_bgra_rect(pixels: &mut [u8], width: u32, dirty: DirtyRect, color: Color) {
    let bgra = color.bgra();
    let pitch = width as usize * 4;
    let row_start = dirty.x as usize * 4;
    let row_end = dirty.end_x() as usize * 4;
    for y in dirty.y as usize..dirty.end_y() as usize {
        let row = &mut pixels[y * pitch + row_start..y * pitch + row_end];
        for pixel in row.chunks_exact_mut(4) {
            pixel.copy_from_slice(&bgra);
        }
    }
}

fn render_cpu_sprites(
    dst: &mut [u8],
    width: u32,
    height: u32,
    atlas: &TextureAtlas,
    sprites: &[Sprite],
    dirty: DirtyRect,
) -> Result<()> {
    let dst_pitch = width as usize * 4;
    let atlas_pitch = atlas.width as usize * 4;
    for sprite in sprites {
        let region = atlas.region(sprite.texture_name())?;
        if sprite.rect.width <= 0.0 || sprite.rect.height <= 0.0 {
            continue;
        }
        let min_x = sprite.rect.x.floor().max(0.0) as i32;
        let min_y = sprite.rect.y.floor().max(0.0) as i32;
        let max_x = (sprite.rect.x + sprite.rect.width).ceil().min(width as f32) as i32;
        let max_y = (sprite.rect.y + sprite.rect.height)
            .ceil()
            .min(height as f32) as i32;
        let min_x = min_x.max(dirty.x as i32);
        let min_y = min_y.max(dirty.y as i32);
        let max_x = max_x.min(dirty.end_x() as i32);
        let max_y = max_y.min(dirty.end_y() as i32);
        if min_x >= max_x || min_y >= max_y {
            continue;
        }

        for y in min_y..max_y {
            let v = ((y as f32 + 0.5 - sprite.rect.y) / sprite.rect.height).clamp(0.0, 0.999_999);
            let src_y = region.y + ((v * region.height as f32) as u32).min(region.height - 1);
            for x in min_x..max_x {
                let u =
                    ((x as f32 + 0.5 - sprite.rect.x) / sprite.rect.width).clamp(0.0, 0.999_999);
                let src_x = region.x + ((u * region.width as f32) as u32).min(region.width - 1);
                let src_index = src_y as usize * atlas_pitch + src_x as usize * 4;
                let dst_index = y as usize * dst_pitch + x as usize * 4;
                blend_tinted_bgra(
                    &atlas.pixels_bgra[src_index..src_index + 4],
                    &mut dst[dst_index..dst_index + 4],
                    sprite.tint,
                );
            }
        }
    }
    Ok(())
}

fn blend_tinted_bgra(src: &[u8], dst: &mut [u8], tint: Color) {
    let src_b = multiply_u8(src[0], tint.b);
    let src_g = multiply_u8(src[1], tint.g);
    let src_r = multiply_u8(src[2], tint.r);
    let src_a = multiply_u8(src[3], tint.a);
    blend_channel(src_b, src_a, &mut dst[0]);
    blend_channel(src_g, src_a, &mut dst[1]);
    blend_channel(src_r, src_a, &mut dst[2]);
    blend_alpha(src_a, &mut dst[3]);
}

fn multiply_u8(a: u8, b: u8) -> u8 {
    ((u32::from(a) * u32::from(b) + 127) / 255) as u8
}

fn blend_channel(src: u8, src_a: u8, dst: &mut u8) {
    let inv_a = 255 - u32::from(src_a);
    *dst = ((u32::from(src) * u32::from(src_a) + u32::from(*dst) * inv_a + 127) / 255) as u8;
}

fn blend_alpha(src_a: u8, dst: &mut u8) {
    let inv_a = 255 - u32::from(src_a);
    *dst = (u32::from(src_a) + (u32::from(*dst) * inv_a + 127) / 255).min(255) as u8;
}

struct RenderedGameFrame<'a> {
    bgra: &'a [u8],
    dirty: Option<DirtyRect>,
}

struct DirtyTracker {
    initialized: bool,
    width: u32,
    height: u32,
    clear_color: Color,
    previous_bounds: Option<DirtyRect>,
}

impl Default for DirtyTracker {
    fn default() -> Self {
        Self {
            initialized: false,
            width: 0,
            height: 0,
            clear_color: Color::BLACK,
            previous_bounds: None,
        }
    }
}

impl DirtyTracker {
    fn next_dirty(&mut self, frame: &GameFrame) -> Option<DirtyRect> {
        let current_bounds = clipped_sprite_bounds(frame);
        let full = !self.initialized
            || self.width != frame.width
            || self.height != frame.height
            || self.clear_color != frame.clear_color;
        let dirty = if full {
            Some(DirtyRect::full(frame.width, frame.height))
        } else {
            match (self.previous_bounds, current_bounds) {
                (Some(previous), Some(current)) => Some(previous.union(current)),
                (Some(previous), None) => Some(previous),
                (None, Some(current)) => Some(current),
                (None, None) => None,
            }
        };

        self.initialized = true;
        self.width = frame.width;
        self.height = frame.height;
        self.clear_color = frame.clear_color;
        self.previous_bounds = current_bounds;
        dirty
    }
}

enum GameRenderMode {
    Cpu(CpuGameRenderer),
    Gpu(GpuGameRenderer),
    Auto {
        cpu: CpuGameRenderer,
        gpu: Option<GpuGameRenderer>,
        gpu_failed: bool,
        use_gpu: bool,
    },
}

struct GameRenderBackend {
    mode: GameRenderMode,
    dirty: DirtyTracker,
}

impl GameRenderBackend {
    fn new(renderer: GameRenderer, atlas: &TextureAtlas) -> Result<Self> {
        let mode = match renderer {
            GameRenderer::Cpu => GameRenderMode::Cpu(CpuGameRenderer::default()),
            GameRenderer::Gpu => GameRenderMode::Gpu(GpuGameRenderer::new(atlas)?),
            GameRenderer::Auto => GameRenderMode::Auto {
                cpu: CpuGameRenderer::default(),
                gpu: None,
                gpu_failed: false,
                use_gpu: false,
            },
        };
        Ok(Self {
            mode,
            dirty: DirtyTracker::default(),
        })
    }

    fn render<'a>(
        &'a mut self,
        atlas: &TextureAtlas,
        frame: &GameFrame,
    ) -> Result<RenderedGameFrame<'a>> {
        let mut dirty = self.dirty.next_dirty(frame);
        let bgra = match &mut self.mode {
            GameRenderMode::Cpu(renderer) => renderer.render_dirty(atlas, frame, dirty)?,
            GameRenderMode::Gpu(renderer) => renderer.render_dirty(atlas, frame, dirty)?,
            GameRenderMode::Auto {
                cpu,
                gpu,
                gpu_failed,
                use_gpu,
            } => {
                if !*use_gpu && !*gpu_failed && should_auto_use_gpu(frame) {
                    match GpuGameRenderer::new(atlas) {
                        Ok(renderer) => {
                            *gpu = Some(renderer);
                            *use_gpu = true;
                            dirty = Some(DirtyRect::full(frame.width, frame.height));
                        }
                        Err(err) => {
                            *gpu_failed = true;
                            eprintln!("NeoGame GPU renderer unavailable; staying on CPU: {err:#}");
                        }
                    }
                }

                if *use_gpu {
                    gpu.as_mut()
                        .expect("Auto game GPU renderer is initialized when use_gpu is true")
                        .render_dirty(atlas, frame, dirty)?
                } else {
                    cpu.render_dirty(atlas, frame, dirty)?
                }
            }
        };
        Ok(RenderedGameFrame { bgra, dirty })
    }
}

fn clipped_sprite_bounds(frame: &GameFrame) -> Option<DirtyRect> {
    frame
        .sprites
        .iter()
        .filter_map(|sprite| DirtyRect::from_sprite(sprite, frame.width, frame.height))
        .reduce(DirtyRect::union)
}

fn clipped_sprite_area(frame: &GameFrame) -> u64 {
    frame
        .sprites
        .iter()
        .filter_map(|sprite| DirtyRect::from_sprite(sprite, frame.width, frame.height))
        .map(DirtyRect::area)
        .sum()
}

fn should_auto_use_gpu(frame: &GameFrame) -> bool {
    if frame.sprites.len() >= AUTO_GPU_SPRITE_THRESHOLD {
        return true;
    }
    let frame_area = u64::from(frame.width) * u64::from(frame.height);
    frame_area > 0
        && clipped_sprite_area(frame) * AUTO_GPU_AREA_DENOMINATOR
            >= frame_area * AUTO_GPU_AREA_NUMERATOR
}

#[cfg(test)]
impl GameRenderBackend {
    fn is_using_gpu(&self) -> bool {
        match &self.mode {
            GameRenderMode::Gpu(_) => true,
            GameRenderMode::Auto { use_gpu, .. } => *use_gpu,
            GameRenderMode::Cpu(_) => false,
        }
    }
}

struct GpuGameRenderer {
    context: NeoContext,
    kernel: Kernel,
    atlas: DeviceBuffer<u8>,
    sprites: DeviceBuffer<u8>,
    sprite_capacity: usize,
    pixels: Option<DeviceBuffer<u8>>,
    host_bgra: Option<ReadablePinnedHostBuffer<u8>>,
    width: u32,
    height: u32,
}

impl GpuGameRenderer {
    fn new(atlas: &TextureAtlas) -> Result<Self> {
        let context = NeoContext::new_default_device()?;
        let module = neo_runtime::Module::from_cuda_source(&context, sprite_cuda_source())?;
        let kernel = module.kernel("neo_game2d_render")?;
        let atlas = DeviceBuffer::upload(&context, &atlas.pixels_bgra)?;
        let sprites = DeviceBuffer::new(&context, PACKED_SPRITE_BYTES)?;
        Ok(Self {
            context,
            kernel,
            atlas,
            sprites,
            sprite_capacity: 1,
            pixels: None,
            host_bgra: None,
            width: 0,
            height: 0,
        })
    }

    #[cfg(test)]
    fn render<'a>(&'a mut self, atlas: &TextureAtlas, frame: &GameFrame) -> Result<&'a [u8]> {
        let dirty = DirtyRect::full(frame.width, frame.height);
        self.render_dirty(atlas, frame, Some(dirty))
    }

    fn render_dirty<'a>(
        &'a mut self,
        atlas: &TextureAtlas,
        frame: &GameFrame,
        mut dirty: Option<DirtyRect>,
    ) -> Result<&'a [u8]> {
        if self.ensure_size(frame.width, frame.height)? {
            dirty = Some(DirtyRect::full(frame.width, frame.height));
        }
        validate_sprite_textures(atlas, &frame.sprites)?;
        let Some(dirty) = dirty else {
            return Ok(self
                .host_bgra
                .as_ref()
                .expect("GPU game host buffer was allocated by ensure_size")
                .as_slice());
        };
        if dirty.width == 0 || dirty.height == 0 {
            return Ok(self
                .host_bgra
                .as_ref()
                .expect("GPU game host buffer was allocated by ensure_size")
                .as_slice());
        }
        self.ensure_sprite_capacity(frame.sprites.len())?;
        let packed_sprites = pack_sprites(atlas, &frame.sprites)?;
        if !packed_sprites.is_empty() {
            self.sprites.upload_range(0, &packed_sprites)?;
        }
        let sprite_count = frame.sprites.len() as u32;
        let clear = frame.clear_color.bgra_u32();
        let dims = LaunchDims::for_2d(dirty.width, dirty.height, BLOCK);
        {
            let pixels = self
                .pixels
                .as_mut()
                .expect("GPU game pixels were allocated by ensure_size");
            let mut launch = self.kernel.launcher();
            launch
                .arg_buffer_mut(pixels)
                .arg(&frame.width)
                .arg(&frame.height)
                .arg_buffer(&self.atlas)
                .arg(&atlas.width)
                .arg(&atlas.height)
                .arg_buffer(&self.sprites)
                .arg(&sprite_count)
                .arg(&clear)
                .arg(&dirty.x)
                .arg(&dirty.y)
                .arg(&dirty.width)
                .arg(&dirty.height);
            unsafe {
                launch.launch(dims)?;
            }
        }
        let row_offset = dirty.y as usize * frame.width as usize * 4;
        let row_len = dirty.height as usize * frame.width as usize * 4;
        if dirty.is_full(frame.width, frame.height) {
            self.pixels
                .as_ref()
                .expect("GPU game pixels were allocated by ensure_size")
                .download_into_readable_pinned(
                    self.host_bgra
                        .as_mut()
                        .expect("GPU game host buffer was allocated by ensure_size"),
                )?;
            self.context.synchronize()?;
        } else {
            let host = self
                .host_bgra
                .as_mut()
                .expect("GPU game host buffer was allocated by ensure_size");
            self.pixels
                .as_ref()
                .expect("GPU game pixels were allocated by ensure_size")
                .download_range(
                    row_offset,
                    &mut host.as_mut_slice()[row_offset..row_offset + row_len],
                )?;
        }
        Ok(self
            .host_bgra
            .as_ref()
            .expect("GPU game host buffer was allocated by ensure_size")
            .as_slice())
    }

    fn ensure_size(&mut self, width: u32, height: u32) -> Result<bool> {
        if self.pixels.is_some() && self.width == width && self.height == height {
            return Ok(false);
        }
        let byte_len = frame_byte_len(width, height)?;
        self.pixels = Some(DeviceBuffer::new(&self.context, byte_len)?);
        self.host_bgra = Some(self.context.alloc_readable_pinned(byte_len)?);
        self.width = width;
        self.height = height;
        Ok(true)
    }

    fn ensure_sprite_capacity(&mut self, sprites: usize) -> Result<()> {
        let needed = sprites.max(1);
        if self.sprite_capacity >= needed {
            return Ok(());
        }
        self.sprite_capacity = needed.next_power_of_two();
        self.sprites =
            DeviceBuffer::new(&self.context, self.sprite_capacity * PACKED_SPRITE_BYTES)?;
        Ok(())
    }
}

fn pack_sprites(atlas: &TextureAtlas, sprites: &[Sprite]) -> Result<Vec<u8>> {
    let mut packed = Vec::with_capacity(sprites.len() * PACKED_SPRITE_BYTES);
    for sprite in sprites {
        let region = atlas.region(sprite.texture_name())?;
        push_f32(&mut packed, sprite.rect.x);
        push_f32(&mut packed, sprite.rect.y);
        push_f32(&mut packed, sprite.rect.width);
        push_f32(&mut packed, sprite.rect.height);
        push_u32(&mut packed, region.x);
        push_u32(&mut packed, region.y);
        push_u32(&mut packed, region.width);
        push_u32(&mut packed, region.height);
        push_u32(&mut packed, sprite.tint.rgba_u32());
    }
    Ok(packed)
}

fn validate_sprite_textures(atlas: &TextureAtlas, sprites: &[Sprite]) -> Result<()> {
    for sprite in sprites {
        atlas.region(sprite.texture_name())?;
    }
    Ok(())
}

fn push_f32(dst: &mut Vec<u8>, value: f32) {
    dst.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(dst: &mut Vec<u8>, value: u32) {
    dst.extend_from_slice(&value.to_le_bytes());
}

fn sprite_cuda_source() -> String {
    r#"
struct NeoGameSprite {
    float x;
    float y;
    float w;
    float h;
    unsigned int sx;
    unsigned int sy;
    unsigned int sw;
    unsigned int sh;
    unsigned int tint_rgba;
};

extern "C" __global__ void neo_game2d_render(
    unsigned char* pixels,
    unsigned int width,
    unsigned int height,
    const unsigned char* atlas,
    unsigned int atlas_width,
    unsigned int atlas_height,
    const unsigned char* sprite_bytes,
    unsigned int sprite_count,
    unsigned int clear_bgra,
    unsigned int dirty_x,
    unsigned int dirty_y,
    unsigned int dirty_width,
    unsigned int dirty_height)
{
    unsigned int local_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int local_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (local_x >= dirty_width || local_y >= dirty_height) {
        return;
    }
    unsigned int x = dirty_x + local_x;
    unsigned int y = dirty_y + local_y;
    if (x >= width || y >= height) {
        return;
    }

    unsigned int out_b = clear_bgra & 255u;
    unsigned int out_g = (clear_bgra >> 8u) & 255u;
    unsigned int out_r = (clear_bgra >> 16u) & 255u;
    unsigned int out_a = (clear_bgra >> 24u) & 255u;
    const NeoGameSprite* sprites = reinterpret_cast<const NeoGameSprite*>(sprite_bytes);

    for (unsigned int i = 0; i < sprite_count; ++i) {
        NeoGameSprite sprite = sprites[i];
        if (sprite.w <= 0.0f || sprite.h <= 0.0f) {
            continue;
        }
        float fx = (float)x + 0.5f;
        float fy = (float)y + 0.5f;
        if (fx < sprite.x || fy < sprite.y || fx >= sprite.x + sprite.w || fy >= sprite.y + sprite.h) {
            continue;
        }
        float u = (fx - sprite.x) / sprite.w;
        float v = (fy - sprite.y) / sprite.h;
        unsigned int src_x = sprite.sx + min(sprite.sw - 1u, (unsigned int)(u * (float)sprite.sw));
        unsigned int src_y = sprite.sy + min(sprite.sh - 1u, (unsigned int)(v * (float)sprite.sh));
        if (src_x >= atlas_width || src_y >= atlas_height) {
            continue;
        }
        unsigned int src_index = (src_y * atlas_width + src_x) * 4u;
        unsigned int src_b = atlas[src_index + 0u];
        unsigned int src_g = atlas[src_index + 1u];
        unsigned int src_r = atlas[src_index + 2u];
        unsigned int src_a = atlas[src_index + 3u];
        unsigned int tint_r = sprite.tint_rgba & 255u;
        unsigned int tint_g = (sprite.tint_rgba >> 8u) & 255u;
        unsigned int tint_b = (sprite.tint_rgba >> 16u) & 255u;
        unsigned int tint_a = (sprite.tint_rgba >> 24u) & 255u;
        src_r = (src_r * tint_r + 127u) / 255u;
        src_g = (src_g * tint_g + 127u) / 255u;
        src_b = (src_b * tint_b + 127u) / 255u;
        src_a = (src_a * tint_a + 127u) / 255u;
        unsigned int inv_a = 255u - src_a;
        out_b = (src_b * src_a + out_b * inv_a + 127u) / 255u;
        out_g = (src_g * src_a + out_g * inv_a + 127u) / 255u;
        out_r = (src_r * src_a + out_r * inv_a + 127u) / 255u;
        out_a = src_a + (out_a * inv_a + 127u) / 255u;
        out_a = out_a > 255u ? 255u : out_a;
    }

    unsigned int dst_index = (y * width + x) * 4u;
    pixels[dst_index + 0u] = (unsigned char)out_b;
    pixels[dst_index + 1u] = (unsigned char)out_g;
    pixels[dst_index + 2u] = (unsigned char)out_r;
    pixels[dst_index + 3u] = (unsigned char)out_a;
}
"#
    .to_string()
}

#[allow(deprecated)]
fn run_game<F>(game: NeoGame, mut callback: F) -> Result<()>
where
    F: 'static + FnMut(&mut GameFrame) -> Result<()>,
{
    game.validate()?;
    let atlas = TextureAtlas::from_specs(&game.textures)?;
    let event_loop = EventLoop::new()?;
    let window = create_window(
        &event_loop,
        &game.config.window.title,
        game.config.window.width,
        game.config.window.height,
    )?;
    let mut presenter = create_game_presenter(&window)?;
    let mut renderer = GameRenderBackend::new(game.config.renderer, &atlas)?;
    let mut frame = GameFrame::new(game.config.window.width, game.config.window.height);
    let start = Instant::now();
    let mut last_frame_at = start;
    let target_interval = game
        .config
        .target_fps
        .map(|fps| Duration::from_secs_f32(1.0 / fps));
    let mut next_frame_at = target_interval.map(|interval| start + interval);
    let mut frame_index = 0u32;

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);
        match event {
            Event::AboutToWait => {
                let now = Instant::now();
                if let Some(next) = next_frame_at
                    && now < next
                {
                    elwt.set_control_flow(ControlFlow::WaitUntil(next));
                    return;
                }
                if let Some(interval) = target_interval {
                    let next = next_frame_at.get_or_insert(now + interval);
                    while *next <= now {
                        *next += interval;
                    }
                }

                let size = window.inner_size();
                if size.width == 0 || size.height == 0 {
                    elwt.set_control_flow(ControlFlow::Wait);
                    return;
                }

                let delta = now.duration_since(last_frame_at).as_secs_f32();
                last_frame_at = now;
                frame.begin(
                    size.width,
                    size.height,
                    start.elapsed().as_secs_f32(),
                    delta,
                    frame_index,
                );

                let result = callback(&mut frame).and_then(|_| {
                    let rendered = renderer.render(&atlas, &frame)?;
                    if let Some(dirty) = rendered.dirty {
                        let rows = DirtyRowRange::new(dirty.y, dirty.end_y(), size.height)?;
                        presenter
                            .present_dirty_rows(size, rendered.bgra, rows)
                            .map(|_| ())
                    } else {
                        Ok(())
                    }
                });
                if let Err(err) = result {
                    eprintln!("NeoGame render error: {err:#}");
                    elwt.exit();
                    return;
                }
                frame_index = frame_index.wrapping_add(1);
                if game
                    .config
                    .max_frames
                    .is_some_and(|max_frames| frame_index >= max_frames)
                {
                    elwt.exit();
                }
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => elwt.exit(),
            _ => {}
        }
    })?;
    Ok(())
}

fn create_game_presenter(window: &Window) -> Result<WindowPresenter> {
    match WindowPresenter::new(
        window,
        PresenterKind::D3d11,
        3,
        D3dUploadMode::UpdateSubresource,
        None,
    ) {
        Ok(presenter) => Ok(presenter),
        Err(d3d11_err) => {
            eprintln!("D3D11 game presenter unavailable; falling back to GDI: {d3d11_err:#}");
            WindowPresenter::new(
                window,
                PresenterKind::Gdi,
                1,
                D3dUploadMode::MappedCopy,
                None,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_atlas() -> TextureAtlas {
        TextureAtlas::pack(vec![
            LoadedTexture {
                name: WHITE_TEXTURE_NAME.to_string(),
                width: 1,
                height: 1,
                bgra: vec![255, 255, 255, 255],
            },
            LoadedTexture {
                name: "two".to_string(),
                width: 2,
                height: 1,
                bgra: vec![0, 0, 255, 255, 0, 255, 0, 255],
            },
        ])
        .unwrap()
    }

    #[test]
    fn game_builder_stores_beginner_config() {
        assert_eq!(NeoGame::new().config().target_fps, Some(DEFAULT_GAME_FPS));
        assert_eq!(NeoGame::new().uncapped().config().target_fps, None);

        let game = NeoGame::new()
            .window("Sprites", 800, 600)
            .texture("player", "assets/player.png")
            .renderer(GameRenderer::Cpu)
            .target_fps(60.0)
            .max_frames(12);
        assert_eq!(game.config().window.title, "Sprites");
        assert_eq!(game.config().window.width, 800);
        assert_eq!(game.config().renderer, GameRenderer::Cpu);
        assert_eq!(game.config().target_fps, Some(60.0));
        assert_eq!(game.config().max_frames, Some(12));
        assert_eq!(game.texture_specs()[0].name, "player");
    }

    #[test]
    fn game_rejects_invalid_target_fps() {
        for fps in [0.0, f32::NAN, f32::INFINITY] {
            let err = NeoGame::new().target_fps(fps).validate().unwrap_err();
            assert!(
                err.to_string()
                    .contains("target_fps must be finite and greater than zero")
            );
        }
    }

    #[test]
    fn closure_state_can_animate_across_bounded_frames() {
        let mut x = 0.0;
        let pixels = NeoGame::new()
            .window("test", 4, 1)
            .renderer(GameRenderer::Cpu)
            .target_fps(10.0)
            .run_for_frames(3, |frame| {
                x += 1.0;
                frame.clear(Color::BLACK);
                frame.draw(Sprite::solid(Color::rgb(255, 0, 0)).rect(x, 0.0, 1.0, 1.0));
                Ok(())
            })
            .unwrap();
        assert_eq!(x, 3.0);
        assert_eq!(&pixels[3 * 4..3 * 4 + 4], &[0, 0, 255, 255]);
    }

    #[test]
    fn auto_renderer_stays_cpu_for_tiny_scene_and_marks_large_scene_for_gpu() {
        let atlas = test_atlas();
        let mut backend = GameRenderBackend::new(GameRenderer::Auto, &atlas).unwrap();
        let mut frame = GameFrame::new(800, 600);
        frame.begin(800, 600, 0.0, 1.0 / 60.0, 0);
        frame.clear(Color::BLACK);
        frame.draw(Sprite::solid(Color::WHITE).rect(120.0, 80.0, 64.0, 64.0));
        frame.draw(Sprite::solid(Color::WHITE).rect(310.0, 540.0, 180.0, 18.0));

        assert!(!should_auto_use_gpu(&frame));
        backend.render(&atlas, &frame).unwrap();
        assert!(!backend.is_using_gpu());

        frame.begin(800, 600, 1.0 / 60.0, 1.0 / 60.0, 1);
        frame.clear(Color::BLACK);
        frame.draw(Sprite::solid(Color::WHITE).rect(0.0, 0.0, 800.0, 600.0));
        assert!(should_auto_use_gpu(&frame));

        if GpuGameRenderer::new(&atlas).is_ok() {
            backend.render(&atlas, &frame).unwrap();
            assert!(backend.is_using_gpu());
        }
    }

    #[test]
    fn dirty_tracker_handles_full_resize_clear_move_remove_and_clip() {
        let mut tracker = DirtyTracker::default();
        let mut frame = GameFrame::new(4, 4);

        frame.begin(4, 4, 0.0, 0.0, 0);
        frame.clear(Color::BLACK);
        assert_eq!(tracker.next_dirty(&frame), Some(DirtyRect::full(4, 4)));

        frame.begin(4, 4, 0.0, 0.0, 1);
        frame.clear(Color::BLACK);
        assert_eq!(tracker.next_dirty(&frame), None);

        frame.begin(5, 4, 0.0, 0.0, 2);
        frame.clear(Color::BLACK);
        assert_eq!(tracker.next_dirty(&frame), Some(DirtyRect::full(5, 4)));

        frame.begin(5, 4, 0.0, 0.0, 3);
        frame.clear(Color::rgb(1, 2, 3));
        assert_eq!(tracker.next_dirty(&frame), Some(DirtyRect::full(5, 4)));

        frame.begin(5, 4, 0.0, 0.0, 4);
        frame.clear(Color::rgb(1, 2, 3));
        frame.draw(Sprite::solid(Color::WHITE).rect(1.0, 1.0, 2.0, 2.0));
        assert_eq!(
            tracker.next_dirty(&frame),
            Some(DirtyRect {
                x: 1,
                y: 1,
                width: 2,
                height: 2,
            })
        );

        frame.begin(5, 4, 0.0, 0.0, 5);
        frame.clear(Color::rgb(1, 2, 3));
        frame.draw(Sprite::solid(Color::WHITE).rect(2.0, 1.0, 2.0, 2.0));
        assert_eq!(
            tracker.next_dirty(&frame),
            Some(DirtyRect {
                x: 1,
                y: 1,
                width: 3,
                height: 2,
            })
        );

        frame.begin(5, 4, 0.0, 0.0, 6);
        frame.clear(Color::rgb(1, 2, 3));
        assert_eq!(
            tracker.next_dirty(&frame),
            Some(DirtyRect {
                x: 2,
                y: 1,
                width: 2,
                height: 2,
            })
        );

        frame.begin(5, 4, 0.0, 0.0, 7);
        frame.clear(Color::rgb(1, 2, 3));
        frame.draw(Sprite::solid(Color::WHITE).rect(-1.0, -1.0, 3.0, 3.0));
        frame.draw(Sprite::solid(Color::WHITE).rect(3.0, 2.0, 10.0, 10.0));
        assert_eq!(
            tracker.next_dirty(&frame),
            Some(DirtyRect {
                x: 0,
                y: 0,
                width: 5,
                height: 4,
            })
        );
    }

    #[test]
    fn incremental_cpu_matches_full_cpu_across_dirty_frames() {
        let atlas = test_atlas();
        let mut incremental = CpuGameRenderer::default();
        let mut full = CpuGameRenderer::default();
        let mut tracker = DirtyTracker::default();
        let mut frame = GameFrame::new(4, 3);

        for frame_index in 0..4 {
            frame.begin(4, 3, frame_index as f32 / 60.0, 1.0 / 60.0, frame_index);
            match frame_index {
                0 => {
                    frame.clear(Color::BLACK);
                    frame.draw(Sprite::solid(Color::rgb(255, 0, 0)).rect(0.0, 0.0, 2.0, 2.0));
                }
                1 => {
                    frame.clear(Color::BLACK);
                    frame.draw(Sprite::image("two").rect(1.0, 1.0, 2.0, 1.0));
                }
                2 => {
                    frame.clear(Color::BLACK);
                    frame.draw(Sprite::solid(Color::rgba(0, 255, 0, 128)).rect(2.0, 0.0, 2.0, 3.0));
                }
                _ => {
                    frame.clear(Color::rgb(1, 2, 3));
                }
            }

            let dirty = tracker.next_dirty(&frame);
            let incremental_pixels = incremental
                .render_dirty(&atlas, &frame, dirty)
                .unwrap()
                .to_vec();
            let full_pixels = full.render(&atlas, &frame).unwrap().to_vec();
            assert_eq!(incremental_pixels, full_pixels);
        }
    }

    #[test]
    fn unknown_sprite_texture_reports_name() {
        let err = NeoGame::new()
            .window("test", 1, 1)
            .run_for_frames(1, |frame| {
                frame.draw(Sprite::image("missing"));
                Ok(())
            })
            .unwrap_err()
            .to_string();
        assert!(err.contains("sprite texture `missing` was not registered"));
    }

    #[test]
    fn cpu_clear_fills_frame() {
        let pixels = NeoGame::new()
            .window("test", 2, 2)
            .run_for_frames(1, |frame| {
                frame.clear(Color::rgb(10, 20, 30));
                Ok(())
            })
            .unwrap();
        assert_eq!(pixels, [30, 20, 10, 255].repeat(4));
    }

    #[test]
    fn cpu_blends_and_preserves_draw_order() {
        let pixels = NeoGame::new()
            .window("test", 1, 1)
            .run_for_frames(1, |frame| {
                frame.clear(Color::rgb(0, 0, 255));
                frame.draw(Sprite::solid(Color::rgba(255, 0, 0, 128)).rect(0.0, 0.0, 1.0, 1.0));
                frame.draw(Sprite::solid(Color::rgba(0, 255, 0, 128)).rect(0.0, 0.0, 1.0, 1.0));
                Ok(())
            })
            .unwrap();
        assert_eq!(pixels[3], 255);
        assert!(
            pixels[1] > pixels[2],
            "later green sprite should dominate red"
        );
    }

    #[test]
    fn cpu_clips_sprites_to_frame_edges() {
        let pixels = NeoGame::new()
            .window("test", 2, 2)
            .run_for_frames(1, |frame| {
                frame.clear(Color::BLACK);
                frame.draw(Sprite::solid(Color::WHITE).rect(-1.0, -1.0, 2.0, 2.0));
                Ok(())
            })
            .unwrap();
        assert_eq!(&pixels[0..4], &[255, 255, 255, 255]);
        assert_eq!(&pixels[4..8], &[0, 0, 0, 255]);
        assert_eq!(&pixels[8..12], &[0, 0, 0, 255]);
    }

    #[test]
    fn cpu_nearest_samples_texture_pixels() {
        let atlas = test_atlas();
        let mut frame = GameFrame::new(2, 1);
        frame.begin(2, 1, 0.0, 0.0, 0);
        frame.clear(Color::BLACK);
        frame.draw(Sprite::image("two").rect(0.0, 0.0, 2.0, 1.0));
        let mut renderer = CpuGameRenderer::default();
        let pixels = renderer.render(&atlas, &frame).unwrap();
        assert_eq!(&pixels[0..4], &[0, 0, 255, 255]);
        assert_eq!(&pixels[4..8], &[0, 255, 0, 255]);
    }

    #[test]
    fn gpu_sprite_kernel_compiles_when_cuda_is_available() {
        let Ok(context) = NeoContext::new_default_device() else {
            return;
        };
        if let Err(err) = neo_runtime::Module::from_cuda_source(&context, sprite_cuda_source()) {
            eprintln!("skipping GPU sprite compile check: {err:#}");
        }
    }

    #[test]
    fn gpu_renderer_matches_cpu_for_tiny_scene_when_available() {
        let atlas = test_atlas();
        let Ok(mut gpu) = GpuGameRenderer::new(&atlas) else {
            return;
        };
        let mut cpu = CpuGameRenderer::default();
        let mut frame = GameFrame::new(2, 1);
        frame.begin(2, 1, 0.0, 0.0, 0);
        frame.clear(Color::rgb(1, 2, 3));
        frame.draw(Sprite::image("two").rect(0.0, 0.0, 2.0, 1.0));
        let cpu_pixels = cpu.render(&atlas, &frame).unwrap().to_vec();
        let gpu_pixels = gpu.render(&atlas, &frame).unwrap().to_vec();
        assert_eq!(gpu_pixels, cpu_pixels);
    }

    #[test]
    fn dirty_gpu_renderer_matches_cpu_across_moving_frames_when_available() {
        let atlas = test_atlas();
        let Ok(mut gpu) = GpuGameRenderer::new(&atlas) else {
            return;
        };
        let mut cpu = CpuGameRenderer::default();
        let mut tracker = DirtyTracker::default();
        let mut frame = GameFrame::new(4, 2);

        for frame_index in 0..2 {
            frame.begin(4, 2, frame_index as f32 / 60.0, 1.0 / 60.0, frame_index);
            frame.clear(Color::rgb(1, 2, 3));
            frame.draw(Sprite::image("two").rect(frame_index as f32, 0.0, 2.0, 1.0));
            frame.draw(Sprite::solid(Color::rgba(0, 255, 0, 128)).rect(2.0, 1.0, 2.0, 1.0));

            let dirty = tracker.next_dirty(&frame);
            let gpu_pixels = gpu.render_dirty(&atlas, &frame, dirty).unwrap().to_vec();
            let cpu_pixels = cpu.render(&atlas, &frame).unwrap().to_vec();
            assert_eq!(gpu_pixels, cpu_pixels);
        }
    }
}
