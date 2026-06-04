#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchDims {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}

impl LaunchDims {
    pub fn for_2d(width: u32, height: u32, block: (u32, u32)) -> Self {
        let grid_x = width.div_ceil(block.0);
        let grid_y = height.div_ceil(block.1);
        Self {
            grid: (grid_x, grid_y, 1),
            block: (block.0, block.1, 1),
            shared_mem_bytes: 0,
        }
    }
}

impl From<LaunchDims> for LaunchConfig {
    fn from(value: LaunchDims) -> Self {
        Self {
            grid_dim: value.grid,
            block_dim: value.block,
            shared_mem_bytes: value.shared_mem_bytes,
        }
    }
}

#[derive(Debug)]
pub struct ImageBuffer {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl ImageBuffer {
    pub fn from_rgba(width: u32, height: u32, rgba: Vec<u8>) -> Result<Self, RuntimeError> {
        let expected = width as usize * height as usize * 4;
        let actual = rgba.len();
        if actual != expected {
            return Err(RuntimeError::InvalidImageBuffer {
                width,
                height,
                expected,
                actual,
            });
        }
        Ok(Self {
            width,
            height,
            rgba,
        })
    }

    pub fn save_png(&self, path: impl AsRef<Path>) -> Result<(), RuntimeError> {
        image::save_buffer_with_format(
            path,
            &self.rgba,
            self.width,
            self.height,
            image::ColorType::Rgba8,
            image::ImageFormat::Png,
        )?;
        Ok(())
    }
}

pub fn run_image_kernel(
    source: &str,
    width: u32,
    height: u32,
) -> Result<ImageBuffer, RuntimeError> {
    let ctx = Context::new_default_device()?;
    let module = ctx.compile_neo_module(source)?;
    let kernel = module.kernel("image")?;
    let mut pixels = ctx.alloc_zeros::<u8>(width as usize * height as usize * 4)?;
    let dims = LaunchDims::for_2d(width, height, (16, 16));

    {
        let mut launch = kernel.launcher();
        launch.arg_buffer_mut(&mut pixels);
        launch.arg(&width);
        launch.arg(&height);
        unsafe {
            launch.launch(dims)?;
        }
    }

    ctx.synchronize()?;
    ImageBuffer::from_rgba(width, height, pixels.download()?)
}
