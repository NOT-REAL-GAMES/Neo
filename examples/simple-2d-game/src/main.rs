use std::path::PathBuf;

use anyhow::Context;
use image::{Rgba, RgbaImage};
use neo_app::{Color, GameRenderer, NeoGame, Sprite};

fn main() -> anyhow::Result<()> {
    let options = ExampleOptions::parse()?;
    let player_path = write_player_texture()?;

    let mut x = 80.0;
    let mut y = 72.0;
    let mut vx = 210.0;
    let mut vy = 150.0;

    let mut game = NeoGame::new()
        .window("Neo Simple 2D Game", 800, 600)
        .texture("player", player_path)
        .target_fps(1000.0)
        .renderer(GameRenderer::Gpu);

    match options.fps {
        FpsSetting::Default => {}
        FpsSetting::Target(fps) => {
            game = game.target_fps(fps);
        }
        FpsSetting::Uncapped => {
            game = game.uncapped();
        }
    }

    if let Some(max_frames) = options.max_frames {
        game = game.max_frames(max_frames);
    }

    game.run(move |frame| {
        let dt = frame.delta_seconds().clamp(0.0, 1.0 / 20.0);
        let sprite_size = 64.0;

        x += vx * dt;
        y += vy * dt;

        let max_x = (frame.width() as f32 - sprite_size).max(0.0);
        let max_y = (frame.height() as f32 - sprite_size).max(0.0);

        if x <= 0.0 || x >= max_x {
            x = x.clamp(0.0, max_x);
            vx = -vx;
        }
        if y <= 0.0 || y >= max_y {
            y = y.clamp(0.0, max_y);
            vy = -vy;
        }

        let paddle_width = 180.0;
        let paddle_x = (frame.width() as f32 - paddle_width) * 0.5;
        let paddle_y = frame.height() as f32 - 56.0;

        frame.clear(Color::rgb(18, 22, 29));
        frame.draw(Sprite::solid(Color::rgba(75, 178, 151, 220)).rect(
            paddle_x,
            paddle_y,
            paddle_width,
            18.0,
        ));
        frame.draw(
            Sprite::image("player")
                .at(x, y)
                .size(sprite_size, sprite_size),
        );
        Ok(())
    })
}

struct ExampleOptions {
    max_frames: Option<u32>,
    renderer: GameRenderer,
    fps: FpsSetting,
}

enum FpsSetting {
    Default,
    Target(f32),
    Uncapped,
}

impl ExampleOptions {
    fn parse() -> anyhow::Result<Self> {
        let mut args = std::env::args().skip(1);
        let mut options = Self {
            max_frames: None,
            renderer: GameRenderer::Auto,
            fps: FpsSetting::Default,
        };

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--frames" => {
                    let value = args
                        .next()
                        .context("--frames requires a positive integer value")?;
                    options.max_frames = Some(
                        value
                            .parse()
                            .context("--frames must be a positive integer")?,
                    );
                }
                "--renderer" => {
                    let value = args
                        .next()
                        .context("--renderer requires auto, cpu, or gpu")?;
                    options.renderer = parse_renderer(&value)?;
                }
                "--fps" => {
                    let value = args.next().context("--fps requires a positive number")?;
                    let fps: f32 = value.parse().context("--fps must be a positive number")?;
                    if !fps.is_finite() || fps <= 0.0 {
                        anyhow::bail!("--fps must be finite and greater than zero");
                    }
                    options.fps = FpsSetting::Target(fps);
                }
                "--uncapped" => {
                    options.fps = FpsSetting::Uncapped;
                }
                "--help" | "-h" => {
                    println!(
                        "Usage: neo-simple-2d-game [--frames N] [--renderer auto|cpu|gpu] [--fps N|--uncapped]"
                    );
                    std::process::exit(0);
                }
                _ => anyhow::bail!("unknown argument `{arg}`"),
            }
        }

        Ok(options)
    }
}

fn parse_renderer(value: &str) -> anyhow::Result<GameRenderer> {
    match value {
        "auto" => Ok(GameRenderer::Auto),
        "cpu" => Ok(GameRenderer::Cpu),
        "gpu" => Ok(GameRenderer::Gpu),
        _ => anyhow::bail!("unknown renderer `{value}`; expected auto, cpu, or gpu"),
    }
}

fn write_player_texture() -> anyhow::Result<PathBuf> {
    let path = PathBuf::from("target/simple-2d-game/player.png");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create generated asset directory `{}`",
                parent.display()
            )
        })?;
    }

    let mut image = RgbaImage::from_pixel(16, 16, Rgba([0, 0, 0, 0]));
    for y in 0..16 {
        for x in 0..16 {
            let dx = x as f32 - 7.5;
            let dy = y as f32 - 7.5;
            let distance = (dx * dx + dy * dy).sqrt();
            let pixel = if distance <= 7.0 {
                Rgba([255, 197, 82, 255])
            } else if distance <= 7.75 {
                Rgba([198, 115, 48, 180])
            } else {
                Rgba([0, 0, 0, 0])
            };
            image.put_pixel(x, y, pixel);
        }
    }

    for (x, y) in [(5, 6), (10, 6)] {
        image.put_pixel(x, y, Rgba([30, 40, 52, 255]));
    }
    for x in 5..=10 {
        image.put_pixel(x, 10, Rgba([128, 66, 38, 255]));
    }

    image
        .save(&path)
        .with_context(|| format!("failed to write generated sprite `{}`", path.display()))?;
    Ok(path)
}
