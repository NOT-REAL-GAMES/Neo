use std::{fs, path::PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "neo")]
#[command(about = "Neo graphics-kernel language prototype")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Report CUDA/NVRTC runtime availability.
    Doctor,
    /// Lower a .neo kernel source file to CUDA C.
    Compile {
        source: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Compile and run an image kernel, then write a PNG.
    Run {
        source: PathBuf,
        #[arg(long, default_value = "target/gradient.png")]
        out: PathBuf,
        #[arg(long, default_value_t = 1024)]
        width: u32,
        #[arg(long, default_value_t = 768)]
        height: u32,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Doctor => doctor_command(),
        Command::Compile { source, out } => compile_command(source, out),
        Command::Run {
            source,
            out,
            width,
            height,
        } => run_command(source, out, width, height),
    }
}

fn doctor_command() -> Result<()> {
    let diagnostics = neo_runtime::RuntimeDiagnostics::collect();
    println!(
        "CUDA driver: {}",
        if diagnostics.cuda_driver_available {
            "available"
        } else {
            "missing"
        }
    );
    if let Some(error) = &diagnostics.cuda_driver_error {
        println!("CUDA driver error: {error}");
    }
    println!(
        "NVRTC: {}",
        if diagnostics.nvrtc_loadable {
            "available"
        } else {
            "missing"
        }
    );
    if diagnostics.nvrtc_found.is_empty() {
        println!("{}", diagnostics.nvrtc_help());
    } else {
        for path in diagnostics.nvrtc_found {
            println!("NVRTC candidate: {}", path.display());
        }
    }
    Ok(())
}

fn compile_command(source_path: PathBuf, out: Option<PathBuf>) -> Result<()> {
    let source = fs::read_to_string(&source_path)
        .with_context(|| format!("failed to read {}", source_path.display()))?;
    let cuda = neo_lang::lower_to_cuda(&source)
        .map_err(|err| anyhow::anyhow!("failed to compile {}: {err}", source_path.display()))?;

    if let Some(out) = out {
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        fs::write(&out, cuda).with_context(|| format!("failed to write {}", out.display()))?;
    } else {
        print!("{cuda}");
    }
    Ok(())
}

fn run_command(source_path: PathBuf, out: PathBuf, width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        bail!("width and height must be greater than zero");
    }

    let source = fs::read_to_string(&source_path)
        .with_context(|| format!("failed to read {}", source_path.display()))?;
    let image = neo_runtime::run_image_kernel(&source, width, height)
        .map_err(|err| anyhow::anyhow!("failed to run {}: {err}", source_path.display()))?;

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    image
        .save_png(&out)
        .with_context(|| format!("failed to write {}", out.display()))?;
    println!("wrote {}", out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::doctor_command;

    #[test]
    fn bad_source_produces_diagnostic() {
        let source = "kernel image() {}";
        let err = neo_lang::lower_to_cuda(source).unwrap_err().to_string();
        assert!(err.contains("expected `fn`"));
    }

    #[test]
    fn doctor_command_does_not_panic() {
        doctor_command().unwrap();
    }
}
