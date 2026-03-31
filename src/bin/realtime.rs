//! VibeVoice-Realtime TTS CLI — streaming synthesis demo.
//!
//! Usage:
//!     cargo run --bin vibe-realtime -- --text "Hello from VibeVoice!"
//!     cargo run --bin vibe-realtime -- --file data/text/vibevoice_intro.txt

use std::path::PathBuf;
use std::process;

use anyhow::Result;
use clap::Parser;

use vibe_rust::constants;
use vibe_rust::realtime::{self, RealtimeConfig, RealtimeTts};
use vibe_rust::utils::{self, Timer};

#[derive(Parser)]
#[command(about = "VibeVoice-Realtime TTS demo")]
struct Cli {
    /// HF model id or local path
    #[arg(long, default_value = realtime::DEFAULT_MODEL)]
    model: String,

    /// Inline text to synthesize
    #[arg(long)]
    text: Option<String>,

    /// Text file to synthesize
    #[arg(long)]
    file: Option<PathBuf>,

    /// Voice preset name
    #[arg(long, default_value = constants::DEFAULT_SPEAKER)]
    speaker: String,

    /// Classifier-free guidance scale
    #[arg(long, default_value_t = constants::DEFAULT_CFG_SCALE)]
    cfg_scale: f32,

    /// Output WAV path
    #[arg(long, default_value = "output/realtime_demo.wav")]
    output: PathBuf,

    /// Force device (cuda, mps, cpu)
    #[arg(long)]
    device: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Resolve text
    let text = if let Some(t) = cli.text {
        t
    } else if let Some(f) = cli.file {
        std::fs::read_to_string(&f)?
    } else {
        constants::DEFAULT_TEXT.to_string()
    };

    println!(
        "Text ({} chars): {}",
        text.chars().count(),
        utils::truncate_str(&text, constants::DEFAULT_TRUNCATE_LENGTH)
    );

    let config = RealtimeConfig {
        model_path: cli.model,
        device: match cli.device {
            Some(dev) => dev.parse()?,
            None => RealtimeConfig::default().device,
        },
        ..Default::default()
    };

    let project_root = std::env::current_dir()?;
    let mut tts = match RealtimeTts::load(config, &project_root) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("\nCould not load model: {e}");
            eprintln!("Download hint: {}", constants::HF_DOWNLOAD_ONNX_HINT);
            process::exit(1);
        }
    };

    let result = {
        let _timer = Timer::new("synthesis");
        tts.synthesize(&text, &cli.speaker, cli.cfg_scale, Some(&cli.output))?
    };

    println!("\nAudio duration : {:.2}s", result.duration_secs);
    println!("Generation time: {:.2}s", result.generation_time_secs);
    println!("RTF            : {:.2}x", result.rtf);
    if result.rtf < constants::RTF_REALTIME_THRESHOLD {
        println!("(realtime)");
    } else {
        println!("(slower than realtime)");
    }
    if let Some(p) = &result.output_path {
        let abs = std::fs::canonicalize(p).unwrap_or_else(|_| p.clone());
        println!("Saved to       : {}", abs.display());
    }

    Ok(())
}
