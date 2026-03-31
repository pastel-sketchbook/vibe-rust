//! VibeVoice-ASR CLI — quick transcription demo.
//!
//! Usage:
//!     cargo run --bin vibe-asr -- --audio data/sample.wav
//!     cargo run --bin vibe-asr                              # uses generated test tone

use std::path::PathBuf;
use std::process;

use anyhow::Result;
use clap::Parser;

use vibe_rust::asr::{self, AsrConfig, AsrModel};
use vibe_rust::constants;
use vibe_rust::utils::{self, Timer};

#[derive(Parser)]
#[command(about = "VibeVoice-ASR quick transcription demo")]
struct Cli {
    /// HF model id or local path
    #[arg(long, default_value = asr::DEFAULT_MODEL)]
    model: String,

    /// Path to an audio file (WAV)
    #[arg(long)]
    audio: Option<PathBuf>,

    /// Force device (cuda, mps, cpu)
    #[arg(long)]
    device: Option<String>,

    /// Maximum new tokens to generate
    #[arg(long, default_value_t = constants::ASR_MAX_TOKENS)]
    max_tokens: u32,

    /// Sampling temperature (0 = greedy)
    #[arg(long, default_value_t = constants::ASR_TEMPERATURE)]
    temperature: f32,

    /// Nucleus sampling threshold (1.0 = disabled)
    #[arg(long, default_value_t = constants::ASR_TOP_P)]
    top_p: f32,
}

fn resolve_audio(path: Option<PathBuf>) -> PathBuf {
    if let Some(p) = path {
        if !p.exists() {
            eprintln!("Error: audio file not found: {}", p.display());
            process::exit(1);
        }
        p
    } else {
        println!("No --audio provided; generating a 3s test tone ...");
        utils::generate_test_tone_default().unwrap_or_else(|e| {
            eprintln!("Failed to generate test tone: {e}");
            process::exit(1);
        })
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let audio_path = resolve_audio(cli.audio);

    // Show audio info
    let audio = utils::load_audio(&audio_path)?;
    println!(
        "\nAudio: {}  ({:.1}s, {} Hz)",
        audio_path.display(),
        audio.duration_secs(),
        audio.sample_rate
    );

    let config = AsrConfig {
        model_path: cli.model,
        device: match cli.device {
            Some(dev) => dev.parse()?,
            None => AsrConfig::default().device,
        },
        ..Default::default()
    };

    let asr = match AsrModel::load(&config) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("\nCould not load ASR model: {e}");
            eprintln!("Hint: the 7B model needs ~16 GB VRAM (CUDA) or ~28 GB RAM (CPU/MPS).");
            eprintln!(
                "Download it first with:  {}",
                constants::HF_DOWNLOAD_ASR_HINT
            );
            process::exit(0);
        }
    };

    let result = {
        let _timer = Timer::new("transcription");
        asr.transcribe(&audio_path, cli.max_tokens, cli.temperature, cli.top_p)?
    };

    asr::print_transcription(&result);
    Ok(())
}
