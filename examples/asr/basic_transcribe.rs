//! Basic ASR transcription example.
//!
//! Transcribes a single audio file using VibeVoice-ASR and prints the
//! structured output (who spoke, when, and what they said).
//!
//! If no `--audio` is supplied a short test-tone WAV is generated automatically
//! so the script can run as a smoke test without real recordings.
//!
//! Usage:
//!     cargo run --example basic_transcribe
//!     cargo run --example basic_transcribe -- --audio data/sample.wav
//!     cargo run --example basic_transcribe -- --audio data/sample.wav --device cpu

use std::collections::HashSet;
use std::path::PathBuf;
use std::process;

use anyhow::Result;
use clap::Parser;

use vibe_rust::asr::{self, AsrConfig, AsrModel};
use vibe_rust::utils::{self, Timer};

#[derive(Parser)]
#[command(about = "Transcribe an audio file with VibeVoice-ASR")]
struct Cli {
    /// Path to audio file (WAV). Omit for a test tone.
    #[arg(long)]
    audio: Option<PathBuf>,

    /// HF model id or local path
    #[arg(long, default_value = asr::DEFAULT_MODEL)]
    model: String,

    /// Force device (cuda, mps, cpu)
    #[arg(long)]
    device: Option<String>,

    /// Maximum new tokens to generate
    #[arg(long, default_value_t = 32_768)]
    max_tokens: u32,

    /// Sampling temperature (0 = greedy)
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
}

fn resolve_audio(path: Option<PathBuf>) -> PathBuf {
    match path {
        Some(p) => {
            if !p.exists() {
                eprintln!("Error: audio file not found: {}", p.display());
                process::exit(1);
            }
            p
        }
        None => {
            println!("No --audio provided; generating a 3 s test tone ...");
            utils::generate_test_tone_default().unwrap_or_else(|e| {
                eprintln!("Failed to generate test tone: {e}");
                process::exit(1);
            })
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let audio_path = resolve_audio(cli.audio);

    // ---- load & inspect audio ------------------------------------------------
    let audio = utils::load_audio(&audio_path)?;
    println!("Audio : {}", audio_path.display());
    println!(
        "Length: {:.1}s  ({} Hz, {} samples)",
        audio.duration_secs(),
        audio.sample_rate,
        audio.samples.len()
    );
    println!();

    // ---- load model ----------------------------------------------------------
    let config = AsrConfig {
        model_path: cli.model,
        device: match cli.device {
            Some(dev) => dev.parse()?,
            None => AsrConfig::default().device,
        },
        ..Default::default()
    };

    let asr = match AsrModel::load(config) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("\nCould not load ASR model: {e}");
            eprintln!("Hint: the 7B model needs ~16 GB VRAM (CUDA) or ~28 GB RAM (CPU/MPS).");
            eprintln!("Download it first with:  hf download microsoft/VibeVoice-ASR");
            return Ok(());
        }
    };

    // ---- transcribe ----------------------------------------------------------
    println!("\nTranscribing...");
    let result = {
        let _timer = Timer::new("transcription");
        asr.transcribe(&audio_path, cli.max_tokens, cli.temperature)?
    };

    asr::print_transcription(&result);

    // ---- summary -------------------------------------------------------------
    let n_speakers = result
        .segments
        .iter()
        .map(|s| s.speaker_id.as_str())
        .collect::<HashSet<_>>()
        .len();
    println!(
        "\nSummary: {} segments, {} speaker(s)",
        result.segments.len(),
        n_speakers
    );

    Ok(())
}
