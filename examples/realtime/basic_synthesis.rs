//! Basic Realtime TTS synthesis example.
//!
//! Generates speech from inline text or a text file using VibeVoice-Realtime-0.5B
//! and saves the result as a WAV file.
//!
//! Usage:
//!     cargo run --example basic_synthesis
//!     cargo run --example basic_synthesis -- --text "Hello from VibeVoice!"
//!     cargo run --example basic_synthesis -- --file data/text/vibevoice_intro.txt
//!     cargo run --example basic_synthesis -- --text "Good morning." --speaker carter

use std::path::PathBuf;
use std::process;

use anyhow::Result;
use clap::Parser;

use vibe_rust::realtime::{self, RealtimeConfig, RealtimeTts};
use vibe_rust::utils::{self, Timer};

const DEFAULT_TEXT: &str = "\
VibeVoice is an open-source family of frontier voice AI models from Microsoft. \
It includes automatic speech recognition, long-form multi-speaker text-to-speech, \
and a lightweight real-time streaming model with around two hundred millisecond latency.";

#[derive(Parser)]
#[command(about = "Synthesize speech with VibeVoice-Realtime")]
struct Cli {
    /// Inline text to speak
    #[arg(long)]
    text: Option<String>,

    /// Path to a .txt file to speak
    #[arg(long)]
    file: Option<PathBuf>,

    /// HF model id or local path
    #[arg(long, default_value = realtime::DEFAULT_MODEL)]
    model: String,

    /// Voice preset name
    #[arg(long, default_value = "carter")]
    speaker: String,

    /// Classifier-free guidance scale
    #[arg(long, default_value_t = 1.5)]
    cfg_scale: f32,

    /// Output WAV path
    #[arg(long, default_value = "output/basic_synthesis.wav")]
    output: PathBuf,

    /// Force device (cuda, mps, cpu)
    #[arg(long)]
    device: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // ---- resolve text --------------------------------------------------------
    let text = if let Some(t) = cli.text {
        t
    } else if let Some(ref f) = cli.file {
        if !f.exists() {
            eprintln!("Error: file not found: {}", f.display());
            process::exit(1);
        }
        std::fs::read_to_string(f)?.trim().to_string()
    } else {
        DEFAULT_TEXT.to_string()
    };

    println!("Input text ({} chars):", text.chars().count());
    println!("  {}", utils::truncate_str(&text, 200));
    println!();

    // ---- load model ----------------------------------------------------------
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
            eprintln!("\nCould not load Realtime TTS model: {e}");
            eprintln!(
                "Hint: download first with:\n  \
                 hf download nenad1002/microsoft-vibevoice-0.5B-onnx-fp16"
            );
            return Ok(());
        }
    };

    // ---- synthesize ----------------------------------------------------------
    println!("\nSynthesizing...");
    let result = {
        let _timer = Timer::new("synthesis");
        tts.synthesize(&text, &cli.speaker, cli.cfg_scale, Some(&cli.output))?
    };

    // ---- report --------------------------------------------------------------
    println!("\nAudio duration : {:.2}s", result.duration_secs);
    println!("Generation time: {:.2}s", result.generation_time_secs);
    let speed = if result.rtf <= 1.0 {
        "realtime"
    } else {
        "slower than realtime"
    };
    println!("RTF            : {:.2}x  ({speed})", result.rtf);

    if let Some(ref p) = result.output_path {
        let abs = std::fs::canonicalize(p).unwrap_or_else(|_| p.clone());
        println!("Output file    : {}", abs.display());
    }

    Ok(())
}
