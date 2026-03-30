//! Multi-voice, multi-language Realtime TTS demo.
//!
//! Generates speech in English, Spanish, and Korean using multiple voice presets
//! per language. Each utterance is saved as an individual WAV file, and a summary
//! table is printed at the end.
//!
//! Usage:
//!     cargo run --example multi_voice_demo
//!     cargo run --example multi_voice_demo -- --output-dir output/multi_voice
//!     cargo run --example multi_voice_demo -- --languages en sp
//!     cargo run --example multi_voice_demo -- --cfg-scale 2.0

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use vibe_rust::realtime::{self, RealtimeConfig, RealtimeTts};
use vibe_rust::utils::Timer;

// ---------------------------------------------------------------------------
// Demo utterances
// ---------------------------------------------------------------------------

struct Utterance {
    lang: &'static str,
    voice: &'static str,
    text: &'static str,
    label: &'static str,
}

fn lang_label(code: &str) -> &str {
    match code {
        "en" => "English",
        "sp" => "Spanish",
        "kr" => "Korean",
        _ => code,
    }
}

static UTTERANCES: &[Utterance] = &[
    // -- English (3 voices: Carter, Emma, Breeze) --
    Utterance {
        lang: "en",
        voice: "en-carter_man",
        text: "Welcome to the VibeVoice demo. This is Carter, speaking with a clear \
               and natural tone. The weather today is absolutely perfect for a walk.",
        label: "Carter (default male)",
    },
    Utterance {
        lang: "en",
        voice: "en-emma_woman",
        text: "Hi there! I'm Emma. VibeVoice can generate speech with distinct voices, \
               each with its own timbre and personality. Pretty cool, right?",
        label: "Emma (default female)",
    },
    Utterance {
        lang: "en",
        voice: "en-breeze_woman",
        text: "And I'm Breeze, one of the experimental voices. The streaming model runs \
               with around two hundred milliseconds of first-chunk latency.",
        label: "Breeze (experimental female)",
    },
    // -- Spanish (3 voices: Spk0 female, Spk1 male, Spk3 experimental male) --
    Utterance {
        lang: "sp",
        voice: "sp-spk0_woman",
        text: "Hola, bienvenidos a la demostración de VibeVoice. Soy la voz femenina \
               predeterminada para español. Espero que disfruten esta experiencia.",
        label: "Spk0 (default female)",
    },
    Utterance {
        lang: "sp",
        voice: "sp-spk1_man",
        text: "Buenos días. Soy la voz masculina predeterminada. Este modelo puede \
               generar audio en varios idiomas con voces naturales y expresivas.",
        label: "Spk1 (default male)",
    },
    Utterance {
        lang: "sp",
        voice: "sp-spk3_man",
        text: "Y yo soy una voz experimental masculina. La tecnología de síntesis de \
               voz ha avanzado mucho en los últimos años. Es increíble, verdad?",
        label: "Spk3 (experimental male)",
    },
    // -- Korean (3 voices: Spk0 female, Spk1 male, Spk2 experimental female) --
    Utterance {
        lang: "kr",
        voice: "kr-spk0_woman",
        text: "안녕하세요, 바이브보이스 데모에 오신 것을 환영합니다. 저는 기본 여성 목소리입니다.",
        label: "Spk0 (default female)",
    },
    Utterance {
        lang: "kr",
        voice: "kr-spk1_man",
        text: "반갑습니다. 저는 기본 남성 목소리입니다. 이 모델은 한국어를 포함한 여러 언어를 \
               지원합니다.",
        label: "Spk1 (default male)",
    },
    Utterance {
        lang: "kr",
        voice: "kr-spk2_woman",
        text: "저는 실험적 여성 목소리입니다. 음성 합성 기술이 정말 놀랍게 발전했습니다.",
        label: "Spk2 (experimental female)",
    },
];

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(about = "Multi-voice, multi-language VibeVoice-Realtime demo")]
struct Cli {
    /// HF model id or local path
    #[arg(long, default_value = realtime::DEFAULT_MODEL)]
    model: String,

    /// Directory for output WAV files
    #[arg(long, default_value = "output/multi_voice")]
    output_dir: PathBuf,

    /// Languages to include (en, sp, kr)
    #[arg(long, value_delimiter = ',', default_values_t = ["en".to_string(), "sp".to_string(), "kr".to_string()])]
    languages: Vec<String>,

    /// Classifier-free guidance scale
    #[arg(long, default_value_t = 1.5)]
    cfg_scale: f32,

    /// Force device (cuda, mps, cpu)
    #[arg(long)]
    device: Option<String>,
}

// ---------------------------------------------------------------------------
// Synthesis result tracking
// ---------------------------------------------------------------------------

struct UttResult {
    lang: String,
    voice: String,
    label: String,
    duration_s: Option<f64>,
    gen_time_s: Option<f64>,
    rtf: Option<f64>,
    #[allow(dead_code)]
    file: Option<String>,
    status: String,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Filter utterances to requested languages
    let selected: Vec<&Utterance> = UTTERANCES
        .iter()
        .filter(|u| cli.languages.iter().any(|l| l == u.lang))
        .collect();

    if selected.is_empty() {
        println!("No utterances matched the selected languages.");
        return Ok(());
    }

    std::fs::create_dir_all(&cli.output_dir)?;

    // ---- load model (once) ---------------------------------------------------
    println!("{}", "=".repeat(72));
    println!("  Multi-Voice Multi-Language VibeVoice-Realtime Demo");
    println!("{}", "=".repeat(72));
    println!();

    let config = RealtimeConfig {
        model_path: cli.model,
        device: match cli.device {
            Some(dev) => dev.parse()?,
            None => RealtimeConfig::default().device,
        },
        ..Default::default()
    };

    let project_root = std::env::current_dir()?;
    let tts = match RealtimeTts::load(config, &project_root) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("\nCould not load model: {e}");
            eprintln!(
                "Hint: download first with:\n  \
                 huggingface-cli download microsoft/VibeVoice-Realtime-0.5B"
            );
            return Ok(());
        }
    };

    // ---- synthesize each utterance -------------------------------------------
    let mut results: Vec<UttResult> = Vec::new();
    let mut total_audio = 0.0_f64;
    let mut total_gen = 0.0_f64;

    for (i, utt) in selected.iter().enumerate() {
        let idx = i + 1;
        let lang_name = lang_label(utt.lang);
        let filename = format!(
            "{}_{utt_voice}_{idx:02}.wav",
            utt.lang,
            utt_voice = utt.voice
        );
        let out_path = cli.output_dir.join(&filename);

        println!();
        println!(
            "--- [{idx}/{}] {lang_name} | {} ---",
            selected.len(),
            utt.label
        );
        println!("  Voice : {}", utt.voice);
        let text_preview_len = utt.text.len().min(80);
        let text_ellipsis = if utt.text.len() > 80 { "..." } else { "" };
        println!("  Text  : {}{text_ellipsis}", &utt.text[..text_preview_len]);

        let synth_result = {
            let _timer = Timer::new(utt.voice);
            tts.synthesize(utt.text, utt.voice, cli.cfg_scale, Some(&out_path))
        };

        match synth_result {
            Ok(result) => {
                let dur = result.duration_secs;
                let gen_time = result.generation_time_secs;
                let rtf = result.rtf;
                total_audio += dur;
                total_gen += gen_time;

                println!("  Audio : {dur:.2}s  |  Gen: {gen_time:.2}s  |  RTF: {rtf:.2}x");
                println!("  Saved : {}", out_path.display());

                results.push(UttResult {
                    lang: lang_name.to_string(),
                    voice: utt.voice.to_string(),
                    label: utt.label.to_string(),
                    duration_s: Some(dur),
                    gen_time_s: Some(gen_time),
                    rtf: Some(rtf),
                    file: Some(filename),
                    status: "OK".into(),
                });
            }
            Err(e) => {
                println!("  ERROR: {e}");
                results.push(UttResult {
                    lang: lang_name.to_string(),
                    voice: utt.voice.to_string(),
                    label: utt.label.to_string(),
                    duration_s: None,
                    gen_time_s: None,
                    rtf: None,
                    file: None,
                    status: format!("FAILED: {e}"),
                });
            }
        }
    }

    // ---- summary -------------------------------------------------------------
    println!();
    println!("{}", "=".repeat(72));
    println!("  Summary");
    println!("{}", "=".repeat(72));
    println!();
    println!(
        "{:>2}  {:<9} {:<22} {:<28} {:>5}  {:>5}  {:>6}  {}",
        "#", "Lang", "Voice", "Label", "Dur", "Gen", "RTF", "Status"
    );
    println!("{}", "-".repeat(100));

    for (i, r) in results.iter().enumerate() {
        let idx = i + 1;
        if r.status == "OK" {
            println!(
                "{idx:>2}  {:<9} {:<22} {:<28} {:>5.2}  {:>5.2}  {:>5.2}x  {}",
                r.lang,
                r.voice,
                r.label,
                r.duration_s.unwrap_or(0.0),
                r.gen_time_s.unwrap_or(0.0),
                r.rtf.unwrap_or(0.0),
                r.status,
            );
        } else {
            println!(
                "{idx:>2}  {:<9} {:<22} {:<28} {:>5}  {:>5}  {:>6}  {}",
                r.lang, r.voice, r.label, "--", "--", "--", r.status,
            );
        }
    }

    println!("{}", "-".repeat(100));
    let avg_rtf = if total_audio > 0.0 {
        total_gen / total_audio
    } else {
        0.0
    };
    println!(
        "Total audio: {total_audio:.2}s  |  Total gen: {total_gen:.2}s  \
         |  Avg RTF: {avg_rtf:.2}x"
    );
    println!(
        "Output dir : {}",
        std::fs::canonicalize(&cli.output_dir)
            .unwrap_or_else(|_| cli.output_dir.clone())
            .display()
    );
    println!();

    Ok(())
}
