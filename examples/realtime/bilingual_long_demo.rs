//! Bilingual EN/KR long-form demo: learning AI voice generation.
//!
//! A conversational walkthrough mixing English (Emma) and Korean (kr-Spk2) voices,
//! covering what AI voice generation is, how it works, and how to get started
//! learning. Each segment is synthesized individually and then concatenated into
//! a single WAV file.
//!
//! Usage:
//!     cargo run --example bilingual_long_demo
//!     cargo run --example bilingual_long_demo -- --output-dir output/bilingual

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use vibe_rust::realtime::{self, OUTPUT_SR, RealtimeConfig, RealtimeTts};
use vibe_rust::utils::{self, Timer};

// ---------------------------------------------------------------------------
// Script -- alternating English / Korean segments
// ---------------------------------------------------------------------------

const EMMA: &str = "en-emma_woman";
const SPK2: &str = "kr-spk2_woman";

/// Silence between segments (seconds).
const PAUSE_BETWEEN: f32 = 0.8;
const PAUSE_SECTION: f32 = 1.5;

struct Segment {
    voice: &'static str,
    text: &'static str,
    label: &'static str,
    pause_after: f32,
}

static SCRIPT: &[Segment] = &[
    // -- Section 1: Introduction --
    Segment {
        voice: EMMA,
        text: "Hello everyone! Today we're going to talk about how to learn \
               AI voice generation. This is a rapidly growing field that combines \
               deep learning, signal processing, and linguistics to create \
               natural-sounding speech from text.",
        label: "EN: Introduction",
        pause_after: PAUSE_SECTION,
    },
    Segment {
        voice: SPK2,
        text: "안녕하세요, 여러분! 오늘은 AI 음성 생성을 배우는 방법에 대해 이야기해 \
               보겠습니다. 이 분야는 딥러닝, 신호 처리, 그리고 언어학을 결합하여 텍스트에서 \
               자연스러운 음성을 만들어내는 기술입니다.",
        label: "KR: Introduction",
        pause_after: PAUSE_SECTION,
    },
    // -- Section 2: What is TTS? --
    Segment {
        voice: EMMA,
        text: "So what exactly is text to speech? At its core, a TTS system takes \
               written text as input and produces an audio waveform that sounds like \
               a human speaking those words. Modern systems use neural networks to \
               learn the complex mapping from text to sound.",
        label: "EN: What is TTS",
        pause_after: PAUSE_BETWEEN,
    },
    Segment {
        voice: SPK2,
        text: "텍스트 투 스피치, 줄여서 TTS란 무엇일까요? 기본적으로 TTS 시스템은 \
               문자 텍스트를 입력받아 사람이 말하는 것처럼 들리는 오디오 파형을 생성합니다. \
               최신 시스템은 신경망을 사용하여 텍스트에서 소리로의 복잡한 매핑을 학습합니다.",
        label: "KR: What is TTS",
        pause_after: PAUSE_SECTION,
    },
    // -- Section 3: Key concepts --
    Segment {
        voice: EMMA,
        text: "If you want to learn voice generation, there are several key concepts \
               you should understand. First, tokenization. Just like language models \
               break text into tokens, speech models break audio into acoustic tokens \
               that represent small chunks of sound.",
        label: "EN: Tokenization",
        pause_after: PAUSE_BETWEEN,
    },
    Segment {
        voice: SPK2,
        text: "음성 생성을 배우려면 몇 가지 핵심 개념을 이해해야 합니다. 첫째, 토큰화입니다. \
               언어 모델이 텍스트를 토큰으로 분리하는 것처럼, 음성 모델은 오디오를 작은 소리 \
               조각을 나타내는 음향 토큰으로 분해합니다.",
        label: "KR: Tokenization",
        pause_after: PAUSE_BETWEEN,
    },
    Segment {
        voice: EMMA,
        text: "Second, you need to learn about vocoders. A vocoder converts the \
               intermediate representation produced by the model back into an actual \
               audio waveform. Popular vocoders include HiFi-GAN and WaveGlow.",
        label: "EN: Vocoders",
        pause_after: PAUSE_BETWEEN,
    },
    Segment {
        voice: SPK2,
        text: "둘째, 보코더에 대해 배워야 합니다. 보코더는 모델이 생성한 중간 표현을 실제 \
               오디오 파형으로 변환합니다. 대표적인 보코더로는 HiFi-GAN과 WaveGlow가 \
               있습니다.",
        label: "KR: Vocoders",
        pause_after: PAUSE_BETWEEN,
    },
    Segment {
        voice: EMMA,
        text: "Third, voice cloning and multi-speaker synthesis. These techniques allow \
               a single model to produce speech in many different voices. The model \
               learns a speaker embedding that captures the unique characteristics of \
               each voice.",
        label: "EN: Voice cloning",
        pause_after: PAUSE_BETWEEN,
    },
    Segment {
        voice: SPK2,
        text: "셋째, 음성 복제와 다화자 합성입니다. 이 기술들은 하나의 모델이 다양한 \
               목소리로 음성을 생성할 수 있게 합니다. 모델은 각 화자의 고유한 특성을 \
               담아내는 화자 임베딩을 학습합니다.",
        label: "KR: Voice cloning",
        pause_after: PAUSE_SECTION,
    },
    // -- Section 4: Getting started --
    Segment {
        voice: EMMA,
        text: "Now let me share a practical learning path. Start with the basics of \
               deep learning, especially sequence to sequence models and attention \
               mechanisms. PyTorch is a great framework to learn.",
        label: "EN: Learning path basics",
        pause_after: PAUSE_BETWEEN,
    },
    Segment {
        voice: SPK2,
        text: "실용적인 학습 경로를 공유해 드리겠습니다. 먼저 딥러닝의 기초부터 시작하세요. \
               특히 시퀀스 투 시퀀스 모델과 어텐션 메커니즘을 배우는 것이 중요합니다. \
               파이토치는 배우기 좋은 프레임워크입니다.",
        label: "KR: Learning path basics",
        pause_after: PAUSE_BETWEEN,
    },
    Segment {
        voice: EMMA,
        text: "Next, study open source TTS projects. Microsoft's VibeVoice is an \
               excellent starting point. It includes models for speech recognition, \
               long-form text-to-speech, and real-time streaming synthesis. The code \
               is well documented and the models are freely available on HuggingFace.",
        label: "EN: Open source projects",
        pause_after: PAUSE_BETWEEN,
    },
    Segment {
        voice: SPK2,
        text: "다음으로, 오픈소스 TTS 프로젝트를 공부하세요. 마이크로소프트의 바이브보이스는 \
               훌륭한 출발점입니다. 음성 인식, 장문 텍스트 투 스피치, 그리고 실시간 스트리밍 \
               합성을 위한 모델을 포함하고 있습니다. 코드가 잘 문서화되어 있고 모델은 \
               허깅페이스에서 무료로 사용할 수 있습니다.",
        label: "KR: Open source projects",
        pause_after: PAUSE_BETWEEN,
    },
    Segment {
        voice: EMMA,
        text: "Then try fine-tuning a pre-trained model on your own data. Even a small \
               dataset of a few hours of speech can produce surprisingly good results. \
               This hands-on experience is invaluable for understanding how the models \
               actually work.",
        label: "EN: Fine-tuning",
        pause_after: PAUSE_BETWEEN,
    },
    Segment {
        voice: SPK2,
        text: "그런 다음, 자신의 데이터로 사전 학습된 모델을 미세 조정해 보세요. 몇 시간 \
               분량의 작은 데이터셋으로도 놀라울 정도로 좋은 결과를 얻을 수 있습니다. \
               이러한 실습 경험은 모델이 실제로 어떻게 작동하는지 이해하는 데 매우 \
               중요합니다.",
        label: "KR: Fine-tuning",
        pause_after: PAUSE_SECTION,
    },
    // -- Section 5: Challenges and tips --
    Segment {
        voice: EMMA,
        text: "There are some challenges you should be aware of. Training voice \
               models requires significant GPU resources. An A100 or H100 GPU is \
               recommended for training, though inference can run on smaller hardware \
               including Apple Silicon.",
        label: "EN: Challenges - compute",
        pause_after: PAUSE_BETWEEN,
    },
    Segment {
        voice: SPK2,
        text: "주의해야 할 몇 가지 과제가 있습니다. 음성 모델 훈련에는 상당한 GPU 자원이 \
               필요합니다. 훈련에는 A100이나 H100 GPU가 권장되지만, 추론은 애플 실리콘을 \
               포함한 더 작은 하드웨어에서도 실행할 수 있습니다.",
        label: "KR: Challenges - compute",
        pause_after: PAUSE_BETWEEN,
    },
    Segment {
        voice: EMMA,
        text: "Data quality is another critical factor. Clean, well-recorded audio \
               with accurate transcriptions will give you much better results than \
               noisy data. Spend time on data preparation. It really makes a \
               difference.",
        label: "EN: Challenges - data quality",
        pause_after: PAUSE_BETWEEN,
    },
    Segment {
        voice: SPK2,
        text: "데이터 품질도 중요한 요소입니다. 정확한 전사와 함께 깨끗하고 잘 녹음된 \
               오디오가 노이즈가 많은 데이터보다 훨씬 좋은 결과를 제공합니다. \
               데이터 준비에 시간을 투자하세요. 정말 큰 차이를 만듭니다.",
        label: "KR: Challenges - data quality",
        pause_after: PAUSE_SECTION,
    },
    // -- Section 6: Closing --
    Segment {
        voice: EMMA,
        text: "To wrap up, AI voice generation is one of the most exciting areas in \
               machine learning right now. The technology is advancing rapidly, and \
               there has never been a better time to start learning. We hope this \
               overview has been helpful!",
        label: "EN: Closing",
        pause_after: PAUSE_BETWEEN,
    },
    Segment {
        voice: SPK2,
        text: "마무리하자면, AI 음성 생성은 현재 머신러닝에서 가장 흥미로운 분야 중 \
               하나입니다. 기술이 빠르게 발전하고 있으며, 배우기 시작하기에 이보다 좋은 \
               때는 없습니다. 이 개요가 도움이 되셨기를 바랍니다. 감사합니다!",
        label: "KR: Closing",
        pause_after: 0.0,
    },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return a Vec of zeros representing silence.
fn make_silence(duration_s: f32, sr: u32) -> Vec<f32> {
    vec![0.0_f32; (sr as f32 * duration_s) as usize]
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(about = "Bilingual EN/KR long-form TTS demo")]
struct Cli {
    /// HF model id or local path
    #[arg(long, default_value = realtime::DEFAULT_MODEL)]
    model: String,

    /// Directory for output WAV files
    #[arg(long, default_value = "output/bilingual")]
    output_dir: PathBuf,

    /// Classifier-free guidance scale
    #[arg(long, default_value_t = 1.5)]
    cfg_scale: f32,

    /// Force device (cuda, mps, cpu)
    #[arg(long)]
    device: Option<String>,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let cli = Cli::parse();

    std::fs::create_dir_all(&cli.output_dir)?;

    // ---- load model ----------------------------------------------------------
    println!("{}", "=".repeat(72));
    println!("  Bilingual EN/KR Demo: Learning AI Voice Generation");
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
    let mut tts = match RealtimeTts::load(config, &project_root) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("\nCould not load model: {e}");
            eprintln!(
                "Hint: download first with:\n  \
                 hf download nenad1002/microsoft-vibevoice-0.5B-onnx-fp16"
            );
            return Ok(());
        }
    };

    // ---- synthesize each segment ---------------------------------------------
    let mut audio_chunks: Vec<Vec<f32>> = Vec::new();
    let mut total_audio_s = 0.0_f64;
    let mut total_gen_s = 0.0_f64;
    let n_segments = SCRIPT.len();

    for (i, seg) in SCRIPT.iter().enumerate() {
        let idx = i + 1;
        let tag = if seg.voice == EMMA { "Emma" } else { "kr-Spk2" };
        println!("\n[{idx}/{n_segments}] {}  ({tag})", seg.label);
        println!("  {}", utils::truncate_str(seg.text, 90));

        let seg_file = cli.output_dir.join(format!("{idx:02}_{}.wav", seg.voice));
        let synth_result = {
            let _timer = Timer::new(seg.label);
            tts.synthesize(seg.text, seg.voice, cli.cfg_scale, Some(&seg_file))
        };

        match synth_result {
            Ok(result) => {
                let dur = result.duration_secs;
                let gen_time = result.generation_time_secs;
                let rtf = result.rtf;
                total_audio_s += dur;
                total_gen_s += gen_time;

                println!(
                    "  {dur:.2}s audio | {gen_time:.2}s gen | RTF {rtf:.2}x | {}",
                    seg_file.file_name().unwrap_or_default().to_string_lossy()
                );

                audio_chunks.push(result.audio);

                // Add silence gap between segments
                if seg.pause_after > 0.0 {
                    audio_chunks.push(make_silence(seg.pause_after, OUTPUT_SR));
                }
            }
            Err(e) => {
                println!("  ERROR: {e}");
            }
        }
    }

    // ---- concatenate all segments into one file ------------------------------
    if !audio_chunks.is_empty() {
        let combined: Vec<f32> = audio_chunks.into_iter().flatten().collect();
        let combined_path = cli.output_dir.join("full_bilingual_demo.wav");
        utils::save_audio(&combined, &combined_path, OUTPUT_SR)?;
        let combined_dur = combined.len() as f64 / OUTPUT_SR as f64;

        println!();
        println!("{}", "=".repeat(72));
        println!("  Summary");
        println!("{}", "=".repeat(72));
        println!("  Segments synthesized : {n_segments}");
        println!("  Total speech audio   : {total_audio_s:.1}s");
        println!("  Total generation time: {total_gen_s:.1}s");
        let avg_rtf = if total_audio_s > 0.0 {
            total_gen_s / total_audio_s
        } else {
            0.0
        };
        println!("  Average RTF          : {avg_rtf:.2}x");
        println!("  Combined file length : {combined_dur:.1}s (incl. pauses)");
        println!(
            "  Combined output      : {}",
            std::fs::canonicalize(&combined_path)
                .unwrap_or_else(|_| combined_path.clone())
                .display()
        );
        println!(
            "  Individual segments  : {}/",
            std::fs::canonicalize(&cli.output_dir)
                .unwrap_or_else(|_| cli.output_dir.clone())
                .display()
        );
        println!();
    }

    Ok(())
}
