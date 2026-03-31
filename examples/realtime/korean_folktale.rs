//! Korean folktale narration: 해와 달이 된 오누이 (The Sun and the Moon).
//!
//! One of Korea's most beloved traditional stories, narrated entirely in Korean
//! using the kr-Spk0 female voice. A mother returning home is tricked by a tiger,
//! and her two children must escape by climbing a tree and praying to heaven.
//! The daughter becomes the sun, and the son becomes the moon.
//!
//! Segments are synthesized in parallel using a pool of ONNX model instances
//! (one per worker thread, controlled by `--workers`).
//!
//! Usage:
//!     cargo run --example korean_folktale
//!     cargo run --example korean_folktale -- --speaker kr-spk1_man
//!     cargo run --example korean_folktale -- --workers 4

use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use rayon::prelude::*;

use vibe_rust::realtime::{self, OUTPUT_SR, RealtimeConfig, RealtimeTts};
use vibe_rust::utils;

// ---------------------------------------------------------------------------
// Story script -- 해와 달이 된 오누이 (The Sun and the Moon)
// ---------------------------------------------------------------------------

const DEFAULT_SPEAKER: &str = "kr-spk0_woman";

/// Silence between segments (seconds).
const PAUSE_BETWEEN: f32 = 1.0;
const PAUSE_SECTION: f32 = 1.8;

struct Segment {
    text: &'static str,
    label: &'static str,
    pause_after: f32,
}

#[rustfmt::skip]
static STORY: &[Segment] = &[
    // -- 도입 (Opening) --
    Segment {
        text: "옛날 옛적에, 깊은 산골 마을에 어머니와 오누이가 살고 있었습니다. \
               아버지는 일찍 돌아가시고, 어머니는 먼 마을 잔치집에서 일을 하며 \
               두 남매를 키우고 있었습니다.",
        label: "도입: 산골 마을의 가족",
        pause_after: PAUSE_SECTION,
    },

    // -- 어머니의 귀갓길 (Mother's journey home) --
    Segment {
        text: "어느 날, 어머니는 잔치집에서 일을 마치고 떡을 한 함지박 \
               이고 집으로 돌아오고 있었습니다. 달빛 아래 산길을 걸어가는데, \
               갑자기 큰 호랑이 한 마리가 나타났습니다.",
        label: "어머니의 귀갓길",
        pause_after: PAUSE_BETWEEN,
    },
    Segment {
        text: "호랑이가 말했습니다. \
               떡 하나 주면 안 잡아먹지! \
               어머니는 무서워서 떡 하나를 주었습니다. \
               호랑이는 떡을 받아먹고 사라졌습니다.",
        label: "호랑이의 첫 번째 요구",
        pause_after: PAUSE_BETWEEN,
    },
    Segment {
        text: "그런데 다음 고개를 넘으니 호랑이가 또 나타났습니다. \
               떡 하나 주면 안 잡아먹지! \
               어머니는 또 떡을 하나 주었습니다. 이렇게 고개를 넘을 때마다 \
               호랑이가 나타나서 떡을 하나씩 빼앗아 갔습니다.",
        label: "반복되는 호랑이의 요구",
        pause_after: PAUSE_BETWEEN,
    },
    Segment {
        text: "마침내 떡이 모두 떨어졌습니다. \
               호랑이가 또 나타나 말했습니다. \
               떡이 없으면 너를 잡아먹겠다! \
               불쌍한 어머니는 결국 호랑이에게 잡혀 먹히고 말았습니다.",
        label: "어머니의 최후",
        pause_after: PAUSE_SECTION,
    },

    // -- 호랑이, 오누이를 찾아오다 (Tiger finds the children) --
    Segment {
        text: "호랑이는 어머니의 옷을 입고 머리에 수건을 쓰고 \
               오누이가 사는 집으로 찾아갔습니다. \
               애들아, 엄마 왔다. 문 열어라! \
               하고 문을 두드렸습니다.",
        label: "호랑이의 변장",
        pause_after: PAUSE_BETWEEN,
    },
    Segment {
        text: "오빠가 문틈으로 밖을 내다보았습니다. \
               우리 엄마 손은 이렇게 까칠까칠하지 않아요! \
               호랑이는 얼른 손에 밀가루를 묻히고 다시 왔습니다. \
               자, 보렴. 엄마 손이란다.",
        label: "오누이의 의심",
        pause_after: PAUSE_BETWEEN,
    },
    Segment {
        text: "누이동생이 문틈으로 다시 보았습니다. \
               우리 엄마 목소리는 이렇게 굵지 않아요! \
               하지만 호랑이는 목소리를 가늘게 바꾸어 \
               다시 말했습니다. 감기에 걸려서 그래. 어서 문 열어라.",
        label: "계속되는 의심과 속임수",
        pause_after: PAUSE_BETWEEN,
    },
    Segment {
        text: "결국 오누이는 문을 열고 말았습니다. \
               그 순간 호랑이의 정체가 드러났습니다! \
               오누이는 깜짝 놀라 뒷문으로 도망쳐 \
               마당의 큰 나무 위로 올라갔습니다.",
        label: "호랑이의 정체 발각",
        pause_after: PAUSE_SECTION,
    },

    // -- 나무 위의 오누이 (Children in the tree) --
    Segment {
        text: "호랑이가 나무 아래에서 올려다보며 물었습니다. \
               너희들 어떻게 올라갔니? \
               오빠가 꾀를 내어 말했습니다. \
               손에 참기름을 바르고 올라오면 돼요!",
        label: "오빠의 꾀",
        pause_after: PAUSE_BETWEEN,
    },
    Segment {
        text: "호랑이는 참기름을 손에 바르고 나무를 올라가려 했지만, \
               미끄러워서 자꾸 떨어졌습니다. \
               그때 누이동생이 그만 사실을 말해 버렸습니다. \
               도끼로 찍으면서 올라오면 돼요!",
        label: "누이동생의 실수",
        pause_after: PAUSE_BETWEEN,
    },
    Segment {
        text: "호랑이는 도끼를 가져와 나무를 찍으며 올라오기 시작했습니다. \
               쿵! 쿵! 쿵! \
               나무가 흔들리고, 호랑이는 점점 가까이 올라왔습니다. \
               오누이는 너무너무 무서웠습니다.",
        label: "호랑이가 나무를 올라옴",
        pause_after: PAUSE_SECTION,
    },

    // -- 하늘의 동아줄 (The rope from heaven) --
    Segment {
        text: "오누이는 하늘을 향해 간절히 빌었습니다. \
               하느님, 하느님! 저희를 살려 주시려거든 새 동아줄을 내려 주시고, \
               죽이시려거든 썩은 동아줄을 내려 주세요!",
        label: "하늘에 비는 오누이",
        pause_after: PAUSE_BETWEEN,
    },
    Segment {
        text: "그러자 하늘에서 튼튼하고 새하얀 동아줄이 내려왔습니다! \
               오누이는 동아줄을 꼭 잡고 하늘로 올라갔습니다. \
               점점 높이, 더 높이 올라가서 구름 위까지 올라갔습니다.",
        label: "새 동아줄이 내려옴",
        pause_after: PAUSE_BETWEEN,
    },
    Segment {
        text: "호랑이도 따라서 빌었습니다. \
               하느님, 나에게도 동아줄을 내려 주세요! \
               하늘에서 동아줄이 내려왔지만, \
               그것은 썩은 동아줄이었습니다.",
        label: "호랑이에게 내려온 썩은 동아줄",
        pause_after: PAUSE_BETWEEN,
    },
    Segment {
        text: "호랑이가 썩은 동아줄을 잡고 올라가다가 \
               뚝! 줄이 끊어져서 수수밭에 떨어져 죽고 말았습니다. \
               그래서 수수의 줄기가 빨간 것은 \
               호랑이의 피가 묻었기 때문이라고 합니다.",
        label: "호랑이의 최후",
        pause_after: PAUSE_SECTION,
    },

    // -- 결말 (Ending) --
    Segment {
        text: "하늘나라에 올라간 오누이는 \
               하느님으로부터 중요한 임무를 받았습니다. \
               누이는 해가 되어 세상을 환하게 비추고, \
               오빠는 달이 되어 밤을 밝히게 되었습니다.",
        label: "해와 달이 된 오누이",
        pause_after: PAUSE_BETWEEN,
    },
    Segment {
        text: "그런데 누이가 말했습니다. \
               저는 밤이 무서워요. 오빠가 밤을 맡아 주세요. \
               그래서 오빠가 달이 되고, 누이가 해가 되었다고 합니다.",
        label: "오빠와 누이의 역할 바꿈",
        pause_after: PAUSE_BETWEEN,
    },
    Segment {
        text: "해가 된 누이는 사람들이 자꾸 쳐다보는 것이 부끄러워서 \
               더욱 밝게 빛나기 시작했답니다. \
               그래서 우리가 해를 쳐다보면 눈이 부신 것이라고 합니다.",
        label: "해가 눈부신 이유",
        pause_after: PAUSE_BETWEEN,
    },
    Segment {
        text: "이렇게 착한 오누이는 하늘에서 해와 달이 되어 \
               오래오래 세상을 비추며 살았답니다. ",
        label: "이야기의 끝",
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

/// Result for a single synthesized segment.
struct SegmentResult {
    index: usize,
    audio: Vec<f32>,
    duration_secs: f64,
    generation_time_secs: f64,
    rtf: f64,
    filename: String,
    pause_after: f32,
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(about = "Korean folktale TTS: The Sun and the Moon")]
struct Cli {
    /// HF model id or local path
    #[arg(long, default_value = realtime::DEFAULT_MODEL)]
    model: String,

    /// Directory for output WAV files
    #[arg(long, default_value = "output/korean_folktale")]
    output_dir: PathBuf,

    /// Voice preset name
    #[arg(long, default_value = DEFAULT_SPEAKER)]
    speaker: String,

    /// Classifier-free guidance scale
    #[arg(long, default_value_t = 1.5)]
    cfg_scale: f32,

    /// Number of parallel worker instances (each loads its own ONNX sessions)
    #[arg(long, default_value_t = 2)]
    workers: usize,

    /// Force device (cuda, mps, cpu)
    #[arg(long)]
    device: Option<String>,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let cli = Cli::parse();
    let n_workers = cli.workers.max(1);

    std::fs::create_dir_all(&cli.output_dir)?;

    // ---- banner --------------------------------------------------------------
    println!("{}", "=".repeat(72));
    println!("  Korean Folktale: 해와 달이 된 오누이");
    println!("  (The Brother and Sister Who Became the Sun and Moon)");
    println!("  Narrator: {}", cli.speaker);
    println!("  Workers : {n_workers}");
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

    // ---- load worker pool ----------------------------------------------------
    println!("Loading {n_workers} model instance(s)...");
    let pool: Vec<Mutex<RealtimeTts>> = (0..n_workers)
        .map(|i| {
            if i > 0 {
                println!("  worker {}/{n_workers}...", i + 1);
            }
            let tts = RealtimeTts::load(config.clone(), &std::env::current_dir()?)?;
            Ok(Mutex::new(tts))
        })
        .collect::<Result<Vec<_>>>()?;

    // ---- synthesize segments in parallel --------------------------------------
    let n_segments = STORY.len();
    let wall_start = Instant::now();

    let results: Vec<Result<SegmentResult>> = STORY
        .par_iter()
        .enumerate()
        .map(|(i, seg)| {
            let idx = i + 1;
            let label_prefix = seg.label.split(':').next().unwrap_or(seg.label).trim();
            let seg_file = cli.output_dir.join(format!("{idx:02}_{label_prefix}.wav"));

            // Grab whichever worker is free (round-robin with blocking lock)
            let worker = &pool[i % pool.len()];
            let mut tts = worker.lock().expect("worker mutex poisoned");

            let result = tts.synthesize(seg.text, &cli.speaker, cli.cfg_scale, Some(&seg_file))?;

            let filename = seg_file
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();

            println!(
                "  [{idx:>2}/{n_segments}] {:.2}s audio | {:.2}s gen | RTF {:.2}x | {filename}",
                result.duration_secs, result.generation_time_secs, result.rtf,
            );

            Ok(SegmentResult {
                index: i,
                audio: result.audio,
                duration_secs: result.duration_secs,
                generation_time_secs: result.generation_time_secs,
                rtf: result.rtf,
                filename,
                pause_after: seg.pause_after,
            })
        })
        .collect();

    let wall_elapsed = wall_start.elapsed().as_secs_f64();

    // ---- collect results in story order --------------------------------------
    let mut ordered: Vec<SegmentResult> = Vec::with_capacity(n_segments);
    let mut errors = 0usize;
    for r in results {
        match r {
            Ok(seg) => ordered.push(seg),
            Err(e) => {
                eprintln!("  ERROR: {e}");
                errors += 1;
            }
        }
    }
    ordered.sort_by_key(|s| s.index);

    // ---- print per-segment summary in story order ----------------------------
    println!();
    for seg_result in &ordered {
        let idx = seg_result.index + 1;
        let label = STORY[seg_result.index].label;
        println!(
            "  [{idx:>2}/{n_segments}] {} | {:.2}s audio | {:.2}s gen | RTF {:.2}x | {}",
            utils::truncate_str(label, 24),
            seg_result.duration_secs,
            seg_result.generation_time_secs,
            seg_result.rtf,
            seg_result.filename,
        );
    }

    // ---- concatenate all segments into one file ------------------------------
    if !ordered.is_empty() {
        let mut combined: Vec<f32> = Vec::new();
        let mut total_audio_s = 0.0_f64;
        let mut total_gen_s = 0.0_f64;

        for seg_result in &ordered {
            total_audio_s += seg_result.duration_secs;
            total_gen_s += seg_result.generation_time_secs;
            combined.extend_from_slice(&seg_result.audio);
            if seg_result.pause_after > 0.0 {
                combined.extend_from_slice(&make_silence(seg_result.pause_after, OUTPUT_SR));
            }
        }

        let combined_path = cli.output_dir.join("해와_달이_된_오누이.wav");
        utils::save_audio(&combined, &combined_path, OUTPUT_SR)?;
        let combined_dur = combined.len() as f64 / OUTPUT_SR as f64;
        let avg_rtf = if total_audio_s > 0.0 {
            total_gen_s / total_audio_s
        } else {
            0.0
        };

        println!();
        println!("{}", "=".repeat(72));
        println!("  Summary");
        println!("{}", "=".repeat(72));
        println!("  Story               : 해와 달이 된 오누이");
        println!("  Narrator            : {}", cli.speaker);
        println!("  Workers             : {n_workers}");
        println!(
            "  Segments synthesized: {} ({errors} errors)",
            ordered.len()
        );
        println!("  Total speech audio  : {total_audio_s:.1}s");
        println!("  Sum of gen times    : {total_gen_s:.1}s");
        println!("  Wall-clock time     : {wall_elapsed:.1}s");
        let speedup = if wall_elapsed > 0.0 {
            total_gen_s / wall_elapsed
        } else {
            1.0
        };
        if n_workers > 1 {
            println!("  Parallel speedup    : {speedup:.2}x");
        }
        println!("  Average RTF         : {avg_rtf:.2}x");
        println!("  Combined length     : {combined_dur:.1}s (incl. pauses)");
        println!(
            "  Output file         : {}",
            std::fs::canonicalize(&combined_path)
                .unwrap_or_else(|_| combined_path.clone())
                .display()
        );
        println!();
    }

    Ok(())
}
