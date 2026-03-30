//! TTS model status check.
//!
//! VibeVoice-TTS (1.5B) code was removed from the upstream repository due to
//! misuse concerns. The model weights remain available on HuggingFace at
//! `microsoft/VibeVoice-1.5B`, but there is no public inference code.
//!
//! This placeholder script checks for the model weights and prints status info.
//!
//! Usage:
//!     cargo run --example status

use vibe_rust::tts;

fn main() {
    println!("VibeVoice-TTS status check");
    println!("{}", "=".repeat(40));
    println!("Model : {}", tts::MODEL_ID);
    println!();
    tts::check_tts_status();
}
