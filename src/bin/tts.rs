//! VibeVoice-TTS CLI — status check (placeholder).
//!
//! Usage:
//!     cargo run --bin vibe-tts

use vibe_rust::tts;

fn main() {
    println!("VibeVoice-TTS (1.5B) -- status");
    println!("{}", "=".repeat(40));
    tts::check_tts_status();
}
