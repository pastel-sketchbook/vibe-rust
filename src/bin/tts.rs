//! VibeVoice-TTS CLI — status check (placeholder).
//!
//! Usage:
//!     cargo run --bin vibe-tts

use vibe_rust::constants;
use vibe_rust::tts;

fn main() {
    println!("VibeVoice-TTS (1.5B) -- status");
    println!("{}", "=".repeat(constants::BANNER_SEPARATOR_WIDTH));
    tts::check_tts_status();
}
