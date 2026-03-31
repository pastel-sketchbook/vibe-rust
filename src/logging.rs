//! Logging configuration for vibe-rust.
//!
//! Provides a simple `init()` function that sets up `tracing` with a
//! configurable verbosity level via the `RUST_LOG` environment variable.
//!
//! Usage:
//! ```ignore
//! vibe_rust::logging::init(vibe_rust::logging::LogLevel::Info);
//! ```
//!
//! Or via environment variable:
//! ```sh
//! RUST_LOG=debug cargo run --bin vibe-realtime
//! ```

use std::sync::Once;

use tracing::level_filters::LevelFilter;
use tracing_subscriber::fmt;

/// Logging verbosity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogLevel {
    /// Only errors.
    Error,
    /// Warnings and errors.
    Warn,
    /// Info, warnings, and errors (default).
    #[default]
    Info,
    /// Debug-level and above.
    Debug,
    /// Trace-level and above (most verbose).
    Trace,
}

impl LogLevel {
    fn to_filter(self) -> LevelFilter {
        match self {
            LogLevel::Error => LevelFilter::ERROR,
            LogLevel::Warn => LevelFilter::WARN,
            LogLevel::Info => LevelFilter::INFO,
            LogLevel::Debug => LevelFilter::DEBUG,
            LogLevel::Trace => LevelFilter::TRACE,
        }
    }
}

static INIT: Once = Once::new();

/// Initialize logging with the given verbosity level.
///
/// Safe to call multiple times — subsequent calls are no-ops.
pub fn init(level: LogLevel) {
    INIT.call_once(|| {
        // Check RUST_LOG env var first; if set, use it directly.
        if std::env::var("RUST_LOG").is_ok() {
            fmt()
                .with_env_filter(
                    tracing_subscriber::EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
                )
                .with_target(false)
                .without_time()
                .init();
        } else {
            fmt()
                .with_max_level(level.to_filter())
                .with_target(false)
                .without_time()
                .init();
        }
    });
}

/// Initialize logging from the `RUST_LOG` environment variable.
/// Defaults to `Info` if the variable is not set.
pub fn init_from_env() {
    let level = std::env::var("RUST_LOG")
        .ok()
        .map(|v| match v.to_lowercase().as_str() {
            "error" => LogLevel::Error,
            "warn" | "warning" => LogLevel::Warn,
            "debug" => LogLevel::Debug,
            "trace" => LogLevel::Trace,
            _ => LogLevel::Info,
        })
        .unwrap_or_default();
    init(level);
}
