//! Voice preset loading and discovery.
//!
//! Voice presets are stored as `.npz` files containing numpy float16 arrays.
//! The `ndarray-npy` crate doesn't support float16, so we manually parse the
//! `.npy` entries inside the zip.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use half::f16;
use ndarray::{ArrayD, IxDyn};

/// KV cache as flat `HashMap` of named f16 arrays.
pub type KvCache = HashMap<String, ArrayD<f16>>;

/// Load a voice preset from an `.npz` file.
///
/// # Errors
///
/// Returns an error if the file cannot be opened, entries cannot be read,
/// or any `.npy` array fails to parse.
pub fn load_voice_preset(path: &Path) -> Result<KvCache> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open voice preset: {}", path.display()))?;
    let mut zip = zip::ZipArchive::new(file).context("invalid zip archive")?;
    let mut cache = KvCache::new();

    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).context("failed to read zip entry")?;
        let name = entry
            .name()
            .strip_suffix(".npy")
            .unwrap_or(entry.name())
            .to_string();

        let mut buf = Vec::new();
        entry.read_to_end(&mut buf)?;

        let arr = parse_npy_f16(&buf)
            .with_context(|| format!("Failed to parse array '{name}' from {}", path.display()))?;
        cache.insert(name, arr);
    }

    Ok(cache)
}

/// Parse a `.npy` byte buffer containing float16 data into an `ArrayD<f16>`.
///
/// Numpy `.npy` v1 format:
///   - 6 bytes magic: `\x93NUMPY`
///   - 1 byte major version
///   - 1 byte minor version
///   - 2 bytes (v1) or 4 bytes (v2+) header length, little-endian
///   - Header: ASCII Python dict
///   - Raw data bytes
fn parse_npy_f16(buf: &[u8]) -> Result<ArrayD<f16>> {
    anyhow::ensure!(buf.len() >= 10, "NPY buffer too short");
    anyhow::ensure!(&buf[..6] == b"\x93NUMPY", "Invalid NPY magic");

    let major = buf[6];
    let header_len_offset: usize;
    let header_len: usize;

    if major == 1 {
        header_len = usize::from(u16::from_le_bytes([buf[8], buf[9]]));
        header_len_offset = 10;
    } else {
        anyhow::ensure!(buf.len() >= 12, "NPY v2+ buffer too short");
        header_len = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
        header_len_offset = 12;
    }

    let header_end = header_len_offset + header_len;
    anyhow::ensure!(buf.len() >= header_end, "NPY header extends past buffer");

    let header = std::str::from_utf8(&buf[header_len_offset..header_end])
        .context("NPY header is not valid UTF-8")?;

    let descr = extract_npy_field(header, "descr").context("Missing 'descr' in NPY header")?;
    anyhow::ensure!(
        descr == "<f2" || descr == "=f2" || descr == "|f2",
        "Unsupported NPY dtype '{descr}' — only float16 (<f2) is supported"
    );

    let shape = parse_npy_shape(header).context("Failed to parse 'shape' from NPY header")?;

    let n_elements: usize = shape.iter().product();
    let data_bytes = &buf[header_end..];
    anyhow::ensure!(
        data_bytes.len() >= n_elements * 2,
        "NPY data too short: expected {} bytes for {} f16 elements, got {}",
        n_elements * 2,
        n_elements,
        data_bytes.len()
    );

    let f16_data: Vec<f16> = data_bytes[..n_elements * 2]
        .chunks_exact(2)
        .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();

    Ok(ArrayD::from_shape_vec(IxDyn(&shape), f16_data)?)
}

/// Extract a string field from a numpy header dict.
fn extract_npy_field(header: &str, field: &str) -> Option<String> {
    let pattern = format!("'{field}': '");
    let start = header.find(&pattern)? + pattern.len();
    let end = header[start..].find('\'')? + start;
    Some(header[start..end].to_string())
}

/// Parse the shape tuple from a numpy header dict.
fn parse_npy_shape(header: &str) -> Option<Vec<usize>> {
    let start = header.find("'shape': (")? + "'shape': (".len();
    let end = header[start..].find(')')? + start;
    let shape_str = &header[start..end];

    if shape_str.trim().is_empty() {
        return Some(vec![]);
    }

    Some(
        shape_str
            .split(',')
            .filter_map(|s| {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    trimmed.parse::<usize>().ok()
                }
            })
            .collect(),
    )
}

/// Extract LM-style KV cache from voice preset with a given prefix.
#[must_use]
pub fn extract_kv(voice: &KvCache, prefix: &str, n_layers: usize) -> KvCache {
    let mut kv = KvCache::new();
    for i in 0..n_layers {
        let key_name = format!("{prefix}_key_{i}");
        let val_name = format!("{prefix}_value_{i}");
        if let Some(k) = voice.get(&key_name) {
            kv.insert(format!("key_{i}"), k.clone());
        }
        if let Some(v) = voice.get(&val_name) {
            kv.insert(format!("value_{i}"), v.clone());
        }
    }
    kv
}

/// Find `.npz` voice preset files in the model directory.
#[must_use]
pub fn list_voice_presets(model_dir: &Path) -> Vec<PathBuf> {
    let mut presets: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(model_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "npz") {
                presets.push(path);
            }
        }
    }
    presets.sort();
    presets
}

/// Resolve a speaker name to an `.npz` voice preset path.
///
/// Uses case-insensitive substring matching, falling back to the first preset.
///
/// # Errors
///
/// Returns an error if no `.npz` voice presets exist in `model_dir`.
pub fn resolve_voice(model_dir: &Path, speaker_name: &str) -> Result<PathBuf> {
    let presets = list_voice_presets(model_dir);
    if presets.is_empty() {
        anyhow::bail!("No .npz voice presets found in {}", model_dir.display());
    }

    let speaker_lower = speaker_name.to_lowercase();
    for p in &presets {
        let stem = p
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_lowercase();
        if stem.contains(&speaker_lower) {
            return Ok(p.clone());
        }
    }

    let chosen = presets[0].clone();
    println!(
        "  No match for '{}', using {}",
        speaker_name,
        chosen.file_name().unwrap_or_default().to_string_lossy()
    );
    Ok(chosen)
}
