//! Main generation loop for speech synthesis.
//!
//! Orchestrates text windowing, speech generation, EOS detection,
//! and vocoder decoding.

use anyhow::{Context, Result};
use half::f16;
use ndarray::{Array2, Array3};
use ort::session::Session;

use crate::constants;
use crate::onnx_backend::config::{DpmSchedule, OnnxConfig};
use crate::onnx_backend::dpm_solver::sample_speech_token;
use crate::onnx_backend::sessions::OnnxSessions;
use crate::onnx_backend::voice::{KvCache, extract_kv};

/// Text LM output: hidden states + updated KV cache.
type TextLmOutput = (Array3<f16>, KvCache);

/// TTS LM output: hidden states + EOS logit + updated KV cache.
type TtsLmOutput = (Array3<f16>, f32, KvCache);

/// Generate speech audio from text using the ONNX pipeline.
///
/// Returns raw f32 PCM samples at 24 kHz.  All latents are decoded in a
/// single vocoder pass for maximum audio quality.
///
/// # Errors
///
/// Returns an error if tokenization, any ONNX session run, or vocoder
/// decoding fails.
pub fn generate(
    sessions: &mut OnnxSessions,
    tokenizer: &tokenizers::Tokenizer,
    config: &OnnxConfig,
    schedule: &DpmSchedule,
    voice: &KvCache,
    text: &str,
    cfg_scale: f64,
) -> Result<Option<Vec<f32>>> {
    generate_core::<fn(&[f32])>(
        sessions, tokenizer, config, schedule, voice, text, cfg_scale, None,
    )
}

/// Streaming variant of [`generate`].
///
/// After each text window produces speech tokens, the new latents from that
/// window are decoded through the vocoder and the resulting audio chunk is
/// passed to `on_chunk`.  This lets callers begin playback while generation
/// continues.
///
/// Returns the **total** audio (concatenated from all chunks).
///
/// # Audio quality note
///
/// Because the vocoder (HiFi-GAN style) relies on convolutional context from
/// neighbouring frames, decoding latents in per-window chunks produces
/// slightly different output than a single full-sequence decode.  Boundary
/// artefacts (small clicks or discontinuities between chunks) may be audible.
/// Use [`generate`] when offline quality matters more than latency.
#[allow(clippy::too_many_arguments)]
pub fn generate_streaming<F>(
    sessions: &mut OnnxSessions,
    tokenizer: &tokenizers::Tokenizer,
    config: &OnnxConfig,
    schedule: &DpmSchedule,
    voice: &KvCache,
    text: &str,
    cfg_scale: f64,
    on_chunk: F,
) -> Result<Option<Vec<f32>>>
where
    F: FnMut(&[f32]),
{
    generate_core(
        sessions,
        tokenizer,
        config,
        schedule,
        voice,
        text,
        cfg_scale,
        Some(on_chunk),
    )
}

/// Shared generation core for batch and streaming modes.
///
/// When `on_chunk` is `Some`, latents are decoded incrementally after each
/// text window (streaming).  When `None`, all latents are decoded in one
/// vocoder pass at the end (batch).
#[allow(clippy::similar_names, clippy::too_many_arguments)]
fn generate_core<F>(
    sessions: &mut OnnxSessions,
    tokenizer: &tokenizers::Tokenizer,
    config: &OnnxConfig,
    schedule: &DpmSchedule,
    voice: &KvCache,
    text: &str,
    cfg_scale: f64,
    mut on_chunk: Option<F>,
) -> Result<Option<Vec<f32>>>
where
    F: FnMut(&[f32]),
{
    let text = sanitize_text(text);
    let text_with_nl = text.trim().to_string() + constants::TEXT_SUFFIX_NEWLINE;

    let encoding = tokenizer
        .encode(text_with_nl.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenizer error: {e}"))?;
    let token_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| i64::from(id)).collect();
    let total_tokens = token_ids.len();

    if total_tokens == 0 {
        return Ok(None);
    }

    let mut lm_kvs = extract_kv(voice, "lm", config.text_lm_layers);
    let mut tts_kvs = extract_kv(voice, "tts_lm", config.tts_lm_layers);
    let mut neg_tts_kvs = extract_kv(voice, "neg_tts_lm", config.tts_lm_layers);

    let neg_tts_hidden_raw = voice
        .get("neg_tts_lm_hidden")
        .context("Missing neg_tts_lm_hidden in voice preset")?;
    let neg_hidden_shape = neg_tts_hidden_raw.shape();
    let mut neg_tts_hidden = Array3::<f16>::from_shape_vec(
        (
            neg_hidden_shape[0],
            neg_hidden_shape[1],
            neg_hidden_shape[2],
        ),
        neg_tts_hidden_raw.iter().copied().collect(),
    )?;

    let mut tts_hidden: Option<Array3<f16>> = None;
    let mut all_latents: Vec<Vec<f32>> = Vec::new();
    let mut decoded_up_to: usize = 0;
    let mut streamed_audio: Vec<f32> = Vec::new();
    let mut win_idx: usize = 0;
    let mut finished = false;

    let h = config.hidden_size;

    println!("Generating speech for {total_tokens} tokens...");

    while !finished {
        let start = win_idx * config.text_window_size;
        let end = ((win_idx + 1) * config.text_window_size).min(total_tokens);
        win_idx += 1;

        if start < total_tokens {
            let cur_ids: Vec<i64> = token_ids[start..end].to_vec();
            let seq_len = cur_ids.len();
            let ids_arr = Array2::from_shape_vec((1, seq_len), cur_ids)?;

            let (lm_hidden, new_lm_kvs) = run_text_lm(
                &mut sessions.text_lm,
                &ids_arr,
                &lm_kvs,
                config.text_lm_layers,
            )?;
            lm_kvs = new_lm_kvs;

            let tts_mask = Array2::from_elem((1, seq_len), 1_i64);
            let (tts_h, _eos, updated_tts_kvs) = run_tts_lm(
                &mut sessions.tts_lm,
                &lm_hidden,
                &tts_mask,
                &tts_kvs,
                config.tts_lm_layers,
            )?;
            tts_kvs = updated_tts_kvs;
            tts_hidden = Some(tts_h);
        }

        if tts_hidden.is_none() {
            continue;
        }

        let stop = generate_speech_tokens(
            sessions,
            config,
            schedule,
            h,
            cfg_scale,
            &mut tts_hidden,
            &mut tts_kvs,
            &mut neg_tts_kvs,
            &mut neg_tts_hidden,
            &mut all_latents,
            total_tokens,
            win_idx,
        )?;
        if stop {
            finished = true;
        }

        println!("  Window {}: {} speech tokens", win_idx, all_latents.len());

        // Streaming: decode new latents from this window immediately.
        if let Some(ref mut callback) = on_chunk {
            let new_count = all_latents.len();
            if new_count > decoded_up_to {
                let chunk_audio =
                    decode_latents_chunk(sessions, config, &all_latents[decoded_up_to..new_count])?;
                callback(&chunk_audio);
                streamed_audio.extend_from_slice(&chunk_audio);
                decoded_up_to = new_count;
            }
        }
    }

    // Batch mode: single full-sequence decode for best quality.
    if on_chunk.is_none() {
        if all_latents.is_empty() {
            return Ok(None);
        }
        return decode_latents(sessions, config, &all_latents);
    }

    // Streaming mode: return the concatenated chunks.
    if streamed_audio.is_empty() {
        Ok(None)
    } else {
        Ok(Some(streamed_audio))
    }
}

/// Inner speech-token generation loop for one text window.
///
/// Returns `true` if generation should stop (EOS or safety stop).
#[allow(clippy::too_many_arguments)]
fn generate_speech_tokens(
    sessions: &mut OnnxSessions,
    config: &OnnxConfig,
    schedule: &DpmSchedule,
    h: usize,
    cfg_scale: f64,
    tts_hidden: &mut Option<Array3<f16>>,
    tts_kvs: &mut KvCache,
    neg_tts_kvs: &mut KvCache,
    neg_tts_hidden: &mut Array3<f16>,
    all_latents: &mut Vec<Vec<f32>>,
    total_tokens: usize,
    win_idx: usize,
) -> Result<bool> {
    // Clone the current hidden state for the first iteration.
    // The caller guarantees tts_hidden is Some.
    let mut current_h = tts_hidden
        .as_ref()
        .expect("caller guarantees tts_hidden is Some")
        .clone();

    for _si in 0..config.speech_window_size {
        let last_idx = current_h.shape()[1] - 1;
        let pos_cond_vec: Vec<f16> = (0..h).map(|j| current_h[[0, last_idx, j]]).collect();
        let pos_cond = Array2::from_shape_vec((1, h), pos_cond_vec)?;

        let neg_last_idx = neg_tts_hidden.shape()[1] - 1;
        let neg_cond_vec: Vec<f16> = (0..h)
            .map(|j| neg_tts_hidden[[0, neg_last_idx, j]])
            .collect();
        let neg_cond = Array2::from_shape_vec((1, h), neg_cond_vec)?;

        let latent = sample_speech_token(
            &mut sessions.diffusion_head,
            schedule,
            &pos_cond,
            &neg_cond,
            cfg_scale,
            config.latent_dim,
        )?;
        all_latents.push(latent.clone());

        let latent_f16: Vec<f16> = latent.iter().map(|&v| f16::from_f32(v)).collect();
        let latent_3d = Array3::from_shape_vec((1, 1, config.latent_dim), latent_f16)?;
        let embed = run_connector(&mut sessions.acoustic_connector, &latent_3d)?;

        let speech_mask = Array2::from_elem((1, 1), 0_i64);

        let (new_tts_h, eos_logits, updated_tts_kvs) = run_tts_lm(
            &mut sessions.tts_lm,
            &embed,
            &speech_mask,
            tts_kvs,
            config.tts_lm_layers,
        )?;
        *tts_kvs = updated_tts_kvs;
        current_h.clone_from(&new_tts_h);
        *tts_hidden = Some(new_tts_h);

        let (new_neg_h, _neg_eos, updated_neg_kvs) = run_tts_lm(
            &mut sessions.tts_lm,
            &embed,
            &speech_mask,
            neg_tts_kvs,
            config.tts_lm_layers,
        )?;
        *neg_tts_kvs = updated_neg_kvs;
        *neg_tts_hidden = new_neg_h;

        let eos_prob = 1.0 / (1.0 + (-f64::from(eos_logits)).exp());
        let all_text_done = win_idx * config.text_window_size >= total_tokens;
        let min_speech = total_tokens.max(constants::MIN_SPEECH_TOKENS);
        if eos_prob > constants::EOS_PROB_THRESHOLD
            && all_latents.len() >= min_speech
            && all_text_done
        {
            println!(
                "  EOS at token {} (prob={:.3})",
                all_latents.len(),
                eos_prob
            );
            return Ok(true);
        }

        if all_latents.len() > total_tokens * constants::SAFETY_STOP_MULTIPLIER && all_text_done {
            println!("  Safety stop at {} speech tokens", all_latents.len());
            return Ok(true);
        }
    }

    Ok(false)
}

/// Decode collected latents into audio via the vocoder (single full pass).
fn decode_latents(
    sessions: &mut OnnxSessions,
    config: &OnnxConfig,
    all_latents: &[Vec<f32>],
) -> Result<Option<Vec<f32>>> {
    let n_frames = all_latents.len();
    println!("Decoding {n_frames} frames...");
    let audio = decode_latents_chunk(sessions, config, all_latents)?;
    Ok(Some(audio))
}

/// Decode a slice of latents into audio via the vocoder.
fn decode_latents_chunk(
    sessions: &mut OnnxSessions,
    config: &OnnxConfig,
    latents: &[Vec<f32>],
) -> Result<Vec<f32>> {
    let n_frames = latents.len();
    let mut latent_seq: Vec<f16> = Vec::with_capacity(n_frames * config.latent_dim);
    for lat in latents {
        for &v in lat {
            #[allow(clippy::cast_possible_truncation)]
            let scaled =
                (f64::from(v) / config.speech_scaling_factor - config.speech_bias_factor) as f32;
            latent_seq.push(f16::from_f32(scaled));
        }
    }
    let latent_arr = Array3::from_shape_vec((1, n_frames, config.latent_dim), latent_seq)?;
    run_vocoder(&mut sessions.vocoder, &latent_arr)
}

/// Run text LM with KV-cache.
fn run_text_lm(
    sess: &mut Session,
    input_ids: &Array2<i64>,
    past_kvs: &KvCache,
    n_layers: usize,
) -> Result<TextLmOutput> {
    let mut inputs: Vec<(String, ort::value::DynValue)> = Vec::new();

    inputs.push((
        "input_ids".to_string(),
        ort::value::Tensor::from_array(input_ids.clone())?.into_dyn(),
    ));

    for i in 0..n_layers {
        let key = past_kvs
            .get(&format!("key_{i}"))
            .context("Missing LM key cache")?;
        let val = past_kvs
            .get(&format!("value_{i}"))
            .context("Missing LM value cache")?;

        let k4 = reshape_kv_4d(key)?;
        let v4 = reshape_kv_4d(val)?;

        inputs.push((
            format!("past_key_{i}"),
            ort::value::Tensor::from_array(k4)?.into_dyn(),
        ));
        inputs.push((
            format!("past_value_{i}"),
            ort::value::Tensor::from_array(v4)?.into_dyn(),
        ));
    }

    let outputs = sess.run(inputs)?;

    let (hidden_shape, hidden_data) = outputs[0].try_extract_tensor::<f16>()?;
    let hidden_arr = extract_array_3d(hidden_shape, hidden_data)?;

    let mut new_kvs = KvCache::new();
    for i in 0..n_layers {
        let (k_shape, k_data) = outputs[1 + 2 * i].try_extract_tensor::<f16>()?;
        let (v_shape, v_data) = outputs[2 + 2 * i].try_extract_tensor::<f16>()?;
        new_kvs.insert(format!("key_{i}"), extract_array_dyn(k_shape, k_data)?);
        new_kvs.insert(format!("value_{i}"), extract_array_dyn(v_shape, v_data)?);
    }

    Ok((hidden_arr, new_kvs))
}

/// Run TTS LM with KV-cache.
fn run_tts_lm(
    sess: &mut Session,
    inputs_embeds: &Array3<f16>,
    tts_text_mask: &Array2<i64>,
    past_kvs: &KvCache,
    n_layers: usize,
) -> Result<TtsLmOutput> {
    let mut inputs: Vec<(String, ort::value::DynValue)> = Vec::new();

    inputs.push((
        "inputs_embeds".to_string(),
        ort::value::Tensor::from_array(inputs_embeds.clone())?.into_dyn(),
    ));
    inputs.push((
        "tts_text_mask".to_string(),
        ort::value::Tensor::from_array(tts_text_mask.clone())?.into_dyn(),
    ));

    for i in 0..n_layers {
        let key = past_kvs
            .get(&format!("key_{i}"))
            .context("Missing TTS LM key cache")?;
        let val = past_kvs
            .get(&format!("value_{i}"))
            .context("Missing TTS LM value cache")?;

        let k4 = reshape_kv_4d(key)?;
        let v4 = reshape_kv_4d(val)?;

        inputs.push((
            format!("past_key_{i}"),
            ort::value::Tensor::from_array(k4)?.into_dyn(),
        ));
        inputs.push((
            format!("past_value_{i}"),
            ort::value::Tensor::from_array(v4)?.into_dyn(),
        ));
    }

    let outputs = sess.run(inputs)?;

    let (hs_shape, hs_data) = outputs[0].try_extract_tensor::<f16>()?;
    let hidden_arr = extract_array_3d(hs_shape, hs_data)?;

    let (_eos_shape, eos_data) = outputs[1].try_extract_tensor::<f16>()?;
    let eos_val = eos_data[0].to_f32();

    let mut new_kvs = KvCache::new();
    for i in 0..n_layers {
        let (k_shape, k_data) = outputs[2 + 2 * i].try_extract_tensor::<f16>()?;
        let (v_shape, v_data) = outputs[3 + 2 * i].try_extract_tensor::<f16>()?;
        new_kvs.insert(format!("key_{i}"), extract_array_dyn(k_shape, k_data)?);
        new_kvs.insert(format!("value_{i}"), extract_array_dyn(v_shape, v_data)?);
    }

    Ok((hidden_arr, eos_val, new_kvs))
}

/// Run acoustic connector: latent → embedding for feedback loop.
fn run_connector(sess: &mut Session, latent: &Array3<f16>) -> Result<Array3<f16>> {
    let outputs = sess.run(ort::inputs![
        "latent" => ort::value::Tensor::from_array(latent.clone())?
    ])?;
    let (shape, data) = outputs[0].try_extract_tensor::<f16>()?;
    extract_array_3d(shape, data)
}

/// Run vocoder: latents → audio waveform.
fn run_vocoder(sess: &mut Session, latents: &Array3<f16>) -> Result<Vec<f32>> {
    let outputs = sess.run(ort::inputs![
        "latents" => ort::value::Tensor::from_array(latents.clone())?
    ])?;
    let (_shape, data) = outputs[0].try_extract_tensor::<f16>()?;
    Ok(data.iter().map(|v| v.to_f32()).collect())
}

/// Helper: extract a tensor output as an ndarray.
///
/// ONNX shape dimensions are `i64`; values are always small non-negative
/// integers, so truncation and sign loss are not a concern.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
fn extract_array_3d(shape: &ort::value::Shape, data: &[f16]) -> Result<Array3<f16>> {
    anyhow::ensure!(shape.len() == 3, "Expected 3D tensor, got {}D", shape.len());
    let (d0, d1, d2) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
    Ok(Array3::from_shape_vec((d0, d1, d2), data.to_vec())?)
}

#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
fn extract_array_dyn(shape: &ort::value::Shape, data: &[f16]) -> Result<ndarray::ArrayD<f16>> {
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    Ok(ndarray::ArrayD::from_shape_vec(
        ndarray::IxDyn(&dims),
        data.to_vec(),
    )?)
}

/// Reshape a potentially flat/dynamic KV array to 4D `[batch, heads, seq, dim]`.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
fn reshape_kv_4d(arr: &ndarray::ArrayD<f16>) -> Result<ndarray::Array4<f16>> {
    match arr.ndim() {
        4 => {
            let s = arr.shape();
            Ok(ndarray::Array4::from_shape_vec(
                (s[0], s[1], s[2], s[3]),
                arr.iter().copied().collect(),
            )?)
        }
        _ => {
            anyhow::bail!(
                "Expected 4D KV cache array, got {}D shape {:?}",
                arr.ndim(),
                arr.shape()
            );
        }
    }
}

/// Sanitize text: replace Unicode smart quotes with ASCII equivalents.
fn sanitize_text(text: &str) -> String {
    text.replace(['\u{2018}', '\u{2019}'], "'")
        .replace(['\u{201c}', '\u{201d}'], "\"")
}
