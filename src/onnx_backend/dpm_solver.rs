//! DPM-Solver++ diffusion scheduler.
//!
//! Pure Rust implementation matching the PyTorch reference.
//! Maintains 2 parallel denoising trajectories (positive + negative) for
//! classifier-free guidance.

use anyhow::Result;
use half::f16;
use ndarray::Array2;
use ort::session::Session;
use rand::RngExt;
use rand_distr::StandardNormal;

use crate::constants;
use crate::onnx_backend::config::DpmSchedule;

/// Convert sigma to alpha and sigma_t.
pub fn sigma_to_alpha_sigma(sigma: f64) -> (f64, f64) {
    let alpha = 1.0 / (1.0 + sigma * sigma).sqrt();
    (alpha, sigma * alpha)
}

/// Convert sigma to lambda (log-SNR).
pub fn sigma_to_lambda(sigma: f64) -> f64 {
    let (alpha, sigma_t) = sigma_to_alpha_sigma(sigma);
    if sigma_t < constants::SIGMA_T_MIN {
        return constants::LAMBDA_MAX;
    }
    (alpha / sigma_t).ln()
}

/// Sample one speech latent via DPM-Solver++ with classifier-free guidance.
pub fn sample_speech_token(
    diff_sess: &mut Session,
    schedule: &DpmSchedule,
    pos_cond: &Array2<f16>,
    neg_cond: &Array2<f16>,
    cfg_scale: f64,
    latent_dim: usize,
) -> Result<Vec<f32>> {
    anyhow::ensure!(
        schedule.sigmas.len() > constants::DPM_SOLVER_STEPS,
        "schedule.sigmas must have at least {} entries, got {}",
        constants::DPM_SOLVER_STEPS + 1,
        schedule.sigmas.len()
    );
    anyhow::ensure!(
        schedule.timesteps.len() >= constants::DPM_SOLVER_STEPS,
        "schedule.timesteps must have at least {} entries, got {}",
        constants::DPM_SOLVER_STEPS,
        schedule.timesteps.len()
    );

    let mut rng = rand::rng();

    let noise: Vec<f64> = (0..constants::NOISE_DIM_MULTIPLIER * latent_dim)
        .map(|_| rng.sample::<f64, _>(StandardNormal))
        .collect();

    let mut sample = noise.clone();
    let mut m_list: Vec<Vec<f64>> = Vec::new();

    for step_idx in 0..constants::DPM_SOLVER_STEPS {
        let half: Vec<f16> = sample[..latent_dim]
            .iter()
            .map(|&v| f16::from_f64(v))
            .collect();
        let half_arr = Array2::from_shape_vec((1, latent_dim), half)?;
        let t_val = f16::from_f64(schedule.timesteps[step_idx] as f64);
        let t_arr = ndarray::Array1::from_vec(vec![t_val]);

        let cond_eps = run_diffusion(diff_sess, &half_arr, &t_arr, pos_cond)?;
        let uncond_eps = run_diffusion(diff_sess, &half_arr, &t_arr, neg_cond)?;

        let mut half_eps = Vec::with_capacity(latent_dim);
        for j in 0..latent_dim {
            let c = cond_eps[[0, j]].to_f64();
            let u = uncond_eps[[0, j]].to_f64();
            half_eps.push(u + cfg_scale * (c - u));
        }

        let mut v_pred = Vec::with_capacity(2 * latent_dim);
        v_pred.extend_from_slice(&half_eps);
        v_pred.extend_from_slice(&half_eps);

        let sig = schedule.sigmas[step_idx];
        let (alpha_s0, sigma_s0) = sigma_to_alpha_sigma(sig);

        let x0: Vec<f64> = sample
            .iter()
            .zip(v_pred.iter())
            .map(|(&s, &vp)| alpha_s0 * s - sigma_s0 * vp)
            .collect();
        m_list.push(x0.clone());

        let sig_next = schedule.sigmas[step_idx + 1];
        let (alpha_t, sigma_t) = sigma_to_alpha_sigma(sig_next);
        let lam_t = sigma_to_lambda(sig_next);
        let lam_s0 = sigma_to_lambda(sig);

        if constants::DPM_FIRST_ORDER_STEPS.contains(&step_idx) {
            let h = lam_t - lam_s0;
            let expm1_neg_h = (-h).exp_m1();
            for i in 0..sample.len() {
                sample[i] = (sigma_t / sigma_s0) * sample[i] - alpha_t * expm1_neg_h * x0[i];
            }
        } else {
            let sig_prev = schedule.sigmas[step_idx - 1];
            let lam_s1 = sigma_to_lambda(sig_prev);
            let h = lam_t - lam_s0;
            let h_0 = lam_s0 - lam_s1;
            let r0 = h_0 / h;
            let expm1_neg_h = (-h).exp_m1();
            let prev_x0 = &m_list[m_list.len() - 2];
            for i in 0..sample.len() {
                let d0 = x0[i];
                let d1 = (1.0 / r0) * (x0[i] - prev_x0[i]);
                sample[i] = (sigma_t / sigma_s0) * sample[i]
                    - alpha_t * expm1_neg_h * d0
                    - 0.5 * alpha_t * expm1_neg_h * d1;
            }
        }
    }

    Ok(sample[..latent_dim].iter().map(|&v| v as f32).collect())
}

/// Run diffusion head (single step): noisy_latent + timestep + condition → v_prediction.
fn run_diffusion(
    sess: &mut Session,
    noisy: &Array2<f16>,
    timestep: &ndarray::Array1<f16>,
    condition: &Array2<f16>,
) -> Result<Array2<f16>> {
    let outputs = sess.run(ort::inputs![
        "noisy_latent" => ort::value::Tensor::from_array(noisy.clone())?,
        "timestep" => ort::value::Tensor::from_array(timestep.clone())?,
        "condition" => ort::value::Tensor::from_array(condition.clone())?
    ])?;
    let (shape, data) = outputs[0].try_extract_tensor::<f16>()?;
    Ok(Array2::from_shape_vec(
        (shape[0] as usize, shape[1] as usize),
        data.to_vec(),
    )?)
}
