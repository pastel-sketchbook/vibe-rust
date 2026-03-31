# 0003 -- Parallel Synthesis with Rayon Worker Pool

**Status:** Accepted
**Date:** 2026-03-30

## Context

Multi-segment demos (bilingual long-form: 22 segments, Korean folktale: 20
segments) take significant wall-clock time when synthesized sequentially.
Each segment involves hundreds of autoregressive TTS LM steps plus diffusion
sampling plus vocoder decoding. We needed a parallelization strategy.

## Decision

Use a pool of independent `RealtimeTts` instances behind `Mutex`, distributed
across rayon's thread pool with `par_iter`.

## Rationale

### Why not share a single model?

The ONNX sessions (`ort::Session`) are not `Sync` -- they hold mutable
internal state (allocators, KV-cache buffers). Running two concurrent
`session.run()` calls on the same session would corrupt state. We need
separate session instances for concurrent synthesis.

### Why Mutex + rayon, not channels?

The synthesis workload is embarrassingly parallel -- each segment is
independent. rayon's `par_iter` maps perfectly: enumerate the segments,
distribute across threads, collect results. The `Mutex<RealtimeTts>` ensures
each worker exclusively owns its model instance during synthesis. Workers
grab whichever mutex is free, giving natural load balancing.

A channel-based approach (mpsc with dedicated worker threads) would work but
adds complexity for no benefit when the parallelism pattern is a simple
parallel map.

### Memory cost

Each `RealtimeTts` instance loads 5 ONNX sessions plus a tokenizer. On an
M4 Pro with the fp16 model, each instance uses roughly 1.5-2 GB of memory.
The default `--workers 2` keeps this manageable. The CLI exposes `--workers`
so users can scale based on available memory.

### Results

Korean folktale (20 segments, kr-spk0_woman):

- **Sequential (1 worker):** Sum of generation times is the wall-clock time.
- **2 workers:** Wall-clock drops significantly because two segments synthesize
  concurrently. The speedup is ~1.5-1.8x (not 2x due to memory bandwidth
  contention on shared L2/L3 cache).
- **4 workers:** Diminishing returns on machines with limited memory bandwidth.
  Useful on high-memory GPU systems.

### Implementation

```rust
// Build pool: one Mutex<RealtimeTts> per worker
let pool: Vec<Mutex<RealtimeTts>> = (0..n_workers)
    .map(|_| Ok(Mutex::new(RealtimeTts::load(config.clone(), &root)?)))
    .collect::<Result<Vec<_>>>()?;

// Parallel synthesis
let results: Vec<Result<SegmentResult>> = STORY
    .par_iter()
    .enumerate()
    .map(|(i, seg)| {
        let worker = &pool[i % pool.len()];
        let mut tts = worker.lock().unwrap();
        tts.synthesize(seg.text, &speaker, cfg_scale, Some(&path))
    })
    .collect();

// Reorder results (par_iter preserves order, but we sort by index for safety)
ordered.sort_by_key(|s| s.index);
```

The results are collected in story order, then concatenated with silence
gaps into a single combined WAV file.

## Tradeoffs

- **Memory:** N workers = N copies of the ONNX model in memory. On
  memory-constrained systems, `--workers 1` is the safe default.
- **Non-determinism:** Parallel execution means segments may be synthesized
  in any order. The output is deterministic (segments are reordered by
  index), but log output interleaves across workers.
- **Error handling:** If one segment fails, others continue. Errors are
  collected and reported in the summary. The combined WAV includes only
  successful segments.

## Consequences

- The `korean_folktale` and `bilingual_long_demo` examples both use this
  pattern (folktale with rayon, bilingual sequentially for simplicity).
- Future examples that synthesize multiple segments should follow the same
  pool pattern when parallelism is desired.
- The `--workers` flag should default to a conservative value (2) to avoid
  OOM on machines with limited RAM.
