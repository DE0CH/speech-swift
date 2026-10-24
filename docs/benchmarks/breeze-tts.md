# Breeze TTS 2 benchmark

Measure the release Swift runtime with compiled MLX Metal shaders. Each synthesis
variant runs in its own process; independent ASR verification runs afterward in
a separate process. Do not overlap builds, tests, synthesis, or ASR with other
model-loading jobs.


## Measured results — 2026-10-23

Apple M5 Pro, 48 GiB unified memory, macOS 26.6.2. Native release Swift runtime,
compiled Metal shaders, seed 42, instruction guidance 4, first chunk 3 frames,
subsequent chunks 25 frames. A fresh process per case and a separate Qwen3-ASR
process afterward. Cached local model files; downloads and WAV writing are excluded.

English voice design: “The quick brown fox jumps over the lazy dog.” Each
variant generated 3.20 seconds of speech. Warm values are medians of three requests.
“Cold” means the first request in a fresh process, with the system shader/file
caches already populated; it is not a cold-machine measurement.

| Variant | Load | First request TTFA | Warm TTFA | Warm RTF | Peak RSS | Peak physical footprint |
|---|---:|---:|---:|---:|---:|---:|
| 8bit | 1.36 s | 341 ms | 319 ms | 0.701 | 3.94 GiB | 8.22 GiB |
| bf16 | 1.47 s | 788 ms | 454 ms | 1.250 | 6.07 GiB | 10.76 GiB |

Other measured cases:

| Case | First request TTFA | Warm TTFA | RTF | Peak RSS | Peak physical footprint |
|---|---:|---:|---:|---:|---:|
| chinese-8bit | 340 ms | — | 0.694 | 3.93 GiB | 11.14 GiB |
| clone-8bit | 555 ms | 239 ms | 0.562 | 4.12 GiB | 11.24 GiB |
| direction-8bit | 549 ms | 381 ms | 0.707 | 4.12 GiB | 11.32 GiB |
| plain-8bit | 202 ms | — | 0.568 | 3.93 GiB | 7.86 GiB |

All **16 final clips from the published 8-bit and BF16 bundles** have
**zero normalized WER and CER** with the independent
Qwen3-ASR 0.6B 4-bit runtime. These are five different texts with deterministic
repeats, covering English and Chinese voice design, plain synthesis, reference
cloning, and voice direction. Cloning uses the verified English design output
as a synthetic reference; no external speaker's identity is used. This smoke
test does not estimate general text accuracy, voice identity, or instruction fidelity.

RSS alone undercounts some GPU mappings. Physical footprint includes live
allocations and reusable MLX buffers; the largest measured 8-bit case reached
11.32 GiB. Longer inputs, different chunk sizes, or other models in the same
process can change this. BF16 is retained for comparison; the default 8-bit
variant is faster on this machine.

The round trip exposed an incorrect pre-upsampling kernel assumption in the
shared codec setup. Breeze now installs the actual kernel-2/stride-2 stages
and checks every decode for exactly 1,920 samples per generated frame. The
initial run with shortened chunks is excluded from these final measurements.
CPU fixture tests verify the official text encoder and cached depth decoding
without M5's reduced-precision TF32 matmuls. Full-model reference cosines:

| Variant | Text features | Backbone | LM logits | Depth logits |
|---|---:|---:|---:|---:|
| BF16 | 0.999694 | 0.999960 | 0.999983 | 0.999969 |
| 8-bit | 0.999425 | 0.999938 | 0.999977 | 0.999930 |

[Raw timing, memory and transcripts](breeze-tts-2026-10-23.json).
The raw record also retains five clips from a discontinued 4-bit export;
that checkpoint is no longer published.

## Reproduce

```bash
python3 scripts/profiling/build_breeze_profile.py
.build/release/breeze-tts-bench --bundle out/Breeze-TTS-2-MLX-8bit \
  --output /tmp/breeze-8bit.json --warm-repeats 3
.build/release/breeze-tts-bench --verify /tmp/breeze-8bit.json \
  --output /tmp/breeze-8bit-asr.json
```

The synthesis JSON records model-load time, a cold request and warm repeats,
time to the first nonempty decoded audio chunk, total generation latency, audio
duration, RTF (wall seconds / audio seconds), RSS, kernel peak RSS, physical
footprint, kernel peak physical footprint, and MLX active/cache/peak bytes.
Model-load time excludes downloads. First-audio latency includes prompt encoding,
prefill, frame generation, and waveform decoding; it excludes model loading.
WAV writing follows the timing interval. The first request is cold for execution
and codec kernels; warm requests reuse the loaded model.

`--reference` and `--reference-text` measure reference cloning; add `--instruction`
for direction. Use `--instruction ""` for cloning without guidance. Reference-file
read/resampling occurs before request timing; reference codec encoding is inside
each measured request. `--first-chunk-frames`, `--chunk-frames`, `--cfg-scale`,
`--temperature`, `--seed`, and `--max-frames` specify the measured workload.

Round-trip verification uses `aufklarer/Qwen3-ASR-0.6B-MLX-4bit`, normalized WER
for English and CER for Chinese. It verifies speech content, not voice identity,
instruction adherence, acoustic quality, or absence of boundary artifacts. A
short-sentence smoke benchmark is not a general multilingual quality evaluation.
