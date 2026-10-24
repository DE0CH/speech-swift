# Breeze TTS 2 benchmark

Measure the release Swift runtime with compiled MLX Metal shaders. Each synthesis
variant runs in its own process; independent ASR verification runs afterward in
a separate process. Do not overlap builds, tests, synthesis, or ASR with other
model-loading jobs.


## Measured results — 2026-10-24

Apple M5 Pro, 48 GiB unified memory, macOS 26.6.2. Native release Swift runtime,
compiled Metal shaders, seed 42, instruction guidance 4, first chunk 3 frames,
subsequent chunks 25 frames. A fresh process per case and a separate Qwen3-ASR
process afterward. Cached local model files; downloads and WAV writing are excluded.

English voice design: “The quick brown fox jumps over the lazy dog.” Each
8-bit request generated 3.20 seconds of speech; BF16 generated 3.68 seconds.
Warm values are medians of three requests.
“Cold” means the first request in a fresh process, with the system shader/file
caches already populated; it is not a cold-machine measurement.

| Variant | Load | First request TTFA | Warm TTFA | Warm RTF | Peak RSS | Peak physical footprint |
|---|---:|---:|---:|---:|---:|---:|
| 8bit | 1.38 s | 349 ms | 329 ms | 0.707 | 3.94 GiB | 8.22 GiB |
| bf16 | 1.39 s | 488 ms | 462 ms | 1.258 | 6.06 GiB | 10.75 GiB |

Other measured cases:

| Case | First request TTFA | Warm TTFA | RTF | Peak RSS | Peak physical footprint |
|---|---:|---:|---:|---:|---:|
| chinese-8bit | 349 ms | — | 0.721 | 3.92 GiB | 11.13 GiB |
| clone-8bit | 302 ms | 243 ms | 0.579 | 4.12 GiB | 11.24 GiB |
| direction-8bit | 425 ms | 395 ms | 0.717 | 4.11 GiB | 11.31 GiB |
| plain-8bit | 221 ms | — | 0.603 | 3.93 GiB | 7.86 GiB |

All **16 final clips from the published 8-bit and BF16 bundles** have
**zero normalized WER and CER** with the independent
Qwen3-ASR 0.6B 4-bit runtime. These are five different texts with deterministic
repeats, covering English and Chinese voice design, plain synthesis, reference
cloning, and voice direction. Cloning uses the verified English design output
as a synthetic reference; no external speaker's identity is used. This smoke
test does not estimate general text accuracy, voice identity, or instruction fidelity.

RSS alone undercounts some GPU mappings. Physical footprint includes live
allocations and reusable MLX buffers; the largest measured 8-bit case reached
11.31 GiB. Longer inputs, different chunk sizes, or other models in the same
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

[Raw timing, memory and transcripts](breeze-tts-2026-10-24.json). Every measured
WAV is byte-identical to the corresponding 2026-10-23 output after integrating
current main. The CI-equivalent unit run passed 1,739 tests with 41 skips and no
failures; three exporter checks and ten real-model E2E checks also passed.
The E2E run downloaded and loaded the published default, checked English and
Chinese round trips for both 8-bit and BF16, and exercised the streaming CLI,
numerical fixtures, reference modes, frame limits, and generation-lock recovery.

The [2026-10-23 archive](breeze-tts-2026-10-23.json) retains five additional
clips from a discontinued 4-bit export; that checkpoint is no longer published.

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
