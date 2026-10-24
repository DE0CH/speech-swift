# Breeze TTS 2 inference

Add the `BreezeTTS` SwiftPM product. The default bundle is
`aufklarer/Breeze-TTS-2-MLX-8bit`. The converted model and self-hosted audio are for
research and non-commercial use under the upstream model license.

```swift
import BreezeTTS

let model = try await BreezeTTSModel.fromPretrained()
let samples = try await model.generate(
    text: "The quick brown fox jumps over the lazy dog.",
    instruction: "A clear, calm English narrator.")

try await model.generateStreaming(
    text: "你好，欢迎使用语音合成。",
    instruction: "温柔清晰的普通话女声。"
) { chunk in
    // chunk.samples is mono Float32 PCM at chunk.sampleRate (24,000 Hz).
    // An empty final chunk can mark completion when no samples remain.
}
```

`fromBundle(_:)` loads a local exported directory without network access.
`fromPretrained(modelId:cacheDir:offlineMode:progressHandler:)` uses the package's
Hugging Face cache. One model instance permits one generation at a time; an
overlapping request fails. Cancellation and a non-EOS frame cap propagate errors.
The shared `SpeechGenerationModel` protocol supports plain synthesis and streaming.

```bash
speech speak "The quick brown fox jumps over the lazy dog." --engine breeze \
  --instruct "A clear, calm English narrator." --stream -o design.wav

speech speak "Hello, world!" --engine breeze \
  --breeze-model-id aufklarer/Breeze-TTS-2-MLX-bf16 -o bf16.wav

speech speak "The weather is beautiful today." --engine breeze \
  --voice-sample reference.wav --breeze-reference-text "The exact reference transcript." \
  --breeze-seed 42 -o clone.wav

speech speak "We need to discuss what happened." --engine breeze \
  --voice-sample reference.wav --breeze-reference-text "The exact reference transcript." \
  --instruct "Speak slowly, with a serious tone." --stream -o directed.wav
```

Input is English or Chinese; the model infers language from text. An optional
`--language` validates English/Chinese aliases and does not insert a language
token. Match the voice instruction to the text's language. A reference clip must
be nonempty, finite, no longer than 30 seconds, and accompanied by its complete
transcript. The CLI loads/resamples audio; the Swift API expects 24 kHz PCM.
`--clean-reference` optionally restores the reference with Sidon before encoding.

| Flag | Default | Meaning |
|---|---|---|
| `--breeze-model-id` | `aufklarer/Breeze-TTS-2-MLX-8bit` | Converted Hugging Face bundle |
| `--breeze-bundle-dir` | unset | Local bundle, overrides model ID |
| `--instruct` | unset | Voice design or reference voice direction |
| `--breeze-reference-text` | unset | Exact transcript of `--voice-sample` |
| `--breeze-temperature` | `0.9` | Temperature; zero is greedy |
| `--breeze-top-k` | `50` | Top-k; zero disables |
| `--breeze-top-p` | `1` | Nucleus threshold in `(0,1]` |
| `--breeze-cfg-scale` | `4` | Instruction guidance; one disables CFG |
| `--breeze-max-frames` | `750` | 80 ms frames; reaching the cap before EOS throws |
| `--breeze-seed` | `42` | Local sampling seed, including zero |
| `--first-chunk-frames` | `3` | Frames before first waveform chunk |
| `--chunk-frames` | `25` | Subsequent chunk size |
| `--stream` | off | Emit audio chunks and print first-audio latency |

For `--stream --play`, the CLI buffers one output chunk before playback.
Use `--chunk-frames 3` to reduce that playback buffer. The default 25-frame
chunks favor throughput; generation TTFA measures PCM delivery and excludes
playback buffering.

Inline vocal events follow upstream text syntax, for example `(sigh)` in English
or `[笑]` in Chinese. Batch files, preset speaker IDs, and speaker listing are
unsupported. Partial streamed audio may precede a later cancellation or frame-limit
error; consumers should only mark an utterance complete after the stream succeeds.

```bash
python3 scripts/export_breeze_mlx.py --bits 8 --output out/Breeze-TTS-2-MLX-8bit
python3 scripts/export_breeze_mlx.py --bits 0 --output out/Breeze-TTS-2-MLX-bf16
```

The exporter requires `mlx`, `numpy`, and `huggingface_hub`. `export_config.json`
records source revision, removed tensors, quantized layers, and file checksums.
Use the [benchmark procedure](../benchmarks/breeze-tts.md) before publishing a
converted bundle. Upstream NVIDIA latency claims do not describe this Swift port.

## Regression checks

The E2E suite loads the published default bundle when `BREEZE_E2E_BUNDLE` is
unset. It checks English and Chinese streaming against independent ASR, official
PyTorch numerical fixtures, frame-limit errors, and overlapping-request rejection
with recovery after callback cancellation. The CLI E2E test also uses the published
default unless a local bundle override is supplied.

After building tests and the debug metallib, run the model suites in isolation:

```bash
SPEECH_CLI="$PWD/.build/release/speech" E2E_ONLY_FILTER=Breeze E2E_SKIP_UNIT=1 \
  scripts/test_e2e_isolated.sh
```

Build the release `speech` product first for the CLI check. Set
`BREEZE_REFERENCE_WAV` and `BREEZE_REFERENCE_TEXT` to a reference file and its
exact transcript to include cloning and direction. Set `BREEZE_E2E_BUNDLE` to a
local BF16 or 8-bit export when testing a specific conversion. Run these checks
sequentially with other model tests and benchmarks.
