# TTS Benchmark & Streaming Architecture

## Benchmark Results (M3 Pro, 2026-04-04)

Compared **kokoro-onnx** (ONNX Runtime, CPU) vs **mlx-audio** (MLX, Apple GPU) using Kokoro-82M.
Voice: `af_heart`, speed: `1.1x`, warmup: 2 runs, measured: 5 runs.

### kokoro-onnx (ONNX Runtime, CPU)

| Input | Chars | Mean | Min | Audio Duration | RTF |
|-------|-------|------|-----|----------------|-----|
| Short | 31 | 897 ms | 859 ms | 1.62s | 0.554x |
| Medium | 119 | 2,322 ms | 2,286 ms | 5.48s | 0.423x |
| Long | 361 | 7,670 ms | 6,577 ms | 21.10s | 0.364x |

### mlx-audio (MLX, Apple GPU)

| Input | Chars | Mean | Min | Audio Duration | RTF |
|-------|-------|------|-----|----------------|-----|
| Short | 31 | 173 ms | 165 ms | 2.10s | 0.083x |
| Medium | 119 | 443 ms | 436 ms | 6.10s | 0.073x |
| Long | 361 | 1,289 ms | 1,163 ms | 21.40s | 0.060x |

### Speedup: mlx-audio over kokoro-onnx

| Input | kokoro-onnx | mlx-audio | Speedup |
|-------|-------------|-----------|---------|
| Short | 897 ms | 173 ms | **5.18x** |
| Medium | 2,322 ms | 443 ms | **5.25x** |
| Long | 7,670 ms | 1,289 ms | **5.95x** |

### Streaming (mlx-audio)

Note: Kokoro generates audio per-sentence, so short/medium inputs produce 1 chunk.
Streaming benefits appear when the model splits long text into multiple sentences.

| Input | TTFC Mean | TTFC Min | Total Mean |
|-------|-----------|----------|------------|
| Short | 151 ms | 143 ms | 157 ms |
| Medium | 377 ms | 315 ms | 389 ms |
| Long | 1,300 ms | 1,265 ms | 1,323 ms |

## Key Findings

1. **mlx-audio is 5-6x faster** than kokoro-onnx on M3 Pro, achieving RTF of 0.06-0.08x (generating audio 12-17x faster than real-time).

2. **kokoro-onnx underutilizes hardware** — runs on CPU only, achieving RTF of 0.35-0.55x. No GPU/ANE acceleration available via the standard onnxruntime pip package on macOS ARM64.

3. **Kokoro splits text at sentence boundaries internally.** The model's `generate()` yields one result per sentence. This means sentence-level streaming is natural — we don't need to split text ourselves.

4. **mlx-audio first load is slow (~7s)** due to pipeline initialization (phonemizer, spacy model). After that, inference is fast. The model must be loaded once at server startup.

## Architecture Changes

### Before (sequential, full-response)

```
User speaks → VAD detects end → Send audio+image to server
→ LLM generates FULL response (2-5s)
→ TTS generates FULL audio from complete text (0.9-7.7s with kokoro-onnx)
→ Send complete WAV over WebSocket
→ Client plays audio
```

**Total time-to-first-audio: 3-13s**

### After (streaming, sentence-level)

```
User speaks → VAD detects end → Send audio+image to server
→ LLM generates FULL response (2-5s, can't stream due to tool-call mode)
→ Server splits response into sentences
→ For each sentence:
    → TTS generates audio chunk (~150-400ms with mlx-audio)
    → Send PCM chunk over WebSocket immediately
    → Client plays chunk (starts playing while next chunk generates)
```

**Total time-to-first-audio: 2-5s (LLM) + ~170ms (first sentence TTS)**

The LLM is the bottleneck now, not TTS. With mlx-audio, even a 4-sentence response completes TTS in ~600ms total.

### Progressive Audio Delivery

Instead of sending one large base64-encoded WAV:

1. Server sends `{"type": "audio_start", "sample_rate": 24000}` to signal streaming begins
2. Server sends `{"type": "audio_chunk", "audio": "<base64 PCM>"}` for each sentence
3. Server sends `{"type": "audio_end", "tts_time": 0.45}` when all chunks are sent
4. Client uses an AudioWorklet or buffered playback to play chunks as they arrive

### Dependencies

- **Remove**: `kokoro-onnx` (ONNX Runtime, CPU)
- **Add**: `mlx-audio` (MLX, Apple GPU), `misaki[en]` (phonemizer), `num2words`

The mlx-audio package pulls in `mlx`, `mlx-metal`, `transformers`, `torch` (for misaki/spacy). Total venv size increases but runtime performance is dramatically better.
