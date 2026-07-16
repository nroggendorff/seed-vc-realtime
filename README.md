# Seed-VC Realtime

Real-time voice conversion, split out of [Seed-VC](https://github.com/Plachtaa/Seed-VC) for easier
deployment and developed further since.

Upstream Seed-VC was **archived by its author on 2025-11-21** and is now read-only. This fork picked
up from there and has since reworked the real-time path fairly heavily — the audio engine, device
handling, VAD, and A/V latency are substantially changed. Everything model-related is still
upstream's work; see [Acknowledgements](#acknowledgements-).

## Features

-   **Real-time Voice Conversion**: Capable of cloning voices with 1~30 seconds of reference speech.
-   **Low Latency**: end-to-end ≈ `2 x block_time + extra_time_right + ~70ms of device`, and it is
    **measured** from the sound card rather than estimated. Measured here: ~690ms mic-to-speaker at
    `block_time = 0.3`, `extra_time_right = 0.02`.
-   **Inference is not on a deadline**: the pipeline runs on a worker thread, so a slow block costs
    latency, not dropped audio.
-   **Zero-shot**: No training required for new voices.

<details>
<summary><b>Changes since the fork</b> (upstream <code>51383ef</code>, 2025-04-20) — click to expand</summary>

### Latency & real-time behaviour

-   **Pipeline moved off the PortAudio callback** onto a worker thread behind ring buffers. It used
    to run resample → VAD → inference → SOLA inside the callback, where overrunning the deadline
    dropped audio outright.
-   **Audio blocksize decoupled from the model hop** — 20ms instead of one hop, since the callback
    no longer has to be long enough to infer in.

    Worth being precise about what that bought, because it is easy to overclaim: at a *fixed*
    `block_time` the two changes are roughly a wash. `2 x block_time` of PortAudio buffering is
    replaced by `2 x block_time` of pipeline priming, and one of those hops is the model waiting for
    its own input — algorithmic, not removable. What they actually buy is **headroom**: overrunning
    a block is now absorbed instead of dropped, which makes a lower `block_time` safe to use. Here
    inference runs 120-250ms depending on GPU contention, so `block_time` could come down from 0.5
    to 0.3 — and *that* took end-to-end latency from 1110ms to 690ms.
-   **Removed a VAD warmup spike**: every 10s the cache reset replayed 1.5s of audio through the
    model *inside the callback*. It was not load-bearing — a fresh FSMN-VAD cache cannot report a
    segment end before a start, so the speech gate stays open across the reset regardless.
-   An underrun now **discards** the audio that should have played rather than just emitting
    silence, so it costs a dropout instead of permanently growing the latency.
-   A pipeline error no longer kills the worker and leaves the stream emitting silence for good.

### A/V latency measurement

-   **Real mic-to-speaker latency is measured** from PortAudio's callback timestamps instead of
    estimated. The old estimate triple-counted the block and added a 0.5s magic constant, over-
    reporting 2050ms against a real ~1110ms — which is why manual sync trimming used to be
    unavoidable.
-   **The measurement is gated on the host API**, because whether those timestamps are real is a
    property of PortAudio's backend and cannot be told from the values. Verified against PortAudio's
    sources: ALSA is real; WASAPI reports momentary queue occupancy; DirectSound has its
    output-latency line commented out; MME never implemented the capture timestamp. So Windows
    always falls back to the estimate — previously WASAPI passed a naive `adc > 0` check with a
    value ~10x too small, making sync *worse* than MME.
-   Latency is published for the video side of
    [Realtime-AV-Swap](https://github.com/jiaheguo521/Realtime-AV-Swap) to hold frames against.

### Device handling

-   **Device selection now actually applies.** `query_hostapis()` has no `index` key, so the lookup
    raised `KeyError` and silently fell back to the default — the picker had never worked.
-   Devices are resolved **before** the buffers are sized; the buffers used to take their
    channels/samplerate from whatever device was selected previously.
-   The default device is taken from the host API rather than index 0, which was typically the
    graphics card's HDMI output.

### VAD

-   FSMN-VAD chunking **decoupled from `block_time`**: the model needs ≥60ms per call, while the
    slider goes to 40ms. Audio is accumulated and fed in whole chunks.
-   Periodic cache reset to bound funasr's state growth: it concatenates its score buffers on every
    chunk and only trims them once a segment end is confirmed, which never comes during sustained
    speech.

### Structure

-   **GUI split from the engine** (`realtime_vc_engine.py`), so the conversion logic is usable
    without PySimpleGUI.
-   Repository streamlined to the real-time path (training, WebUI and V2 inference removed).
-   Dropped `descript-audio-codec` (a protobuf conflict source that was never actually used), and
    renamed `modules/` → `audio_modules/` to avoid clashing with the video project.
-   Relative paths resolved against the script directory, not the CWD.
-   Legacy GUI: dragging any slider no longer stops the stream (the handler was a catch-all).

</details>

## Requirements

-   **OS**: Windows or Linux.
-   **GPU**: NVIDIA GPU with CUDA support is **strongly recommended** for real-time performance.
-   **Python**: 3.10

## Installation & Usage



1.  **Create and activate a virtual environment**:
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install dependencies**:

    *Note: The default `requirements.txt` includes CUDA 12.1 support. If you have a different CUDA version, please edit `requirements.txt` accordingly.*

    ```bash
    pip install -r requirements.txt
    pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

    ```

3.  **Run the GUI**:
```bash
python real-time-gui.py --checkpoint-path <path-to-checkpoint> --config-path <path-to-config>
```
- `checkpoint` is the path to the model checkpoint if you have trained or fine-tuned your own model, leave to blank to auto-download default model from huggingface. (`seed-uvit-tat-xlsr-tiny`)
- `config` is the path to the model config if you have trained or fine-tuned your own model, leave to blank to auto-download default config from huggingface  


## Configuration & Performance

It is strongly recommended to use a GPU for real-time voice conversion. Below are some benchmark results on an NVIDIA RTX 3060 Laptop GPU:

| Model Configuration | Diffusion Steps | Inference CFG Rate | Max Prompt Length | Block Time (s) | Latency (ms) | Inference Time (ms) |
|---------------------|-----------------|--------------------|-------------------|----------------|--------------|---------------------|
| seed-uvit-xlsr-tiny | 10              | 0.7                | 3.0               | 0.18s          | 430ms        | 150ms               |

### Key Parameters

-   **Diffusion Steps**: 4~10 recommended for fastest real-time inference.
-   **Block Time**: the model hop, and the dominant latency term (`~2 x block_time`). Inference must
    keep up with it *on average* — a single block overrunning is absorbed rather than dropped, so
    there is roughly `block_time` of headroom for spikes. Watch `Infer` and leave some margin.
-   **Extra context**: Increasing this improves stability but adds latency.
-   **Virtual Cable**: Use [VB-CABLE](https://vb-audio.com/Cable/) to route GUI output to a virtual microphone for use in other applications (Discord, Zoom, etc.).

## Acknowledgements 🙏

This project is a separated version of [Seed-VC](https://github.com/Plachtaa/Seed-VC). All credit goes to the original authors.

-   [Seed-VC](https://github.com/Plachtaa/Seed-VC) - Original Project
- [Amphion](https://github.com/open-mmlab/Amphion) for providing computational resources and inspiration!
- [Vevo](https://github.com/open-mmlab/Amphion/tree/main/models/vc/vevo) for theoretical foundation of V2 model
- [MegaTTS3](https://github.com/bytedance/MegaTTS3) for multi-condition CFG inference implemented in V2 model
- [ASTRAL-quantiztion](https://github.com/Plachtaa/ASTRAL-quantization) for the amazing speaker-disentangled speech tokenizer used by V2 model
- [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) for foundationing the real-time voice conversion
- [SEED-TTS](https://arxiv.org/abs/2406.02430) for the initial idea
