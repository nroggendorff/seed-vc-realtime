import math
import os
import random
import shutil
from fractions import Fraction

import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import yaml
from tqdm import tqdm

DATA_DIR = "./dataset"
RUN_NAME = "my_voice"  # output goes to ./runs/<RUN_NAME>/
CONFIG_PATH = "./configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml"
PRETRAINED_CKPT = None  # None = the GUI's default model; or a previous ft_model.pth from which to continue
MAX_STEPS = 1000
EFFECTIVE_BATCH_SIZE = 32
BATCH_SIZE = 5
LEARNING_RATE = 1e-5
WARMUP_STEPS = 50
MIN_LR_RATIO = 0.1  # cosine decay ends at LEARNING_RATE * MIN_LR_RATIO
SAVE_EVERY = 100
MIN_SECONDS = 1.0  # shorter clips are skipped
MAX_SECONDS = 30.0  # longer files are split into equal pieces no longer than this
MAX_VRAM_GB = 10.0
# Condition on each clip's RMVPE F0 so the input's pitch drives the output; content features
# carry almost none. The F0 embedding starts at zero and needs a higher LR to move in time.
F0_CONDITION = True
F0_LEARNING_RATE = 1e-3
F0_THRESHOLD = 0.03  # RMVPE voicing threshold; keep it equal to the engine's
# Share of clips whose prompt is pitch-shifted (formants kept) by up to F0_SHIFT_MAX semitones,
# so F0 is the only reliable pitch cue. Off: it didn't help the real-time engine and hurt seams.
F0_SHIFT_SHARE = 0.0
F0_SHIFT_MAX = 4
F0_SHIFT_ENVELOPE = 20  # cepstral coefficients in the envelope; more eats harmonics
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus")


def pitch_shift(wave, k, n_fft=1024):
    """Pitch-shift by about k semitones, keeping the length; returns (wave, exact factor).

    The factor is a small-denominator fraction (63/50 for +4) so the resampling kernel stays
    small; torchaudio's PitchShift resamples by e.g. 27781/22050, which takes gigabytes.
    """
    factor = Fraction(2 ** (k / 12)).limit_denominator(64)
    hop = n_fft // 4
    window = torch.hann_window(n_fft, device=wave.device)
    spec = torch.stft(wave, n_fft, hop, window=window, return_complex=True)
    phase_advance = torch.linspace(0, math.pi * hop, spec.size(-2), device=wave.device)[
        :, None
    ]
    spec = torchaudio.functional.phase_vocoder(spec, 1 / float(factor), phase_advance)
    stretched = torch.istft(
        spec, n_fft, hop, window=window, length=round(len(wave) * float(factor))
    )
    shifted = torchaudio.functional.resample(
        stretched, factor.numerator, factor.denominator
    )
    shifted = torch.nn.functional.pad(
        shifted[: len(wave)], (0, max(0, len(wave) - len(shifted)))
    )
    return shifted, float(factor)


def load_clips(sr):
    clips = []
    for root, _, files in os.walk(DATA_DIR):
        for name in sorted(files):
            if not name.lower().endswith(AUDIO_EXTS):
                continue
            path = os.path.join(root, name)
            try:
                wave, _ = librosa.load(path, sr=sr)
            except Exception as e:
                print(f"Skipping {path}: {e}")
                continue
            wave = wave / max(1.0, np.abs(wave).max())  # resampling can overshoot +-1
            n_pieces = int(np.ceil(len(wave) / (MAX_SECONDS * sr)))
            for piece in np.array_split(wave, max(n_pieces, 1)):
                if len(piece) >= MIN_SECONDS * sr:
                    clips.append(torch.from_numpy(piece))
    if not clips:
        raise SystemExit(
            f"No usable audio (>= {MIN_SECONDS}s) found in {os.path.abspath(DATA_DIR)}"
        )
    print(
        f"Loaded {len(clips)} clips, {sum(len(c) for c in clips) / sr / 60:.1f} minutes total"
    )
    return clips


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if DEVICE == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, MAX_VRAM_GB * 1024**3 / total)
        )

    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    from audio_modules.audio import mel_spectrogram
    from audio_modules.campplus.DTDNN import CAMPPlus
    from audio_modules.commons import build_model, load_checkpoint, recursive_munch
    from audio_modules.openvoice.api import ToneColorConverter
    from hf_utils import load_custom_model_from_hf

    config = yaml.safe_load(open(CONFIG_PATH))
    if F0_CONDITION:
        config["model_params"]["length_regulator"]["f0_condition"] = True
    params = recursive_munch(config["model_params"])
    use_f0 = bool(params.length_regulator.get("f0_condition", False))
    params.dit_type = "DiT"
    if params.speech_tokenizer.type != "xlsr":
        raise SystemExit("Only the xlsr (real-time) configs are supported.")
    sr = config["preprocess_params"]["sr"]
    spect = config["preprocess_params"]["spect_params"]
    mel_args = dict(
        n_fft=spect["n_fft"],
        win_size=spect["win_length"],
        hop_size=spect["hop_length"],
        num_mels=spect["n_mels"],
        sampling_rate=sr,
        fmin=spect.get("fmin", 0),
        fmax=None if spect.get("fmax", "None") == "None" else spect["fmax"],
        center=False,
    )

    clips = load_clips(sr)

    run_dir = os.path.join("runs", RUN_NAME)
    os.makedirs(run_dir, exist_ok=True)
    if F0_CONDITION:
        # The engine reads f0_condition from this copy, so it must carry the override.
        with open(os.path.join(run_dir, "config.yml"), "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
    else:
        shutil.copyfile(CONFIG_PATH, os.path.join(run_dir, "config.yml"))
    out_path = os.path.join(run_dir, "ft_model.pth")

    clip_f0 = None
    if use_f0:
        from audio_modules.rmvpe import RMVPE

        rmvpe = RMVPE(
            load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt"),
            is_half=False,
            device=DEVICE,
        )
        # Precomputed, since RMVPE is slow. 100 frames/s, 0 where unvoiced.
        clip_f0 = [
            torch.from_numpy(
                rmvpe.infer_from_audio(
                    torchaudio.functional.resample(c, sr, 16000), thred=F0_THRESHOLD
                )
            ).float()
            for c in tqdm(clips, desc="F0")
        ]
        del rmvpe
        torch.cuda.empty_cache()

    model = build_model(params, stage="DiT")
    ckpt = PRETRAINED_CKPT or load_custom_model_from_hf(
        "Plachta/Seed-VC", config["pretrained_model"]
    )
    model, _, _, _ = load_checkpoint(
        model,
        None,
        ckpt,
        load_only_params=True,
        ignore_modules=[],
        is_distributed=False,
    )
    for key in model:
        model[key].to(DEVICE).train()
    model.cfm.estimator.setup_caches(max_batch_size=BATCH_SIZE, max_seq_length=8192)
    lr_named = list(model.length_regulator.named_parameters())
    param_groups = [
        {
            "params": list(model.cfm.parameters())
            + [p for n, p in lr_named if not n.startswith("f0_")]
        }
    ]
    f0_params = [p for n, p in lr_named if n.startswith("f0_")]
    if f0_params:
        param_groups.append({"params": f0_params, "lr": F0_LEARNING_RATE})
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=LEARNING_RATE,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.01,
    )

    def lr_factor(step):
        if step < WARMUP_STEPS:
            return (step + 1) / WARMUP_STEPS
        progress = min(1.0, (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return MIN_LR_RATIO + (1 - MIN_LR_RATIO) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    campplus = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus.load_state_dict(
        torch.load(
            load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin"),
            map_location="cpu",
        )
    )
    campplus.eval().to(DEVICE)

    # Content comes from a re-voiced copy, so timbre must come from the prompt and style.
    ov_ckpt, ov_config = load_custom_model_from_hf(
        "myshell-ai/OpenVoiceV2", "converter/checkpoint.pth", "converter/config.json"
    )
    converter = ToneColorConverter(ov_config, device=DEVICE)
    converter.load_ckpt(ov_ckpt)
    se_db = torch.load(
        load_custom_model_from_hf("Plachta/Seed-VC", "se_db.pt"), map_location="cpu"
    )

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(params.speech_tokenizer.name)
    # Eager: fp16 mem-efficient SDPA intermittently hits an illegal memory access (torch 2.11).
    wav2vec = Wav2Vec2Model.from_pretrained(
        params.speech_tokenizer.name, attn_implementation="eager"
    )
    wav2vec.encoder.layers = wav2vec.encoder.layers[
        : params.speech_tokenizer.output_layer
    ]
    wav2vec = wav2vec.half().eval().to(DEVICE)

    @torch.no_grad()
    def content_features(waves_16k):
        inputs = extractor(
            [w.cpu().numpy() for w in waves_16k],
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
            sampling_rate=16000,
        )
        return wav2vec(inputs.input_values.to(DEVICE).half()).last_hidden_state.float()

    def save():
        torch.save({"net": {key: model[key].state_dict() for key in model}}, out_path)

    shift_steps = [k for k in range(-F0_SHIFT_MAX, F0_SHIFT_MAX + 1) if k]
    n_mels = mel_args["num_mels"]
    dct = torch.cos(
        math.pi
        / n_mels
        * (torch.arange(n_mels)[None] + 0.5)
        * torch.arange(n_mels)[:, None]
    ) * math.sqrt(2 / n_mels)
    dct[0] /= math.sqrt(2)  # orthonormal DCT-II over mel bins
    dct = dct.to(DEVICE)
    lifter = (torch.arange(n_mels, device=DEVICE) < F0_SHIFT_ENVELOPE).float()[:, None]

    def envelope(log_mel):
        return dct.T @ (lifter * (dct @ log_mel))

    @torch.no_grad()
    def shift_prompt(wave, mel, k):
        """Shifted wave, its log-mel with the original envelope, and the exact F0 factor."""
        shifted, factor = pitch_shift(wave, k)
        shifted_mel = mel_spectrogram(shifted[None], **mel_args)[0][:, : mel.size(1)]
        return shifted, shifted_mel - envelope(shifted_mel) + envelope(mel), factor

    def compute_loss(n):
        picks = random.choices(range(len(clips)), k=n)
        waves = [clips[i].to(DEVICE) for i in picks]
        mels = [mel_spectrogram(w[None], **mel_args)[0] for w in waves]
        # Both conditions get the clip's own F0, as the engine feeds the input's F0.
        f0 = (
            torch.nn.utils.rnn.pad_sequence(
                [clip_f0[i] for i in picks], batch_first=True
            ).to(DEVICE)
            if use_f0
            else None
        )
        # The prompt's audio, mel and F0: as recorded, or pitch-shifted (see F0_SHIFT_SHARE).
        prompt_waves, prompt_mels, prompt_f0 = list(waves), list(mels), f0
        if use_f0 and F0_SHIFT_SHARE > 0:
            prompt_f0 = f0.clone()
            for i in range(n):
                if random.random() < F0_SHIFT_SHARE:
                    prompt_waves[i], prompt_mels[i], factor = shift_prompt(
                        waves[i], mels[i], random.choice(shift_steps)
                    )
                    prompt_f0[i] *= factor
        wave_lengths = torch.tensor([len(w) for w in waves], device=DEVICE)
        mel_lengths = torch.tensor([m.size(1) for m in mels], device=DEVICE)
        waves = torch.nn.utils.rnn.pad_sequence(waves, batch_first=True)
        prompt_waves = torch.nn.utils.rnn.pad_sequence(prompt_waves, batch_first=True)
        mels, prompt_mels = (
            torch.nn.utils.rnn.pad_sequence(
                [m.T for m in ms], batch_first=True, padding_value=-10
            ).transpose(1, 2)
            for ms in (mels, prompt_mels)
        )

        with torch.no_grad():
            waves_22k = torchaudio.functional.resample(waves, sr, 22050)
            lengths_22k = (wave_lengths.float() * 22050 / sr).long()
            src_se = converter.extract_se(waves_22k, lengths_22k)
            tgt_se = se_db[torch.randint(0, len(se_db), (n,))].to(DEVICE)
            converted_22k = converter.convert(
                waves_22k, lengths_22k, src_se, tgt_se
            ).squeeze(1)

            waves_16k = torchaudio.functional.resample(waves, sr, 16000)
            lengths_16k = (wave_lengths.float() * 16000 / sr).long()
            # Prompt content comes from the prompt's own audio, as in the engine.
            S_ori = content_features(
                torchaudio.functional.resample(prompt_waves, sr, 16000)
            )
            S_alt = content_features(
                torchaudio.functional.resample(converted_22k, 22050, 16000)
            )

            style = []
            for w, length in zip(waves_16k, lengths_16k):
                feat = kaldi.fbank(
                    w[None, :length], num_mel_bins=80, dither=0, sample_frequency=16000
                )
                style.append(campplus((feat - feat.mean(dim=0, keepdim=True))[None]))
            style = torch.cat(style)

        ori_cond = model.length_regulator(S_ori, ylens=mel_lengths, f0=prompt_f0)[0]
        alt_cond = model.length_regulator(S_alt, ylens=mel_lengths, f0=f0)[0]

        prompt_len = (torch.rand(n, device=DEVICE) * (mel_lengths - 1)).floor().long()
        prompt_len[torch.rand(n, device=DEVICE) < 0.1] = 0
        cond = alt_cond.clone()
        for i in range(n):
            cond[i, : prompt_len[i]] = ori_cond[i, : prompt_len[i]]
            mels[i, :, : prompt_len[i]] = prompt_mels[i, :, : prompt_len[i]]

        loss, _ = model.cfm(mels, mel_lengths, prompt_len, cond, style)
        return loss

    # The loss is a per-clip mean, so weighting each micro-batch by its share of the clips gives
    # the gradient of one EFFECTIVE_BATCH_SIZE batch.
    micro_batches = [BATCH_SIZE] * (EFFECTIVE_BATCH_SIZE // BATCH_SIZE)
    if EFFECTIVE_BATCH_SIZE % BATCH_SIZE:
        micro_batches.append(EFFECTIVE_BATCH_SIZE % BATCH_SIZE)

    smooth_loss = None
    bar = tqdm(range(1, MAX_STEPS + 1))
    for step in bar:
        optimizer.zero_grad()
        step_loss = 0.0
        for n in micro_batches:
            loss = compute_loss(n)
            (loss * n / EFFECTIVE_BATCH_SIZE).backward()
            step_loss += loss.item() * n / EFFECTIVE_BATCH_SIZE
        torch.nn.utils.clip_grad_norm_(model.cfm.parameters(), 10.0)
        torch.nn.utils.clip_grad_norm_(model.length_regulator.parameters(), 10.0)
        optimizer.step()
        scheduler.step()

        smooth_loss = (
            step_loss if smooth_loss is None else 0.99 * smooth_loss + 0.01 * step_loss
        )
        bar.set_postfix(
            loss=f"{smooth_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}"
        )
        if step % SAVE_EVERY == 0 or step == MAX_STEPS:
            save()

    print(
        f"\nDone. Saved {os.path.abspath(out_path)}\nRun it with:\n"
        f"  python real-time-gui.py --checkpoint-path {out_path} --config-path {os.path.join(run_dir, 'config.yml')}"
    )


if __name__ == "__main__":
    main()
