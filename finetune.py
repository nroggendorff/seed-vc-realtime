import math
import os
import random
import shutil

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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus")


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

    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    from audio_modules.audio import mel_spectrogram
    from audio_modules.campplus.DTDNN import CAMPPlus
    from audio_modules.commons import build_model, load_checkpoint, recursive_munch
    from audio_modules.openvoice.api import ToneColorConverter
    from hf_utils import load_custom_model_from_hf

    config = yaml.safe_load(open(CONFIG_PATH))
    params = recursive_munch(config["model_params"])
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
    shutil.copyfile(CONFIG_PATH, os.path.join(run_dir, "config.yml"))
    out_path = os.path.join(run_dir, "ft_model.pth")

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
    optimizer = torch.optim.AdamW(
        list(model.cfm.parameters()) + list(model.length_regulator.parameters()),
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

    # OpenVoice re-voices each clip as a random other speaker before content features are taken, so the model has to get the target timbre from the prompt/style.
    ov_ckpt, ov_config = load_custom_model_from_hf(
        "myshell-ai/OpenVoiceV2", "converter/checkpoint.pth", "converter/config.json"
    )
    converter = ToneColorConverter(ov_config, device=DEVICE)
    converter.load_ckpt(ov_ckpt)
    se_db = torch.load(
        load_custom_model_from_hf("Plachta/Seed-VC", "se_db.pt"), map_location="cpu"
    )

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(params.speech_tokenizer.name)
    wav2vec = Wav2Vec2Model.from_pretrained(params.speech_tokenizer.name)
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

    def compute_loss(n):
        waves = [c.to(DEVICE) for c in random.choices(clips, k=n)]
        mels = [mel_spectrogram(w[None], **mel_args)[0] for w in waves]
        wave_lengths = torch.tensor([len(w) for w in waves], device=DEVICE)
        mel_lengths = torch.tensor([m.size(1) for m in mels], device=DEVICE)
        waves = torch.nn.utils.rnn.pad_sequence(waves, batch_first=True)
        mels = torch.nn.utils.rnn.pad_sequence(
            [m.T for m in mels], batch_first=True, padding_value=-10
        ).transpose(1, 2)

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
            S_ori = content_features(waves_16k)
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

        ori_cond = model.length_regulator(S_ori, ylens=mel_lengths)[0]
        alt_cond = model.length_regulator(S_alt, ylens=mel_lengths)[0]

        prompt_len = (torch.rand(n, device=DEVICE) * (mel_lengths - 1)).floor().long()
        prompt_len[torch.rand(n, device=DEVICE) < 0.1] = 0
        cond = alt_cond.clone()
        for i in range(n):
            cond[i, : prompt_len[i]] = ori_cond[i, : prompt_len[i]]

        loss, _ = model.cfm(mels, mel_lengths, prompt_len, cond, style)
        return loss

    # Full forward passes of BATCH_SIZE plus one smaller pass for any remainder. The loss is a per-clip mean, so weighting each pass by its share of the clips gives the same gradient as a single EFFECTIVE_BATCH_SIZE batch.
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
