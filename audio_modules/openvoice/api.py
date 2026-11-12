import torch
import numpy as np
from . import utils

from .mel_processing import spectrogram_torch
from .models import SynthesizerTrn


class OpenVoiceBaseClass(object):
    def __init__(self, config_path, device="cuda:0"):
        if "cuda" in device:
            assert torch.cuda.is_available()

        hps = utils.get_hparams_from_file(config_path)

        model = SynthesizerTrn(
            len(getattr(hps, "symbols", [])),
            hps.data.filter_length // 2 + 1,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        ).to(device)

        model.eval()
        self.model = model
        self.hps = hps
        self.device = device

    def load_ckpt(self, ckpt_path):
        checkpoint_dict = torch.load(ckpt_path, map_location=torch.device(self.device))
        a, b = self.model.load_state_dict(checkpoint_dict["model"], strict=False)
        print("Loaded checkpoint '{}'".format(ckpt_path))
        print("missing/unexpected keys:", a, b)


class ToneColorConverter(OpenVoiceBaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.version = getattr(self.hps, "_version_", "v1")

    def extract_se(self, waves, wave_lengths):

        device = self.device
        hps = self.hps
        gs = []

        for wav_tensor, wav_len in zip(waves, wave_lengths):
            y = wav_tensor[:wav_len][None, :]
            y = spectrogram_torch(
                y,
                hps.data.filter_length,
                hps.data.hop_length,
                hps.data.win_length,
                center=False,
            ).to(device)
            with torch.no_grad():
                gs.append(self.model.ref_enc(y.transpose(1, 2)).unsqueeze(-1).detach())
        return torch.stack(gs).squeeze(1).squeeze(-1)

    def convert(self, src_waves, src_wave_lengths, src_se, tgt_se, tau=0.3):
        hps = self.hps
        with torch.no_grad():
            y = src_waves
            spec = spectrogram_torch(
                y,
                hps.data.filter_length,
                hps.data.hop_length,
                hps.data.win_length,
                center=False,
            ).to(self.device)
            spec_lengths = src_wave_lengths // hps.data.hop_length
            spec_lengths = spec_lengths.clamp(min=1, max=spec.size(2))
            audio = self.model.voice_conversion(
                spec,
                spec_lengths,
                sid_src=src_se.unsqueeze(-1),
                sid_tgt=tgt_se.unsqueeze(-1),
                tau=tau,
            )[0]
        return audio

    def add_watermark(self, audio, message):
        return audio
        device = self.device
        bits = utils.string_to_bits(message).reshape(-1)
        n_repeat = len(bits) // 32

        K = 16000
        coeff = 2
        for n in range(n_repeat):
            trunck = audio[(coeff * n) * K : (coeff * n + 1) * K]
            if len(trunck) != K:
                print("Audio too short, fail to add watermark")
                break
            message_npy = bits[n * 32 : (n + 1) * 32]

            with torch.no_grad():
                signal = torch.FloatTensor(trunck).to(device)[None]
                message_tensor = torch.FloatTensor(message_npy).to(device)[None]
                signal_wmd_tensor = self.watermark_model.encode(signal, message_tensor)
                signal_wmd_npy = signal_wmd_tensor.detach().cpu().squeeze()
            audio[(coeff * n) * K : (coeff * n + 1) * K] = signal_wmd_npy
        return audio

    def detect_watermark(self, audio, n_repeat):
        bits = []
        K = 16000
        coeff = 2
        for n in range(n_repeat):
            trunck = audio[(coeff * n) * K : (coeff * n + 1) * K]
            if len(trunck) != K:
                print("Audio too short, fail to detect watermark")
                return "Fail"
            with torch.no_grad():
                signal = torch.FloatTensor(trunck).to(self.device).unsqueeze(0)
                message_decoded_npy = (
                    (self.watermark_model.decode(signal) >= 0.5)
                    .int()
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze()
                )
            bits.append(message_decoded_npy)
        return utils.bits_to_string(np.stack(bits).reshape(-1, 8))
