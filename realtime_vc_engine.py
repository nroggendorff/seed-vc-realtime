import os
import sys
import shutil
import time
import json
import warnings
import yaml
import re
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torchaudio.transforms as tat
import sounddevice as sd
import librosa
from audio_modules.commons import str2bool, recursive_munch, build_model, load_checkpoint

# Add current directory to path for imports
now_dir = os.getcwd()
sys.path.append(now_dir)

# Script directory for relative path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from hf_utils import load_custom_model_from_hf
from funasr import AutoModel

# Suppress warnings
warnings.simplefilter("ignore")

# Host APIs whose callback timestamps are real hardware timestamps, so that
# outputBufferDacTime - inputBufferAdcTime is the true mic->speaker latency.
# This is a per-host-API property of PortAudio's implementation, not something we
# can detect at runtime - a bad backend still returns plausible-looking numbers.
# Verified against the PortAudio sources (see docs/DELAY_LOGIC.md):
#   ALSA        - driver timestamp (snd_pcm_status) + snd_pcm_delay  -> real. TRUSTED.
#   WASAPI      - PaUtil_GetTime() +/- IAudioClient::GetCurrentPadding(): momentary
#                 queue occupancy, not latency. Reads ~10x low and jitters.
#   DirectSound - outputBufferDacTime = currentTime; the "+ outputLatency" line is
#                 commented out in the source, so the whole output side is missing.
#   MME         - "@todo implement inputBufferAdcTime": adc is always 0.
#   WDM-KS      - timestamps not implemented (PortAudio issue #531).
# So on Windows there is no usable measurement: every host API falls back below.
TIMESTAMP_CAPABLE_HOSTAPIS = frozenset({"ALSA"})

# Even on a trusted host API, an xrun can make snd_pcm_delay return nonsense.
# Any capture->playback span beyond this is not a real latency (see _measure_latency).
MAX_PLAUSIBLE_LATENCY_S = 5.0

# FSMN-VAD needs at least this much audio per call. Below it the LFR window yields
# zero frames and the model raises "mat1 and mat2 shapes cannot be multiplied
# (1x0 and 400x140)". Measured: 40ms fails, 60ms works.
MIN_VAD_CHUNK_MS = 60

class RealtimeVCConfig:
    def __init__(self):
        self.reference_audio_path = ""
        self.diffusion_steps = 10
        self.sr_type = "sr_model"
        self.block_time = 0.25  # s
        self.threhold = -60
        self.crossfade_time = 0.05
        self.extra_time_ce = 2.5
        self.extra_time = 0.5
        self.extra_time_right = 2.0
        self.I_noise_reduce = False
        self.O_noise_reduce = False
        self.inference_cfg_rate = 0.7
        self.max_prompt_length = 3.0
        self.sg_hostapi = ""
        self.wasapi_exclusive = False
        self.sg_input_device = ""
        self.sg_output_device = ""
        self.samplerate = 44100  # default, will be updated
        self.channels = 2
        self.sr_model = True
        self.sr_device = False
        self.function = "vc"  # "vc" for voice conversion, "im" for input monitoring (passthrough)

class RealtimeVCEngine:
    def __init__(self, args):
        self.args = args
        self.config = RealtimeVCConfig()
        self.device = self._get_device(args.gpu)
        self.fp16 = args.fp16
        
        # Runtime state
        self.flag_vc = False
        self.stream = None
        self.delay_time = 0.0
        self.measured_delay_time = 0.0  # end-to-end latency (s) that drives A/V sync
        self._hw_latency = 0.0          # input+output hardware latency, for the fallback estimate
        self._hostapi_name = ""         # host API actually in use; gates the measurement
        self._measured_logged = False
        self.last_error_print_time = 0
        
        # Audio processing state
        self.prompt_condition = None
        self.mel2 = None
        self.style2 = None
        self.reference_wav_name = ""
        self.prompt_len = 3
        self.ce_dit_difference = 2.0

        # Geometry params the current buffers were built from; set by _prepare_buffers.
        self.active_extra_time_ce = self.config.extra_time_ce
        self.active_extra_time = self.config.extra_time
        self.active_extra_time_right = self.config.extra_time_right
        
        # Load models
        print(f"Loading models on {self.device}...")
        self.model_set = self._load_models(args)
        self.vad_model = AutoModel(model="fsmn-vad", model_revision="v2.0.4", disable_pbar=True, log_level="ERROR", disable_update=True)
        
        # Devices
        self.hostapis = []
        self.input_devices = []
        self.output_devices = []
        self.input_devices_indices = []
        self.output_devices_indices = []
        self.update_devices()
        
        # Callbacks
        # Reporting hook only: the engine never applies A/V sync policy itself, it
        # just reports its own timings. The host app (launcher) owns all sync.
        # Signature: on_perf_update(infer_time, delay_time, measured_delay_time)
        self.on_perf_update = None

    def _get_device(self, gpu_id):
        cuda_target = f"cuda:{gpu_id}" if gpu_id else "cuda"
        if torch.cuda.is_available():
            return torch.device(cuda_target)
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _load_models(self, args):
        print(f"Using fp16: {self.fp16}")
        if args.checkpoint_path is None or args.checkpoint_path == "":
            dit_checkpoint_path, dit_config_path = load_custom_model_from_hf("Plachta/Seed-VC",
                                                                             "DiT_uvit_tat_xlsr_ema.pth",
                                                                             "config_dit_mel_seed_uvit_xlsr_tiny.yml")
        else:
            dit_checkpoint_path = args.checkpoint_path
            dit_config_path = args.config_path
            
        config = yaml.safe_load(open(dit_config_path, "r"))
        model_params = recursive_munch(config["model_params"])
        model_params.dit_type = 'DiT'
        model = build_model(model_params, stage="DiT")
        hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
        sr = config["preprocess_params"]["sr"]

        # Load checkpoints
        model, _, _, _ = load_checkpoint(
            model,
            None,
            dit_checkpoint_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        for key in model:
            model[key].eval()
            model[key].to(self.device)
        model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

        # Load additional modules
        from audio_modules.campplus.DTDNN import CAMPPlus

        campplus_ckpt_path = load_custom_model_from_hf(
            "funasr/campplus", "campplus_cn_common.bin", config_filename=None
        )
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        campplus_model.eval()
        campplus_model.to(self.device)

        vocoder_type = model_params.vocoder.type

        if vocoder_type == 'bigvgan':
            from audio_modules.bigvgan import bigvgan
            bigvgan_name = model_params.vocoder.name
            bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
            bigvgan_model.remove_weight_norm()
            bigvgan_model = bigvgan_model.eval().to(self.device)
            vocoder_fn = bigvgan_model
        elif vocoder_type == 'hifigan':
            from audio_modules.hifigan.generator import HiFTGenerator
            from audio_modules.hifigan.f0_predictor import ConvRNNF0Predictor
            hifigan_config_path = os.path.join(SCRIPT_DIR, 'configs/hifigan.yml')
            hift_config = yaml.safe_load(open(hifigan_config_path, 'r'))
            hift_gen = HiFTGenerator(**hift_config['hift'], f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor']))
            hift_path = load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", 'hift.pt', None)
            hift_gen.load_state_dict(torch.load(hift_path, map_location='cpu'))
            hift_gen.eval()
            hift_gen.to(self.device)
            vocoder_fn = hift_gen
        elif vocoder_type == "vocos":
            vocos_config = yaml.safe_load(open(model_params.vocoder.vocos.config, 'r'))
            vocos_path = model_params.vocoder.vocos.path
            vocos_model_params = recursive_munch(vocos_config['model_params'])
            vocos = build_model(vocos_model_params, stage='mel_vocos')
            vocos_checkpoint_path = vocos_path
            vocos, _, _, _ = load_checkpoint(vocos, None, vocos_checkpoint_path,
                                             load_only_params=True, ignore_modules=[], is_distributed=False)
            _ = [vocos[key].eval().to(self.device) for key in vocos]
            _ = [vocos[key].to(self.device) for key in vocos]
            vocoder_fn = vocos.decoder
        else:
            raise ValueError(f"Unknown vocoder type: {vocoder_type}")

        speech_tokenizer_type = model_params.speech_tokenizer.type
        
        # Define semantic_fn based on type
        if speech_tokenizer_type == 'whisper':
            from transformers import AutoFeatureExtractor, WhisperModel
            whisper_name = model_params.speech_tokenizer.name
            whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(self.device)
            del whisper_model.decoder
            whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

            def semantic_fn(waves_16k):
                ori_inputs = whisper_feature_extractor([waves_16k.squeeze(0).cpu().numpy()],
                                                       return_tensors="pt",
                                                       return_attention_mask=True)
                ori_input_features = whisper_model._mask_input_features(
                    ori_inputs.input_features, attention_mask=ori_inputs.attention_mask).to(self.device)
                with torch.no_grad():
                    ori_outputs = whisper_model.encoder(
                        ori_input_features.to(whisper_model.encoder.dtype),
                        head_mask=None,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                S_ori = ori_outputs.last_hidden_state.to(torch.float32)
                S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
                return S_ori
        
        elif speech_tokenizer_type == 'cnhubert':
            from transformers import Wav2Vec2FeatureExtractor, HubertModel
            hubert_model_name = config['model_params']['speech_tokenizer']['name']
            hubert_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model_name)
            hubert_model = HubertModel.from_pretrained(hubert_model_name)
            hubert_model = hubert_model.to(self.device)
            hubert_model = hubert_model.eval()
            hubert_model = hubert_model.half()

            def semantic_fn(waves_16k):
                ori_waves_16k_input_list = [waves_16k[bib].cpu().numpy() for bib in range(len(waves_16k))]
                ori_inputs = hubert_feature_extractor(ori_waves_16k_input_list,
                                                      return_tensors="pt",
                                                      return_attention_mask=True,
                                                      padding=True,
                                                      sampling_rate=16000).to(self.device)
                with torch.no_grad():
                    ori_outputs = hubert_model(ori_inputs.input_values.half())
                S_ori = ori_outputs.last_hidden_state.float()
                return S_ori

        elif speech_tokenizer_type == 'xlsr':
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
            model_name = config['model_params']['speech_tokenizer']['name']
            output_layer = config['model_params']['speech_tokenizer']['output_layer']
            wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
            wav2vec_model.encoder.layers = wav2vec_model.encoder.layers[:output_layer]
            wav2vec_model = wav2vec_model.to(self.device)
            wav2vec_model = wav2vec_model.eval()
            wav2vec_model = wav2vec_model.half()

            def semantic_fn(waves_16k):
                ori_waves_16k_input_list = [waves_16k[bib].cpu().numpy() for bib in range(len(waves_16k))]
                ori_inputs = wav2vec_feature_extractor(ori_waves_16k_input_list,
                                                       return_tensors="pt",
                                                       return_attention_mask=True,
                                                       padding=True,
                                                       sampling_rate=16000).to(self.device)
                with torch.no_grad():
                    ori_outputs = wav2vec_model(ori_inputs.input_values.half())
                S_ori = ori_outputs.last_hidden_state.float()
                return S_ori
        else:
             raise ValueError(f"Unknown speech tokenizer type: {speech_tokenizer_type}")

        # Generate mel spectrograms
        mel_fn_args = {
            "n_fft": config['preprocess_params']['spect_params']['n_fft'],
            "win_size": config['preprocess_params']['spect_params']['win_length'],
            "hop_size": config['preprocess_params']['spect_params']['hop_length'],
            "num_mels": config['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": sr,
            "fmin": config['preprocess_params']['spect_params'].get('fmin', 0),
            "fmax": None if config['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
            "center": False
        }
        from audio_modules.audio import mel_spectrogram
        to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

        return (
            model,
            semantic_fn,
            vocoder_fn,
            campplus_model,
            to_mel,
            mel_fn_args,
        )

    def update_devices(self, hostapi_name=None):
        self.stop_stream()
        sd._terminate()
        sd._initialize()
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        for hostapi in hostapis:
            for device_idx in hostapi["devices"]:
                devices[device_idx]["hostapi_name"] = hostapi["name"]
        self.hostapis = [hostapi["name"] for hostapi in hostapis]
        if hostapi_name not in self.hostapis:
            hostapi_name = self.hostapis[0]
        self.input_devices = [
            d["name"]
            for d in devices
            if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]
        self.output_devices = [
            d["name"]
            for d in devices
            if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]
        self.input_devices_indices = [
            d["index"] if "index" in d else d["name"]
            for d in devices
            if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]
        self.output_devices_indices = [
            d["index"] if "index" in d else d["name"]
            for d in devices
            if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]

        # Which device this host API considers the default. Callers should prefer
        # these over input_devices[0]/output_devices[0]: ALSA enumerates raw hw:
        # devices first, so the first output is typically the GPU's HDMI/DisplayPort
        # audio (the monitor), while the device the user actually configured
        # ('default'/'pipewire') sorts last.
        self.default_input_device = ""
        self.default_output_device = ""
        for hostapi in hostapis:
            if hostapi["name"] != hostapi_name:
                continue
            for kind, pool in (("input", self.input_devices), ("output", self.output_devices)):
                idx = hostapi.get(f"default_{kind}_device", -1)
                if idx is None or idx < 0:
                    continue
                name = devices[idx]["name"]
                if name in pool:
                    setattr(self, f"default_{kind}_device", name)
            break

    def preferred_device(self, kind):
        """Best default choice for 'input'/'output': the host API default, else the
        first enumerated device, else ''."""
        pool = self.input_devices if kind == "input" else self.output_devices
        default = self.default_input_device if kind == "input" else self.default_output_device
        if default:
            return default
        return pool[0] if pool else ""

    def get_device_samplerate(self):
        return int(sd.query_devices(device=sd.default.device[0])["default_samplerate"])

    def get_device_channels(self):
        max_input_channels = sd.query_devices(device=sd.default.device[0])["max_input_channels"]
        max_output_channels = sd.query_devices(device=sd.default.device[1])["max_output_channels"]
        return min(max_input_channels, max_output_channels, 2)

    def set_config(self, **kwargs):
        """Update configuration parameters."""
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        
        # Handle derived config logic here if necessary
        if "sr_type" in kwargs:
             self.config.sr_model = (self.config.sr_type == "sr_model")
             self.config.sr_device = (self.config.sr_type == "sr_device")

    def _prepare_buffers(self):
        # Initialize variables based on current config
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()

        self.reference_wav, _ = librosa.load(
            self.config.reference_audio_path, sr=self.model_set[-1]["sampling_rate"]
        )
        
        self.config.samplerate = (
            self.model_set[-1]["sampling_rate"]
            if self.config.sr_type == "sr_model"
            else self.get_device_samplerate()
        )
        self.config.channels = self.get_device_channels()
        self.zc = self.config.samplerate // 50
        self.block_frame = int(np.round(self.config.block_time * self.config.samplerate / self.zc)) * self.zc
        self.block_frame_16k = 320 * self.block_frame // self.zc
        self.crossfade_frame = int(np.round(self.config.crossfade_time * self.config.samplerate / self.zc)) * self.zc
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = int(np.round(self.config.extra_time_ce * self.config.samplerate / self.zc)) * self.zc
        self.extra_frame_right = int(np.round(self.config.extra_time_right * self.config.samplerate / self.zc)) * self.zc

        # Snapshot the params the buffers above were sized from. The streaming path
        # must keep using these even if the user drags a slider mid-stream, or the
        # config would disagree with the buffer geometry and the slicing would go
        # out of step. Slider changes take effect on the next start_stream().
        self.active_extra_time_ce = self.config.extra_time_ce
        self.active_extra_time = self.config.extra_time
        self.active_extra_time_right = self.config.extra_time_right


        self.input_wav = torch.zeros(
            self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame + self.extra_frame_right,
            device=self.device, dtype=torch.float32
        )
        self.input_wav_denoise = self.input_wav.clone()
        self.input_wav_res = torch.zeros(
            320 * self.input_wav.shape[0] // self.zc,
            device=self.device, dtype=torch.float32
        )
        self.rms_buffer = np.zeros(4 * self.zc, dtype="float32")
        self.sola_buffer = torch.zeros(self.sola_buffer_frame, device=self.device, dtype=torch.float32)
        self.nr_buffer = self.sola_buffer.clone()
        self.output_buffer = self.input_wav.clone()
        self.skip_head = self.extra_frame // self.zc
        self.skip_tail = self.extra_frame_right // self.zc
        self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // self.zc
        
        self.fade_in_window = (torch.sin(0.5 * np.pi * torch.linspace(0.0, 1.0, steps=self.sola_buffer_frame, device=self.device, dtype=torch.float32)) ** 2)
        self.fade_out_window = 1 - self.fade_in_window
        
        self.resampler = tat.Resample(orig_freq=self.config.samplerate, new_freq=16000, dtype=torch.float32).to(self.device)
        if self.model_set[-1]["sampling_rate"] != self.config.samplerate:
            self.resampler2 = tat.Resample(orig_freq=self.model_set[-1]["sampling_rate"], new_freq=self.config.samplerate, dtype=torch.float32).to(self.device)
        else:
            self.resampler2 = None

        # VAD state
        self.vad_cache = {}
        # Keep the VAD chunk within what FSMN-VAD can handle, independent of
        # block_time: the slider goes down to 0.04s, which is below the model's
        # minimum. _audio_callback accumulates audio to fill exactly this much.
        self.vad_chunk_size = min(500, max(MIN_VAD_CHUNK_MS, 1000 * self.config.block_time))
        self.vad_pending = np.array([], dtype=np.float32)
        self.vad_speech_detected = False
        self.vad_pos_start = False
        self.last_vad_time = 0
        self.last_vad_end_time = 0
        self.last_vad_reset_time = time.time()

    def start_stream(self):
        if not self.flag_vc:
            # Select the devices BEFORE preparing buffers: _prepare_buffers sizes
            # everything from get_device_samplerate()/get_device_channels(), which
            # read sd.default.device. Preparing first would build the buffers from
            # whatever device was previously selected.
            if self.config.sg_input_device and self.config.sg_output_device:
                 try:
                    # Guard the host API lookup: a stale config (e.g. a Windows
                    # "MME"/"Windows WASAPI" value loaded on Linux) won't match any
                    # current host API. Instead of crashing, fall back to matching
                    # the device names across all host APIs.
                    # NOTE: query_hostapis() dicts have no 'index' key - a host API's
                    # index is simply its position in the list.
                    hostapi_idx = next(
                        (i for i, h in enumerate(sd.query_hostapis()) if h['name'] == self.config.sg_hostapi),
                        None,
                    )
                    if hostapi_idx is None:
                        print(f"Host API '{self.config.sg_hostapi}' not found; matching devices by name across all host APIs.")
                    # Find device indices
                    input_idx = -1
                    output_idx = -1
                    for d in sd.query_devices():
                         if hostapi_idx is not None and d['hostapi'] != hostapi_idx:
                             continue
                         if d['name'] == self.config.sg_input_device and d['max_input_channels'] > 0:
                             input_idx = d['index']
                         if d['name'] == self.config.sg_output_device and d['max_output_channels'] > 0:
                             output_idx = d['index']

                    if input_idx >= 0 and output_idx >= 0:
                        sd.default.device = (input_idx, output_idx)
                    else:
                        print("Selected input/output device not found; using system default devices.")
                 except Exception as e:
                     print(f"Error setting devices: {e}")

            # Resolve the host API actually in use, which is what decides whether the
            # timestamps are real. config.sg_hostapi is only what the GUI selected: if
            # the device match above failed we are now on the system default, which may
            # sit on a different host API.
            try:
                self._hostapi_name = sd.query_hostapis(
                    sd.query_devices(kind="input")["hostapi"]
                )["name"]
            except Exception as e:
                self._hostapi_name = ""
                print(f"Could not resolve the host API in use ({e}); assuming no timestamps.")

            self._prepare_buffers()
            self.flag_vc = True

            if "WASAPI" in self.config.sg_hostapi and self.config.wasapi_exclusive:
                extra_settings = sd.WasapiSettings(exclusive=True)
            else:
                extra_settings = None
                
            self.stream = sd.Stream(
                callback=self._audio_callback,
                blocksize=self.block_frame,
                samplerate=self.config.samplerate,
                channels=self.config.channels,
                dtype="float32",
                latency="low",
                extra_settings=extra_settings,
            )

            # Compute delay_time BEFORE starting the stream, so the audio
            # callback (which reads self.delay_time) never sees a stale/undefined
            # value. self.stream.latency is available once the stream is opened.
            input_latency = self.stream.latency[0]
            output_latency = self.stream.latency[1]
            # Duplex buffering only: in_lat/out_lat each already cover ~one block,
            # so this must NOT add block_time again. Used as the fallback latency
            # when the host API has no usable timestamps.
            self._hw_latency = input_latency + output_latency
            self.delay_time = (
                input_latency + output_latency
                + self.config.block_time
                + self.config.crossfade_time
                + 0.01
                + 0.5
            )
            print(f"Audio Delay Calculated: {self.delay_time*1000:.2f}ms")
            # Breakdown to diagnose where the latency comes from. in_lat/out_lat
            # is the PortAudio hardware buffering (dominated by 'high' latency
            # mode on Linux/Pulse); the algorithmic look-ahead (extra_time_right)
            # is REAL perceived latency but is NOT part of delay_time.
            print(
                f"  [breakdown] in_lat={input_latency*1000:.1f}ms "
                f"out_lat={output_latency*1000:.1f}ms "
                f"block={self.config.block_time*1000:.1f}ms "
                f"crossfade={self.config.crossfade_time*1000:.1f}ms fixed=510.0ms | "
                f"look-ahead extra_time_right={self.config.extra_time_right*1000:.1f}ms (not in delay_time)"
            )

            self.stream.start()

    def stop_stream(self):
        if self.flag_vc:
            self.flag_vc = False
            if self.stream is not None:
                self.stream.abort()
                self.stream.close()
                self.stream = None

            # Latency is meaningless while stopped; consumers read 0 as "audio idle".
            self.measured_delay_time = 0.0
            self._measured_logged = False

    @torch.no_grad()
    def _custom_infer(self, input_wav_res):
        (model, semantic_fn, vocoder_fn, campplus_model, to_mel, mel_fn_args) = self.model_set
        sr = mel_fn_args["sampling_rate"]
        hop_length = mel_fn_args["hop_size"]
        
        cd_difference = self.active_extra_time_ce - self.active_extra_time
        max_prompt_length = self.config.max_prompt_length # assumed available in config

        if self.ce_dit_difference != cd_difference:
            self.ce_dit_difference = cd_difference
        
        reference_changed = self.reference_wav_name != self.config.reference_audio_path
        if self.prompt_condition is None or reference_changed or self.prompt_len != max_prompt_length:
            self.prompt_len = max_prompt_length
            print(f"Setting max prompt length to {max_prompt_length} seconds.")
            if reference_changed:
                # The samples must follow the path, not just the name: _prepare_buffers
                # only loads the reference once, so without this a mid-stream switch
                # would rebuild the prompt from the PREVIOUS file's audio and keep
                # converting to the old voice.
                self.reference_wav, _ = librosa.load(self.config.reference_audio_path, sr=sr)
            # Re-process reference wav
            ref_wav_slice = self.reference_wav[:int(sr * self.prompt_len)]
            reference_wav_tensor = torch.from_numpy(ref_wav_slice).to(self.device)
            
            ori_waves_16k = torchaudio.functional.resample(reference_wav_tensor, sr, 16000)
            S_ori = semantic_fn(ori_waves_16k.unsqueeze(0))
            feat2 = torchaudio.compliance.kaldi.fbank(
                ori_waves_16k.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000
            )
            feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
            self.style2 = campplus_model(feat2.unsqueeze(0))
            
            self.mel2 = to_mel(reference_wav_tensor.unsqueeze(0))
            target2_lengths = torch.LongTensor([self.mel2.size(2)]).to(self.mel2.device)
            self.prompt_condition = model.length_regulator(
                S_ori, ylens=target2_lengths, n_quantizers=3, f0=None
            )[0]
            self.reference_wav_name = self.config.reference_audio_path

        converted_waves_16k = input_wav_res
        
        S_alt = semantic_fn(converted_waves_16k.unsqueeze(0))
        
        ce_dit_frame_difference = int(self.ce_dit_difference * 50)
        S_alt = S_alt[:, ce_dit_frame_difference:]
        target_lengths = torch.LongTensor([(self.skip_head + self.return_length + self.skip_tail - ce_dit_frame_difference) / 50 * sr // hop_length]).to(S_alt.device)
        
        cond = model.length_regulator(
            S_alt, ylens=target_lengths, n_quantizers=3, f0=None
        )[0]
        cat_condition = torch.cat([self.prompt_condition, cond], dim=1)
        
        with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.fp16 else torch.float32):
            vc_target = model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(self.mel2.device),
                self.mel2,
                self.style2,
                None,
                n_timesteps=int(self.config.diffusion_steps),
                inference_cfg_rate=self.config.inference_cfg_rate,
            )
            vc_target = vc_target[:, :, self.mel2.size(-1) :]
            vc_wave = vocoder_fn(vc_target).squeeze()
            
        output_len = self.return_length * sr // 50
        tail_len = self.skip_tail * sr // 50
        output = vc_wave[-output_len - tail_len: -tail_len]
        return output

    def _measure_latency(self, times):
        """Real capture->playback latency (s) from PortAudio timestamps, or None.

        Returns None when the host API does not implement the timestamps, in which
        case the caller falls back to the hardware-latency estimate. Checking the
        numbers alone is not enough: WASAPI and DirectSound both return values that
        look plausible but are far too small, so the host API is checked first.
        """
        if self._hostapi_name not in TIMESTAMP_CAPABLE_HOSTAPIS:
            return None
        adc = times.inputBufferAdcTime
        dac = times.outputBufferDacTime
        if adc <= 0 or dac <= 0:
            return None
        span = dac - adc
        if not 0.0 < span < MAX_PLAUSIBLE_LATENCY_S:
            return None
        return span + self.active_extra_time_right

    def _audio_callback(self, indata, outdata, frames, times, status):
        start_time = time.perf_counter()

        # End-to-end audio latency driving A/V sync. Prefer the real measurement
        # from PortAudio timestamps; fall back to the hardware-latency estimate
        # where the host API cannot provide them.
        measured = self._measure_latency(times)
        if measured is None:
            self.measured_delay_time = self._hw_latency + self.active_extra_time_right
            if not self._measured_logged:
                print(
                    f"  [measured] host API '{self._hostapi_name}' does not implement real "
                    f"timestamps; falling back to estimate = {self.measured_delay_time*1000:.1f}ms. "
                    f"A/V sync needs a one-off Sync Offset trim (the estimate is short by a fixed "
                    f"amount for a given device and block_time)."
                )
                self._measured_logged = True
        else:
            self.measured_delay_time = measured
            if not self._measured_logged:
                print(
                    f"  [measured] real audio latency = {self.measured_delay_time*1000:.1f}ms "
                    f"(vs estimated delay_time = {self.delay_time*1000:.1f}ms)"
                )
                self._measured_logged = True

        indata = librosa.to_mono(indata.T)
        
        # Synchronization setup
        if self.device.type == "mps":
             torch.mps.synchronize()
        elif self.device.type == "cuda":
             torch.cuda.synchronize()

        # VAD Processing
        indata_16k = librosa.resample(indata, orig_sr=self.config.samplerate, target_sr=16000)
        
        # FSMN-VAD's streaming cache degrades under sustained speech: funasr grows
        # cache["stats"].scores / data_buf_all by concatenation on every chunk and
        # only trims them once a segment end is confirmed, which never comes while
        # someone keeps talking. Resetting periodically bounds that.
        # No warmup is needed after the reset: vad_pos_start only clears on an
        # explicit segment end, and funasr cannot report an end before a start on a
        # fresh cache, so the gate below stays open across the reset.
        if time.time() - self.last_vad_reset_time > 10.0:
            self.vad_cache = {}
            self.last_vad_reset_time = time.time()

        # Feed the VAD in fixed chunks rather than passing the block straight
        # through: a block smaller than the model's minimum makes it raise, and a
        # block larger than vad_chunk_size would disagree with the chunk_size we
        # declare. Accumulate, then consume whole chunks.
        self.vad_pending = np.concatenate((self.vad_pending, indata_16k))
        chunk_samples = int(self.vad_chunk_size * 16)  # 16 samples per ms at 16 kHz
        while len(self.vad_pending) >= chunk_samples:
            vad_chunk = self.vad_pending[:chunk_samples]
            self.vad_pending = self.vad_pending[chunk_samples:]
            res = self.vad_model.generate(
                input=vad_chunk, cache=self.vad_cache, is_final=False, chunk_size=self.vad_chunk_size
            )
            for segment in res[0]["value"]:
                s_time, e_time = segment
                if s_time != -1:
                    self.vad_pos_start = True
                if e_time != -1:
                    self.vad_pos_start = False
                    self.last_vad_end_time = time.time()


        if self.vad_pos_start:
            self.vad_speech_detected = True
        else:
            if self.vad_speech_detected and (time.time() - self.last_vad_end_time > 0.5):
                self.vad_speech_detected = False

        if not self.vad_pos_start and not self.vad_speech_detected:
            self.vad_cache = {}
            # The cache is already fresh, so hold off the periodic reset above -
            # otherwise its timer keeps ageing through silence and fires for nothing.
            self.last_vad_reset_time = time.time()

        # Preprocessing
        self.input_wav[: -self.block_frame] = self.input_wav[self.block_frame :].clone()
        self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(self.device)
        
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[self.block_frame_16k :].clone()
        # Note: Using librosa resample on CPU then moving to GPU to match original implementation's behavior logic roughly
        # The original code had a commented out torch resampler and used librosa.
        resampled_indata = librosa.resample(self.input_wav[-indata.shape[0] - 2 * self.zc :].cpu().numpy(), orig_sr=self.config.samplerate, target_sr=16000)[320:]
        self.input_wav_res[-320 * (indata.shape[0] // self.zc + 1) :] = torch.from_numpy(resampled_indata).to(self.device)

        # Inference
        infer_wav = None
        
        # Check if we're in input monitoring mode (passthrough)
        if self.config.function == "im":
            # Input monitoring mode: directly pass through the input audio
            infer_wav = self.input_wav[self.extra_frame : self.extra_frame + self.block_frame + self.sola_buffer_frame + self.sola_search_frame].clone()
        elif self.active_extra_time_ce - self.active_extra_time < 0:
             if time.time() - self.last_error_print_time > 1.0:
                 print("Error: Content encoder extra context must be greater than DiT extra context!")
                 self.last_error_print_time = time.time()
             infer_wav = torch.zeros_like(self.input_wav[self.extra_frame :])
        else:
             try:
                 infer_wav = self._custom_infer(self.input_wav_res)
                 if self.resampler2 is not None:
                     infer_wav = self.resampler2(infer_wav)
             except Exception as e:
                 if time.time() - self.last_error_print_time > 1.0:
                     print(f"Inference error: {e}")
                     self.last_error_print_time = time.time()
                 infer_wav = torch.zeros_like(self.input_wav[self.extra_frame :])

        # Only apply VAD muting during voice conversion mode
        if self.config.function == "vc" and not self.vad_speech_detected:
             infer_wav = torch.zeros_like(self.input_wav[self.extra_frame :])

        # SOLA
        conv_input = infer_wav[None, None, : self.sola_buffer_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, self.sola_buffer_frame, device=self.device),
            ) + 1e-8
        )
        tensor = cor_nom[0, 0] / cor_den[0, 0]
        if tensor.numel() > 1:
            if sys.platform == "darwin":
                _, sola_offset = torch.max(tensor, dim=0)
                sola_offset = sola_offset.item()
            else:
                sola_offset = torch.argmax(tensor, dim=0).item()
        else:
            sola_offset = tensor.item()
            
        infer_wav = infer_wav[int(sola_offset):]
        infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
        infer_wav[: self.sola_buffer_frame] += (self.sola_buffer * self.fade_out_window)
        self.sola_buffer[:] = infer_wav[self.block_frame : self.block_frame + self.sola_buffer_frame]
        
        final_output = (
            infer_wav[: self.block_frame]
            .repeat(self.config.channels, 1)
            .t()
            .cpu()
            .numpy()
        )
        
        total_time = time.perf_counter() - start_time
        if self.on_perf_update:
            self.on_perf_update(total_time, self.delay_time, self.measured_delay_time)

        outdata[:] = final_output

