import os
import sys
import time
import json
import struct
import warnings
import re
import argparse
import numpy as np
import torch
import FreeSimpleGUI as sg
from multiprocessing import shared_memory
from dotenv import load_dotenv
from realtime_vc_engine import RealtimeVCEngine
from audio_modules.commons import str2bool

load_dotenv()
os.environ["OMP_NUM_THREADS"] = "4"
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

# Relative to this script, not the CWD, which the external launcher changes.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "configs", "config.json")
INUSE_CONFIG_DIR = os.path.join(SCRIPT_DIR, "configs", "inuse")
INUSE_CONFIG_PATH = os.path.join(INUSE_CONFIG_DIR, "config.json")

warnings.simplefilter("ignore")


def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)


class GUI:
    def __init__(self, args) -> None:
        self.engine = RealtimeVCEngine(args)

        self.engine.on_perf_update = self.update_perf_stats
        self.last_gui_update = 0

        self.function = "vc"

        # Publishes the measured latency for the external video process to sync to.
        self.shm = None
        shm_name = os.environ.get("RAS_SHARED_MEM_NAME")
        if shm_name:
            try:
                self.shm = shared_memory.SharedMemory(name=shm_name)
                print(f"Connected to Shared Memory: {shm_name}")
            except Exception as e:
                print(f"Failed to connect to Shared Memory: {e}")

        self.update_devices()
        self.launcher()

    def publish_audio_delay(self, delay_seconds):
        """Publish audio latency to shared memory for the legacy video process."""
        if not self.shm:
            return
        try:
            self.shm.buf[0:8] = struct.pack("d", delay_seconds * 1000)
        except Exception as e:
            print(f"Failed to publish audio delay to SHM: {e}")

    def update_perf_stats(self, infer_time, _delay_time, measured_delay_time):
        """Callback to update UI with performance stats"""
        # Every call, so video tracks latency promptly; only the UI update is throttled.
        self.publish_audio_delay(measured_delay_time)

        if time.time() - self.last_gui_update < 0.1:
            return
        self.last_gui_update = time.time()

        try:
            self.window["infer_time"].update(int(infer_time * 1000))
            self.window["delay_time"].update(int(np.round(measured_delay_time * 1000)))
        except:
            pass

    def load(self):
        try:
            os.makedirs(INUSE_CONFIG_DIR, exist_ok=True)
            if not os.path.exists(INUSE_CONFIG_PATH):
                import shutil

                shutil.copy(DEFAULT_CONFIG_PATH, INUSE_CONFIG_PATH)
            with open(INUSE_CONFIG_PATH, "r") as j:
                data = json.load(j)
                data["sr_model"] = data["sr_type"] == "sr_model"
                data["sr_device"] = data["sr_type"] == "sr_device"

                if data["sg_hostapi"] in self.engine.hostapis:
                    self.update_devices(hostapi_name=data["sg_hostapi"])
                    if (
                        data["sg_input_device"] not in self.engine.input_devices
                        or data["sg_output_device"] not in self.engine.output_devices
                    ):
                        self.update_devices()
                        data["sg_hostapi"] = self.engine.hostapis[0]
                        # Not devices[0]: on Linux that's usually the monitor's HDMI audio.
                        data["sg_input_device"] = self.engine.preferred_device("input")
                        data["sg_output_device"] = self.engine.preferred_device(
                            "output"
                        )
                else:
                    self.update_devices()
                    data["sg_hostapi"] = self.engine.hostapis[0]
                    data["sg_input_device"] = self.engine.preferred_device("input")
                    data["sg_output_device"] = self.engine.preferred_device("output")
        except Exception as e:
            print(f"Error loading config: {e}")
            self.update_devices()
            data = {
                "sg_hostapi": self.engine.hostapis[0] if self.engine.hostapis else "",
                "sg_input_device": (
                    self.engine.input_devices[0] if self.engine.input_devices else ""
                ),
                "sg_output_device": (
                    self.engine.output_devices[0] if self.engine.output_devices else ""
                ),
                "sr_type": "sr_model",
                "sr_model": True,
                "sr_device": False,
                "diffusion_steps": 10,
                "inference_cfg_rate": 0.7,
                "max_prompt_length": 3.0,
                "block_time": 0.25,
                "crossfade_length": 0.05,
                "extra_time_ce": 2.5,
                "extra_time": 0.5,
                "extra_time_right": 2.0,
            }
        return data

    def launcher(self):
        data = self.load()
        sg.theme("LightBlue3")
        layout = [
            [
                sg.Frame(
                    title="Load reference audio",
                    layout=[
                        [
                            sg.Input(
                                default_text=data.get("reference_audio_path", ""),
                                key="reference_audio_path",
                            ),
                            sg.FileBrowse(
                                "choose an audio file",
                                initial_folder=os.path.join(
                                    os.getcwd(), "examples/reference"
                                ),
                                file_types=[
                                    ("WAV Files", "*.wav"),
                                    ("MP3 Files", "*.mp3"),
                                    ("FLAC Files", "*.flac"),
                                    ("M4A Files", "*.m4a"),
                                    ("OGG Files", "*.ogg"),
                                    ("Opus Files", "*.opus"),
                                ],
                            ),
                        ],
                    ],
                )
            ],
            [
                sg.Frame(
                    layout=[
                        [
                            sg.Text("Device type"),
                            sg.Combo(
                                self.engine.hostapis,
                                key="sg_hostapi",
                                default_value=data.get("sg_hostapi", ""),
                                enable_events=True,
                                size=(20, 1),
                            ),
                            sg.Checkbox(
                                "WASAPI Exclusive Device",
                                key="sg_wasapi_exclusive",
                                default=data.get("sg_wasapi_exclusive", False),
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Text("Input Device"),
                            sg.Combo(
                                self.engine.input_devices,
                                key="sg_input_device",
                                default_value=data.get("sg_input_device", ""),
                                enable_events=True,
                                size=(45, 1),
                            ),
                        ],
                        [
                            sg.Text("Output Device"),
                            sg.Combo(
                                self.engine.output_devices,
                                key="sg_output_device",
                                default_value=data.get("sg_output_device", ""),
                                enable_events=True,
                                size=(45, 1),
                            ),
                        ],
                        [
                            sg.Button("Reload devices", key="reload_devices"),
                            sg.Radio(
                                "Use model sampling rate",
                                "sr_type",
                                key="sr_model",
                                default=data.get("sr_model", True),
                                enable_events=True,
                            ),
                            sg.Radio(
                                "Use device sampling rate",
                                "sr_type",
                                key="sr_device",
                                default=data.get("sr_device", False),
                                enable_events=True,
                            ),
                            sg.Text("Sampling rate:"),
                            sg.Text("", key="sr_stream"),
                        ],
                    ],
                    title="Sound Device",
                )
            ],
            [
                sg.Frame(
                    layout=[
                        [
                            sg.Text("Diffusion steps"),
                            sg.Slider(
                                range=(1, 30),
                                key="diffusion_steps",
                                resolution=1,
                                orientation="h",
                                default_value=data.get("diffusion_steps", 10),
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Text("Inference cfg rate"),
                            sg.Slider(
                                range=(0.0, 1.0),
                                key="inference_cfg_rate",
                                resolution=0.1,
                                orientation="h",
                                default_value=data.get("inference_cfg_rate", 0.7),
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Text("Max prompt length (s)"),
                            sg.Slider(
                                range=(1.0, 20.0),
                                key="max_prompt_length",
                                resolution=0.5,
                                orientation="h",
                                default_value=data.get("max_prompt_length", 3.0),
                                enable_events=True,
                            ),
                        ],
                    ],
                    title="Regular settings",
                ),
                sg.Frame(
                    layout=[
                        [
                            sg.Text("Block time"),
                            sg.Slider(
                                range=(0.04, 3.0),
                                key="block_time",
                                resolution=0.02,
                                orientation="h",
                                default_value=data.get("block_time", 1.0),
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Text("Crossfade length"),
                            sg.Slider(
                                range=(0.02, 0.5),
                                key="crossfade_length",
                                resolution=0.02,
                                orientation="h",
                                default_value=data.get("crossfade_length", 0.1),
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Text("Extra content encoder context time (left)"),
                            sg.Slider(
                                range=(0.5, 10.0),
                                key="extra_time_ce",
                                resolution=0.1,
                                orientation="h",
                                default_value=data.get("extra_time_ce", 5.0),
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Text("Extra DiT context time (left)"),
                            sg.Slider(
                                range=(0.5, 10.0),
                                key="extra_time",
                                resolution=0.1,
                                orientation="h",
                                default_value=data.get("extra_time", 5.0),
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Text("Extra context time (right)"),
                            sg.Slider(
                                range=(0.02, 10.0),
                                key="extra_time_right",
                                resolution=0.02,
                                orientation="h",
                                default_value=data.get("extra_time_right", 2.0),
                                enable_events=True,
                            ),
                        ],
                    ],
                    title="Performance settings",
                ),
            ],
            [
                sg.Button("Start Voice Conversion", key="start_vc"),
                sg.Button("Stop Voice Conversion", key="stop_vc"),
                sg.Radio(
                    "Input listening",
                    "function",
                    key="im",
                    default=False,
                    enable_events=True,
                ),
                sg.Radio(
                    "Voice Conversion",
                    "function",
                    key="vc",
                    default=True,
                    enable_events=True,
                ),
                sg.Text("Algorithm delay (ms):"),
                sg.Text("0", key="delay_time"),
                sg.Text("Inference time (ms):"),
                sg.Text("0", key="infer_time"),
            ],
        ]
        self.window = sg.Window("Seed-VC - GUI", layout=layout, finalize=True)
        self.event_handler()

    def update_devices(self, hostapi_name=None):
        self.engine.update_devices(hostapi_name)

    def event_handler(self):
        while True:
            event, values = self.window.read()
            if event == sg.WINDOW_CLOSED:
                self.engine.stop_stream()
                self.publish_audio_delay(0.0)  # tell the video process audio is idle
                exit()

            if event == "reload_devices" or event == "sg_hostapi":
                hostapi = values["sg_hostapi"]
                self.update_devices(hostapi_name=hostapi)

                if hostapi not in self.engine.hostapis:
                    hostapi = self.engine.hostapis[0] if self.engine.hostapis else ""

                self.window["sg_hostapi"].Update(values=self.engine.hostapis)
                self.window["sg_hostapi"].Update(value=hostapi)

                self.window["sg_input_device"].Update(values=self.engine.input_devices)
                if self.engine.input_devices:
                    self.window["sg_input_device"].Update(
                        value=self.engine.input_devices[0]
                    )

                self.window["sg_output_device"].Update(
                    values=self.engine.output_devices
                )
                if self.engine.output_devices:
                    self.window["sg_output_device"].Update(
                        value=self.engine.output_devices[0]
                    )

            if event == "start_vc" and not self.engine.flag_vc:
                if self.set_values(values) == True:
                    printt("cuda_is_available: %s", torch.cuda.is_available())
                    self.engine.start_stream()

                    settings = {
                        "reference_audio_path": values["reference_audio_path"],
                        "sg_hostapi": values["sg_hostapi"],
                        "sg_wasapi_exclusive": values["sg_wasapi_exclusive"],
                        "sg_input_device": values["sg_input_device"],
                        "sg_output_device": values["sg_output_device"],
                        "sr_type": ["sr_model", "sr_device"][
                            [values["sr_model"], values["sr_device"]].index(True)
                        ],
                        "diffusion_steps": values["diffusion_steps"],
                        "inference_cfg_rate": values["inference_cfg_rate"],
                        "max_prompt_length": values["max_prompt_length"],
                        "block_time": values["block_time"],
                        "crossfade_length": values["crossfade_length"],
                        "extra_time_ce": values["extra_time_ce"],
                        "extra_time": values["extra_time"],
                        "extra_time_right": values["extra_time_right"],
                    }
                    try:
                        os.makedirs(INUSE_CONFIG_DIR, exist_ok=True)
                        with open(INUSE_CONFIG_PATH, "w") as j:
                            json.dump(settings, j)
                    except:
                        pass

                    self.window["sr_stream"].update(self.engine.config.samplerate)
                    if hasattr(self.engine, "delay_time"):
                        self.window["delay_time"].update(
                            int(np.round(self.engine.delay_time * 1000))
                        )

            # These apply live; every other setting takes effect on the next start.
            elif event == "diffusion_steps":
                self.engine.set_config(diffusion_steps=values["diffusion_steps"])
            elif event == "inference_cfg_rate":
                self.engine.set_config(inference_cfg_rate=values["inference_cfg_rate"])
            elif event in ["vc", "im"]:
                self.function = event
                self.engine.set_config(function=event)

            elif event == "stop_vc":
                # Only explicit stops (and device reloads) end the stream.
                self.engine.stop_stream()
                self.publish_audio_delay(0.0)  # tell the video process audio is idle

    def set_values(self, values):
        if len(values["reference_audio_path"].strip()) == 0:
            sg.popup("Choose an audio file")
            return False
        if re.compile("[^\x00-\x7f]+").findall(values["reference_audio_path"]):
            sg.popup("audio file path contains non-ascii characters")
            return False

        sr_type = ["sr_model", "sr_device"][
            [values["sr_model"], values["sr_device"]].index(True)
        ]

        self.engine.set_config(
            sg_hostapi=values["sg_hostapi"],
            wasapi_exclusive=values["sg_wasapi_exclusive"],
            sg_input_device=values["sg_input_device"],
            sg_output_device=values["sg_output_device"],
            reference_audio_path=values["reference_audio_path"],
            sr_type=sr_type,
            diffusion_steps=values["diffusion_steps"],
            inference_cfg_rate=values["inference_cfg_rate"],
            max_prompt_length=values["max_prompt_length"],
            block_time=values["block_time"],
            crossfade_time=values["crossfade_length"],
            extra_time_ce=values["extra_time_ce"],
            extra_time=values["extra_time"],
            extra_time_right=values["extra_time_right"],
        )
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-path", type=str, default=None, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--config-path", type=str, default=None, help="Path to the model config (.yml)"
    )
    parser.add_argument(
        "--fp16",
        type=str2bool,
        nargs="?",
        const=True,
        help="Whether to use fp16",
        default=True,
    )
    parser.add_argument("--gpu", type=int, help="Which GPU id to use", default=0)
    args = parser.parse_args()

    gui = GUI(args)
