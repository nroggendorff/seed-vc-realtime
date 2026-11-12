import os
import shutil

import torch

CKPT_A = "./runs/voice_a/ft_model.pth"
CKPT_B = "./runs/voice_b/ft_model.pth"
MIX = 0.5  # 0 = all A, 1 = all B
OUTPUT = "./runs/merged/ft_model.pth"


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    a = torch.load(CKPT_A, map_location="cpu")["net"]
    b = torch.load(CKPT_B, map_location="cpu")["net"]
    if a.keys() != b.keys() or any(a[m].keys() != b[m].keys() for m in a):
        raise SystemExit(
            "The checkpoints have different layers. Were both trained with the same config?"
        )

    merged = {}
    for module in a:
        merged[module] = {}
        for name, ta in a[module].items():
            tb = b[module][name]
            if ta.shape != tb.shape:
                raise SystemExit(
                    f"{module}.{name} differs in shape. Were both trained with the same config?"
                )
            # Integer buffers (positions etc.) are identical in both; only blend real weights.
            merged[module][name] = (
                (1 - MIX) * ta + MIX * tb if ta.is_floating_point() else ta
            )

    out_dir = os.path.dirname(OUTPUT) or "."
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"net": merged}, OUTPUT)

    # The GUI needs the config alongside the checkpoint; finetune.py leaves one next to each.
    config_src = os.path.join(os.path.dirname(CKPT_A), "config.yml")
    config_dst = os.path.join(out_dir, "config.yml")
    if os.path.abspath(config_src) != os.path.abspath(config_dst):
        shutil.copyfile(config_src, config_dst)

    print(
        f"Saved {os.path.abspath(OUTPUT)} ({1 - MIX:.0%} A, {MIX:.0%} B)\nRun it with:\n"
        f"  python real-time-gui.py --checkpoint-path {OUTPUT} --config-path {config_dst}"
    )


if __name__ == "__main__":
    main()
