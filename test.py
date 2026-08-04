#!/usr/bin/env python3
"""Evaluate a DIAF-Net checkpoint on aligned, synthetic, or natural misalignment."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import DIAFConfig
from datasets import MultiContrastMRIDataset
from models import DIAFNet
from utils import AverageMeter, image_metrics, save_image, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", choices=("brats2020", "fastmri"), required=True)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--channels", type=int, choices=(1, 3), default=3)
    parser.add_argument("--split", default="val")
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-dir", default=None)
    return parser.parse_args()


def extract_state_dict(checkpoint) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(checkpoint, dict) and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise TypeError("Could not find a model state dict in the checkpoint")


@torch.inference_mode()
def evaluate_condition(
    model: DIAFNet,
    loader: DataLoader,
    device: torch.device,
    save_dir: Path | None,
) -> dict[str, float]:
    model.eval()
    meters = {name: AverageMeter() for name in ("psnr", "ssim", "rmse")}
    for batch in tqdm(loader, leave=False):
        target_lr = batch["target_lr"].to(device, non_blocking=True)
        reference_hr = batch["reference_hr"].to(device, non_blocking=True)
        target_hr = batch["target_hr"].to(device, non_blocking=True)
        prediction = model(target_lr, reference_hr)

        names = batch["name"]
        for index in range(prediction.shape[0]):
            metrics = image_metrics(prediction[index], target_hr[index])
            for name, value in metrics.items():
                meters[name].update(value)
            if save_dir is not None:
                save_image(prediction[index], save_dir / str(names[index]))
    return {name: meter.average for name, meter in meters.items()}


def main() -> None:
    args = parse_args()
    requested = torch.device(args.device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        requested = torch.device("cpu")
    device = requested

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_kwargs = checkpoint.get("model_kwargs") if isinstance(checkpoint, dict) else None
    if not isinstance(model_kwargs, dict):
        model_kwargs = DIAFConfig(channels=args.channels).model_kwargs()
    model_kwargs["in_channels"] = args.channels
    model_kwargs["out_channels"] = args.channels
    model = DIAFNet(**model_kwargs)

    state_dict = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    conditions = args.conditions
    if conditions is None:
        conditions = ["native"] if args.dataset == "fastmri" else ["clean", "mis3", "mis6", "mis9"]

    results = {}
    for condition in conditions:
        dataset = MultiContrastMRIDataset(
            root=args.data_root,
            dataset=args.dataset,
            scale=args.scale,
            split=args.split,
            condition=condition,
            channels=args.channels,
            train_misalignment=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        condition_save_dir = None
        if args.save_dir:
            condition_save_dir = Path(args.save_dir) / condition
        metrics = evaluate_condition(model, loader, device, condition_save_dir)
        results[condition] = metrics
        print(
            f"{condition:>6s} | PSNR {metrics['psnr']:.3f} | "
            f"SSIM {metrics['ssim']:.4f} | RMSE {metrics['rmse']:.4f}"
        )

    if args.save_dir:
        save_json(results, Path(args.save_dir) / "metrics.json")


if __name__ == "__main__":
    main()
