#!/usr/bin/env python3
"""Train DIAF-Net for multi-contrast MRI super-resolution."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import DIAFConfig
from datasets import MultiContrastMRIDataset
from losses import (
    DIAFLoss,
    sample_nonzero_translation,
    translate_reference_batch,
)
from models import DIAFNet
from utils import AverageMeter, count_parameters, image_metrics, save_json, set_seed


def parse_args() -> argparse.Namespace:
    defaults = DIAFConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=defaults.data_root)
    parser.add_argument("--dataset", choices=("brats2020", "fastmri"), default=defaults.dataset)
    parser.add_argument("--scale", type=int, default=defaults.scale)
    parser.add_argument("--channels", type=int, choices=(1, 3), default=defaults.channels)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--lfcl-weight", type=float, default=defaults.lfcl_weight)
    parser.add_argument("--val-condition", default=defaults.val_condition)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--amp", action="store_true", default=defaults.amp)
    parser.add_argument(
        "--no-train-misalignment",
        action="store_false",
        dest="train_misalignment",
        default=defaults.train_misalignment,
    )
    parser.add_argument("--resume", default=None, help="Path to latest.pth or another checkpoint.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> DIAFConfig:
    return replace(
        DIAFConfig(),
        data_root=args.data_root,
        dataset=args.dataset,
        scale=args.scale,
        channels=args.channels,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lfcl_weight=args.lfcl_weight,
        val_condition=("native" if args.dataset == "fastmri" else args.val_condition),
        device=args.device,
        seed=args.seed,
        amp=args.amp,
        train_misalignment=args.train_misalignment,
    )


def build_loaders(config: DIAFConfig) -> tuple[DataLoader, DataLoader]:
    train_set = MultiContrastMRIDataset(
        root=config.data_root,
        dataset=config.dataset,
        scale=config.scale,
        split=config.train_split,
        condition="native" if config.dataset == "fastmri" else "clean",
        channels=config.channels,
        train_misalignment=config.train_misalignment,
        misalignment_probability=config.misalignment_probability,
        max_rotation_deg=config.max_rotation_deg,
        max_translation_px=config.max_translation_px,
    )
    val_set = MultiContrastMRIDataset(
        root=config.data_root,
        dataset=config.dataset,
        scale=config.scale,
        split=config.val_split,
        condition=config.val_condition,
        channels=config.channels,
        train_misalignment=False,
    )
    common = {
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": config.num_workers > 0,
    }
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader


def train_epoch(
    model: DIAFNet,
    loader: DataLoader,
    criterion: DIAFLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    config: DIAFConfig,
) -> dict[str, float]:
    model.train()
    meters = {name: AverageMeter() for name in ("total", "l1", "lfcl", "uedc")}
    progress = tqdm(loader, desc="train", leave=False)
    uedc_mode = "absolute" if config.dataset == "brats2020" else "relative"

    for batch in progress:
        target_lr = batch["target_lr"].to(device, non_blocking=True)
        reference_hr = batch["reference_hr"].to(device, non_blocking=True)
        reference_native = batch["reference_native"].to(device, non_blocking=True)
        target_hr = batch["target_hr"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        dx, dy = sample_nonzero_translation(
            reference_native,
            config.uedc_max_translation_px,
        )
        shifted_reference = translate_reference_batch(reference_native, dx, dy)

        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            prediction, auxiliary = model(target_lr, reference_hr, return_aux=True)
            native_alignment = model(
                target_lr,
                reference_native,
                alignment_aux_only=True,
            )
            shifted_alignment = model(
                target_lr,
                shifted_reference,
                alignment_aux_only=True,
            )
            loss, components = criterion(
                prediction=prediction,
                target=target_hr,
                sfde_output=auxiliary["sfde"],
                native_offsets=native_alignment["effective_offsets"],
                shifted_offsets=shifted_alignment["effective_offsets"],
                dx=dx,
                dy=dy,
                reference_size=reference_native.shape[-2:],
                uedc_mode=uedc_mode,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = target_lr.shape[0]
        for name in meters:
            meters[name].update(float(components[name].item()), batch_size)
        progress.set_postfix(loss=f"{meters['total'].average:.4f}")

    return {name: meter.average for name, meter in meters.items()}


@torch.inference_mode()
def validate(
    model: DIAFNet,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    meters = {name: AverageMeter() for name in ("psnr", "ssim", "rmse")}
    for batch in tqdm(loader, desc="val", leave=False):
        target_lr = batch["target_lr"].to(device, non_blocking=True)
        reference_hr = batch["reference_hr"].to(device, non_blocking=True)
        target_hr = batch["target_hr"].to(device, non_blocking=True)
        prediction = model(target_lr, reference_hr)
        for index in range(prediction.shape[0]):
            metrics = image_metrics(prediction[index], target_hr[index])
            for name, value in metrics.items():
                meters[name].update(value)
    return {name: meter.average for name, meter in meters.items()}


def save_checkpoint(
    path: Path,
    model: DIAFNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_psnr: float,
    config: DIAFConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_psnr": best_psnr,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config.to_dict(),
            "model_kwargs": config.model_kwargs(),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    set_seed(config.seed)

    requested = torch.device(config.device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        requested = torch.device("cpu")
    device = requested
    amp_enabled = bool(config.amp and device.type == "cuda")

    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config.to_dict(), output_dir / "config.json")

    train_loader, val_loader = build_loaders(config)
    model = DIAFNet(**config.model_kwargs()).to(device)
    criterion = DIAFLoss(
        lfcl_weight=config.lfcl_weight,
        uedc_weight=config.uedc_weight,
        uedc_smooth_l1_beta=config.uedc_smooth_l1_beta,
        uedc_ignore_border=config.uedc_ignore_border,
        uedc_flow_sign=config.uedc_flow_sign,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.min_learning_rate,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 1
    best_psnr = float("-inf")
    history: list[dict] = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_psnr = float(checkpoint.get("best_psnr", best_psnr))
        print(f"Resumed from epoch {start_epoch - 1}.")

    print(f"Device: {device}")
    print(f"Trainable parameters: {count_parameters(model) / 1e6:.3f} M")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    for epoch in range(start_epoch, config.epochs + 1):
        started = time.time()
        train_stats = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            amp_enabled,
            config,
        )
        val_stats = validate(model, val_loader, device)
        scheduler.step()

        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
            "train": train_stats,
            "validation": val_stats,
        }
        history.append(record)
        save_json(history, output_dir / "history.json")

        improved = val_stats["psnr"] > best_psnr
        if improved:
            best_psnr = val_stats["psnr"]
            save_checkpoint(
                checkpoint_dir / "best.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_psnr,
                config,
            )
        save_checkpoint(
            checkpoint_dir / "latest.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            best_psnr,
            config,
        )
        if epoch % config.save_every == 0:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:03d}.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_psnr,
                config,
            )

        print(
            f"Epoch {epoch:03d}/{config.epochs} | "
            f"loss {train_stats['total']:.4f} | "
            f"PSNR {val_stats['psnr']:.3f} | "
            f"SSIM {val_stats['ssim']:.4f} | "
            f"RMSE {val_stats['rmse']:.4f}"
            + (" | best" if improved else "")
        )


if __name__ == "__main__":
    main()
