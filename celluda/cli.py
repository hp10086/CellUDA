# -*- coding: utf-8 -*-
"""Command-line front end used by ``run.py``.

One YAML config from ``configs/`` decides what happens: a config without a ``uda`` block trains
the source-only classifier head, ``uda.enable: true`` runs CellUDA adaptation; ``--mode`` switches
to embedding extraction only or to evaluation of an existing head. A few entries (GPU, seed,
checkpoint / data / output paths) can be overridden on the command line.
"""

import argparse
import os
from typing import Optional

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")
os.environ.setdefault("WANDB_START_METHOD", "thread")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("WANDB__SERVICE_WAIT", "300")

import wandb
import yaml

from celluda.experiment import ExperimentCellVitClassifier

MODES = ("train", "extract", "evaluate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CellUDA: source-only head training, CellUDA adaptation, embedding extraction or evaluation, "
        "selected by the config (uda.enable) and --mode.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="YAML config, see configs/")
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="train",
        help="train: source-only head (no uda block) or CellUDA adaptation (uda.enable true), then the final "
        "target evaluation; extract: only CellViT segmentation + cell-embedding caching; "
        "evaluate: evaluate `classifier_path` on the target test split",
    )
    parser.add_argument("--gpu", type=int, help="CUDA device id (overrides `gpu`)")
    parser.add_argument("--seed", type=int, help="Random seed (overrides `random_seed`)")
    parser.add_argument("--cellvit_path", help="CellViT-SAM-H segmentation checkpoint (overrides `cellvit_path`)")
    parser.add_argument(
        "--classifier_path",
        help="Classifier-head checkpoint to initialise from / to evaluate (overrides `classifier_path`)",
    )
    parser.add_argument("--dataset_path", help="Source dataset root (overrides `data.dataset_path`)")
    parser.add_argument("--target_dataset_path", help="Target dataset root (overrides `data.target_dataset_path`)")
    parser.add_argument(
        "--log_dir", help="Log / offline-W&B directory (overrides `logging.log_dir` and `logging.wandb_dir`)"
    )
    parser.add_argument("--cache_dir", help="HDF5 cell-embedding cache directory (overrides `training.cache_dir`)")
    parser.add_argument(
        "--export_dir", help="Directory for the exported classifier head (overrides `training.classifier_export_dir`)"
    )
    parser.add_argument("--epochs", type=int, help="Overrides `training.epochs`")
    parser.add_argument("--checkpoint", help="Resume training from a trainer checkpoint (.pth)")
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="--mode extract: re-extract even if a cache file exists"
    )
    return parser


def load_config(args: argparse.Namespace) -> dict:
    with open(args.config, "r") as f:
        cfg = dict(yaml.safe_load(f))
    cfg.setdefault("data", {})
    cfg.setdefault("training", {})
    cfg.setdefault("logging", {})

    if args.gpu is not None:
        cfg["gpu"] = int(args.gpu)
    if args.seed is not None:
        cfg["random_seed"] = int(args.seed)
    if args.cellvit_path:
        cfg["cellvit_path"] = args.cellvit_path
    if args.classifier_path:
        cfg["classifier_path"] = args.classifier_path
    if args.dataset_path:
        cfg["data"]["dataset_path"] = args.dataset_path
    if args.target_dataset_path:
        cfg["data"]["target_dataset_path"] = args.target_dataset_path
    if args.log_dir:
        cfg["logging"]["log_dir"] = args.log_dir
        cfg["logging"]["wandb_dir"] = args.log_dir
    if args.cache_dir:
        cfg["training"]["cache_dir"] = args.cache_dir
    if args.export_dir:
        cfg["training"]["classifier_export_dir"] = args.export_dir
    if args.epochs is not None:
        cfg["training"]["epochs"] = int(args.epochs)
    if args.checkpoint:
        cfg["checkpoint"] = args.checkpoint
    if args.overwrite_cache:
        cfg["training"]["overwrite_cache"] = True
    return cfg


def main(argv: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args)

    uda_enabled = bool((cfg.get("uda") or {}).get("enable", False))
    if args.mode == "train":
        print(f"[CellUDA] {args.config}: " + ("CellUDA adaptation (uda.enable: true)" if uda_enabled else "source-only head training (no uda block)"))
    elif args.mode == "evaluate" and not cfg.get("classifier_path"):
        parser.error("--mode evaluate needs a head: pass --classifier_path or set `classifier_path` in the config.")

    experiment = ExperimentCellVitClassifier(
        default_conf=cfg,
        checkpoint=cfg.get("checkpoint"),
        just_load_model=False,
    )
    experiment.run_experiment(mode=args.mode)
    wandb.finish()
