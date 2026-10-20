# -*- coding: utf-8 -*-
# CellVit Experiment Class for Cell Classification Module Fine-Tuning
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
# Modified from CellViT++ (https://github.com/TIO-IKIM/CellViT-plus-plus) — changes: rewritten for source->target UDA: builds source/target DetectionDatasets from separate dataset roots, loads a source classifier head, dispatches celluda.trainer.CellViTHeadTrainer with train/extract/evaluate modes; SAM-H backbone only; W&B sweep support removed.

import copy
import datetime
import os
import random
import uuid
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import wandb

os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")
os.environ.setdefault("WANDB_START_METHOD", "thread")

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    ExponentialLR,
    _LRScheduler,
)
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from wandb.sdk.lib.runid import generate_id

from third_party.cellvit.config.config import BACKBONE_EMBED_DIM
from third_party.cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
from third_party.cellvit.models.classifier.linear_classifier import LinearClassifier
from third_party.cellvit.training.base_ml.base_experiment import BaseExperiment
from third_party.cellvit.training.base_ml.base_loss import retrieve_loss_fn
from celluda.dataset import (
    DetectionDataset as DetectionDatasetUDA,
)
from celluda.trainer import (
    CellViTHeadTrainer as CellViTHeadTrainerUDA,
)

from third_party.cellvit.utils.tools import close_logger, unflatten_dict


class ExperimentCellVitClassifier(BaseExperiment):
    """CellVit Experiment Class"""

    @staticmethod
    def seed_run(seed: int) -> None:
        """Seed the experiment"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        random.seed(seed)

    def run_experiment(self, mode: str = "train") -> Path:
        """Main experiment code.

        Args:
            mode: ``"train"`` (default) trains the classifier head (source-only when
                ``uda.enable`` is false, pseudo-label UDA otherwise) and then evaluates
                on the target validation split; ``"extract"`` only runs CellViT
                segmentation + token extraction and writes the HDF5 caches;
                ``"evaluate"`` loads ``classifier_path`` and runs the target-domain
                evaluation only.
        """
        if mode not in {"train", "extract", "evaluate"}:
            raise ValueError(f"Unknown mode '{mode}'. Use train, extract or evaluate.")
        self.close_remaining_logger()
        self.seed_run(self.default_conf["random_seed"])
        self.run_conf = copy.deepcopy(self.default_conf)
        self.run_name = (
            f"{datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S')}_"
            f"{self.run_conf['logging']['log_comment']}"
        )
        wandb_run_id = generate_id()
        resume = None
        if self.checkpoint is not None:
            wandb_run_id = self.checkpoint["wandb_id"]
            resume = "must"
            self.run_name = self.checkpoint["run_name"]

        wandb_dir_path = Path(self.run_conf["logging"]["wandb_dir"]).resolve()
        wandb_dir_path.mkdir(parents=True, exist_ok=True)
        (wandb_dir_path / "wandb").mkdir(parents=True, exist_ok=True)

        wandb_settings = wandb.Settings(start_method="thread", _disable_service=True)
        run = wandb.init(
            project=self.run_conf["logging"]["project"],
            tags=self.run_conf["logging"].get("tags", []),
            name=self.run_name,
            notes=self.run_conf["logging"]["notes"],
            dir=self.run_conf["logging"]["wandb_dir"],
            mode=self.run_conf["logging"]["mode"].lower(),
            group=self.run_conf["logging"].get("group", str(uuid.uuid4())),
            allow_val_change=True,
            id=wandb_run_id,
            resume=resume,
            settings=wandb_settings,
        )

        self.run_conf["logging"]["run_id"] = run.id
        self.run_conf["logging"]["wandb_file"] = run.id

        self.run_conf["logging"]["log_dir"] = str(
            Path(self.default_conf["logging"]["log_dir"]) / self.run_name
        )

        wandb.config.update(self.run_conf, allow_val_change=True)
        self.create_output_dir(self.run_conf["logging"]["log_dir"])
        self.logger = self.instantiate_logger()
        self.logger.info("Instantiated Logger. WandB init and config update finished.")
        self.logger.info(f"Run is stored here: {self.run_conf['logging']['log_dir']}")
        self.store_config()
        self.logger.info(
            f"Cuda devices: {[torch.cuda.device(i) for i in range(torch.cuda.device_count())]}"
        )

        device = f"cuda:{self.run_conf['gpu']}"
        self.logger.info(f"Using GPU: {device}")
        self.logger.info(f"Using device: {device}")

        loss_fn = self.get_loss_fn()
        self.logger.info("Loss function:")
        self.logger.info(loss_fn)

        cellvit_model, cellvit_run_conf = self.load_cellvit_model(self.run_conf["cellvit_path"])
        cellvit_model.to(device)

        embed_dim = BACKBONE_EMBED_DIM[cellvit_run_conf["model"]["backbone"]]
        classifier_embed_dim = int(embed_dim)

        model = LinearClassifier(
            embed_dim=classifier_embed_dim,
            hidden_dim=self.run_conf["model"]["hidden_dim"],
            num_classes=self.run_conf["data"]["num_classes"],
            drop_rate=self.run_conf["training"].get("drop_rate", 0.0),
        )
        self.logger.info(f"\n{summary(model, input_size=(1, classifier_embed_dim), device='cpu')}")
        model.to(device)

        classifier_ckpt = self.run_conf.get("classifier_path")
        if mode == "evaluate" and not classifier_ckpt:
            raise ValueError("mode='evaluate' requires `classifier_path` in the config.")

        if classifier_ckpt:
            ckpt = torch.load(classifier_ckpt, map_location=device)
            state = ckpt.get("model_state_dict", ckpt)

            load_info = model.load_state_dict(state, strict=True)
            self.logger.info(f"Loaded classifier weights from {Path(classifier_ckpt).resolve()}")
            self.logger.info(load_info)
        else:
            self.logger.info("No classifier checkpoint provided; training head from scratch.")

        optimizer = self.get_optimizer(
            model,
            self.run_conf["training"]["optimizer"],
            self.run_conf["training"]["optimizer_hyperparameter"],
        )
        scheduler = self.get_scheduler(
            optimizer=optimizer,
            scheduler_type=self.run_conf["training"]["scheduler"]["scheduler_type"],
        )

        (
            source_train_dataset,
            source_val_dataset,
            target_train_dataset,
            target_val_dataset,
        ) = self.get_datasets()

        extraction_batch_size = int(self.run_conf["training"].get("extraction_batch_size", 4))
        extraction_num_workers = int(self.run_conf["training"].get("extraction_num_workers", 0))
        extraction_pin_memory = bool(self.run_conf["training"].get("extraction_pin_memory", False))

        source_train_dataloader = DataLoader(
            source_train_dataset,
            batch_size=extraction_batch_size,
            num_workers=extraction_num_workers,
            shuffle=False,
            pin_memory=extraction_pin_memory,
            worker_init_fn=self.seed_worker,
            collate_fn=source_train_dataset.collate_batch,
        )

        source_val_dataloader = DataLoader(
            source_val_dataset,
            batch_size=extraction_batch_size,
            num_workers=extraction_num_workers,
            shuffle=False,
            pin_memory=extraction_pin_memory,
            worker_init_fn=self.seed_worker,
            collate_fn=source_val_dataset.collate_batch,
        )

        target_val_dataloader = DataLoader(
            target_val_dataset,
            batch_size=extraction_batch_size,
            num_workers=extraction_num_workers,
            shuffle=False,
            pin_memory=extraction_pin_memory,
            worker_init_fn=self.seed_worker,
            collate_fn=target_val_dataset.collate_batch,
        )

        if target_train_dataset is not None:
            target_train_dataloader = DataLoader(
                target_train_dataset,
                batch_size=extraction_batch_size,
                num_workers=extraction_num_workers,
                shuffle=False,
                pin_memory=extraction_pin_memory,
                worker_init_fn=self.seed_worker,
                collate_fn=target_train_dataset.collate_batch,
            )
        else:
            target_train_dataloader = None

        trainer_class = CellViTHeadTrainerUDA
        self.logger.info("Instantiate Trainer: %s", trainer_class.__module__)
        trainer = trainer_class(
            model=model,
            cellvit_model=cellvit_model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=self.logger,
            logdir=self.run_conf["logging"]["log_dir"],
            num_classes=self.run_conf["data"]["num_classes"],
            experiment_config=self.run_conf,
            mixed_precision=self.run_conf["training"].get("mixed_precision"),
        )

        if self.checkpoint is not None:
            self.logger.info("Checkpoint was provided. Restore ...")
            trainer.resume_checkpoint(self.checkpoint, just_load_model=self.just_load_model)

        if mode == "extract":
            self.logger.info("Calling Trainer extract_and_cache")
            trainer.extract_and_cache(
                source_train_dataloader=source_train_dataloader,
                source_val_dataloader=source_val_dataloader,
                target_train_dataloader=target_train_dataloader,
                target_val_dataloader=target_val_dataloader,
                overwrite=bool(self.run_conf["training"].get("overwrite_cache", False)),
            )
        elif mode == "evaluate":
            self.logger.info("Calling Trainer evaluate_target")
            trainer._calculate_hashes()
            trainer.evaluate_target(
                target_val_dataloader, checkpoint_path=classifier_ckpt, step=0
            )
        else:
            self.logger.info("Calling Trainer Fit")
            trainer.fit(
                epochs=self.run_conf["training"]["epochs"],
                source_train_dataloader=source_train_dataloader,
                source_val_dataloader=source_val_dataloader,
                target_train_dataloader=target_train_dataloader,
                target_val_dataloader=target_val_dataloader,
                eval_every=self.run_conf["training"].get("eval_every", 1),
                patience=self.run_conf["training"].get("early_stopping_patience"),
                early_stopping_min_delta=self.run_conf["training"].get("early_stopping_min_delta", 0.01),
                export_name=self.export_name(),
                classifier=self.run_conf.get("classifier_path"),
            )

        close_logger(self.logger)

        return self.run_conf["logging"]["log_dir"]

    def get_loss_fn(
        self,
    ) -> Callable:
        """Return loss function"""
        loss_fn = retrieve_loss_fn("CrossEntropyLoss")
        return loss_fn

    def get_scheduler(self, scheduler_type: str, optimizer: Optimizer) -> _LRScheduler:
        """Get the learning rate scheduler for CellViT"""
        implemented_schedulers = ["constant", "exponential", "cosine"]
        if scheduler_type.lower() not in implemented_schedulers:
            self.logger.warning(
                f"Unknown Scheduler - No scheduler from the list {implemented_schedulers} selected. "
                "Using default scheduling."
            )

        if scheduler_type.lower() == "constant":
            scheduler = ConstantLR(optimizer)

        elif scheduler_type.lower() == "exponential":
            scheduler = ExponentialLR(
                optimizer,
                gamma=self.run_conf["training"]["scheduler"].get("gamma", 0.95),
            )
        elif scheduler_type.lower() == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.run_conf["training"]["epochs"],
                eta_min=self.run_conf["training"]["scheduler"].get("eta_min", 1e-5),
            )
        else:
            scheduler = super().get_scheduler(optimizer)
        return scheduler

    def _optional_filelist(self, data_conf: dict, split_key: str):
        """Path of the split CSV, or None (= every image of the split folder) if it does not exist."""
        path = CellViTHeadTrainerUDA._split_filelist_path(data_conf, split_key)
        if Path(path).exists():
            return path
        self.logger.info(f"No split file {path}: using all images of the {data_conf[split_key]} folder.")
        return None

    def export_name(self) -> str:
        """Stem of exported classifier checkpoints: `training.export_name` if set,
        otherwise `<source dir>_to_<target dir>` (or `<source dir>` without a target root)."""
        explicit = self.run_conf.get("training", {}).get("export_name")
        if explicit:
            return str(explicit)
        data_conf = self.run_conf["data"]
        source = Path(CellViTHeadTrainerUDA._domain_root(data_conf, "source_train_split")).name
        target = Path(CellViTHeadTrainerUDA._domain_root(data_conf, "target_train_split")).name
        return source if source == target else f"{source}_to_{target}"

    def get_datasets(
        self,
    ) -> Tuple[Dataset, Dataset, Optional[Dataset], Dataset]:
        """Retrieve the source (train/val) and target (train/val) detection datasets."""

        data_conf = self.run_conf["data"]
        class_map_conf = data_conf["class_map"]

        train_map = class_map_conf["train_mapping"]
        target_map = class_map_conf["target_eval_mapping"]

        src_label_mapping = {int(k): int(v) for k, v in train_map.items()}
        target_label_mapping = {int(k): int(v) for k, v in target_map.items()}
        uda_conf = self.run_conf.get("uda", {}) or {}
        use_uda = bool(uda_conf.get("enable", False))
        target_eval_mode = str(data_conf.get("evaluation", "metric")).strip().lower()
        if target_eval_mode not in {"metric", "plot"}:
            raise ValueError("data.evaluation must be either 'metric' or 'plot'.")
        target_eval_has_labels = target_eval_mode == "metric"

        source_root = CellViTHeadTrainerUDA._domain_root(data_conf, "source_train_split")
        target_root = CellViTHeadTrainerUDA._domain_root(data_conf, "target_train_split")
        source_train_split = data_conf["source_train_split"]
        source_val_split = data_conf["source_val_split"]
        target_val_split = data_conf["target_val_split"]

        resolved_source_train_filelist = self._optional_filelist(data_conf, "source_train_split")
        resolved_source_val_filelist = self._optional_filelist(data_conf, "source_val_split")
        resolved_target_val_filelist = self._optional_filelist(data_conf, "target_val_split")

        normalize_mean = tuple(data_conf.get("normalize_mean", (0.485, 0.456, 0.406)))
        normalize_std = tuple(data_conf.get("normalize_std", (0.229, 0.224, 0.225)))

        self.logger.info("Caching datasets")

        source_train_dataset = DetectionDatasetUDA(
            dataset_path=source_root,
            split=source_train_split,
            filelist_path=resolved_source_train_filelist,
            label_mapping=src_label_mapping,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )
        source_train_dataset.cache_dataset()

        source_val_dataset = DetectionDatasetUDA(
            dataset_path=source_root,
            split=source_val_split,
            filelist_path=resolved_source_val_filelist,
            label_mapping=src_label_mapping,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )
        source_val_dataset.cache_dataset()

        if use_uda:
            target_train_split = data_conf["target_train_split"]
            resolved_target_train_filelist = self._optional_filelist(data_conf, "target_train_split")
            target_train_dataset = DetectionDatasetUDA(
                dataset_path=target_root,
                split=target_train_split,
                filelist_path=resolved_target_train_filelist,
                label_mapping=None,
                no_label=True,
                normalize_mean=normalize_mean,
                normalize_std=normalize_std,
            )
            target_train_dataset.cache_dataset()
        else:
            target_train_dataset = None

        target_val_dataset = DetectionDatasetUDA(
            dataset_path=target_root,
            split=target_val_split,
            filelist_path=resolved_target_val_filelist,
            label_mapping=target_label_mapping if target_eval_has_labels else None,
            no_label=not target_eval_has_labels,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )
        target_val_dataset.cache_dataset()

        return source_train_dataset, source_val_dataset, target_train_dataset, target_val_dataset

    def load_cellvit_model(self, checkpoint_path: Union[Path, str]) -> Tuple[nn.Module, dict]:
        """Load a pretrained CellViT model"""
        model_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        cellvit_run_conf = unflatten_dict(model_checkpoint["config"], ".")
        model = self._get_cellvit_architecture(
            model_type=model_checkpoint["arch"], model_conf=cellvit_run_conf
        )
        self.logger.info(f"Loading checkpoint {Path(checkpoint_path).resolve()}")
        self.logger.info(model.load_state_dict(model_checkpoint["model_state_dict"]))
        model.eval()
        cellvit_run_conf["model"]["token_patch_size"] = model.patch_size
        return model, cellvit_run_conf

    def _get_cellvit_architecture(
        self,
        model_type: Literal["CellViTSAM"],
        model_conf: dict,
    ) -> CellViTSAM:
        """Build the segmentation model for inference (CellUDA ships the SAM-H backbone only)."""
        if model_type != "CellViTSAM":
            raise NotImplementedError(
                f"Unsupported model type '{model_type}'. CellUDA supports CellViTSAM checkpoints "
                "only (CellViT-SAM-H-x40-AMP.pth from CellViT++)."
            )
        model = CellViTSAM(
            model_path=None,
            num_nuclei_classes=model_conf["data"]["num_nuclei_classes"],
            num_tissue_classes=model_conf["data"]["num_tissue_classes"],
            vit_structure=model_conf["model"]["backbone"],
            regression_loss=model_conf["model"].get("regression_loss", False),
        )
        return model
