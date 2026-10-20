# -*- coding: utf-8 -*-
# CellUDA: embedding-space pseudo-label UDA trainer for the CellViT++ classifier head.
# Modified from CellViT++ (https://github.com/TIO-IKIM/CellViT-plus-plus)
import copy
import hashlib
import itertools
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as tmf
import tqdm
import wandb
from einops import rearrange
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import AUROC, Accuracy, AveragePrecision
from third_party.cellvit.inference.postprocessing_cupy import DetectionCellPostProcessorCupy
from third_party.cellvit.training.base_ml.base_early_stopping import EarlyStopping
from third_party.cellvit.training.base_ml.base_experiment import BaseExperiment
from third_party.cellvit.training.base_ml.base_trainer import BaseTrainer
from third_party.cellvit.training.datasets.base_cell_dataset import BaseCellEmbeddingDataset
from celluda.visualization import (
    CellClassificationVisualizer,
    as_rgb_tuple,
    resolve_overlay_dir,
)
from third_party.cellvit.training.utils.metrics import cell_detection_scores
from third_party.cellvit.training.utils.tools import AverageMeter, pair_coordinates


class _UnlabeledEmbeddingDataset(Dataset):
    """Target embedding dataset: only tokens."""

    def __init__(self, extracted_cells: List[dict]):
        self.cells = extracted_cells

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, idx):
        tok = self.cells[idx]["token"]
        if not torch.is_tensor(tok):
            tok = torch.as_tensor(tok, dtype=torch.float32)
        return tok


class CellViTHeadTrainer(BaseTrainer):
    """Embedding-space pseudo-label UDA trainer."""

    SRC_TRAIN = "source_train"
    SRC_VAL = "source_val"
    TGT_TRAIN = "target_train"
    TGT_VAL = "target_val"

    @staticmethod
    def _domain_root(data_conf: Dict[str, Any], split_key: str) -> str:
        """Dataset root for a split key: `data.dataset_path` for `source_*`,
        `data.target_dataset_path` (falls back to `dataset_path`) for `target_*`."""
        if str(split_key).startswith("target_"):
            return str(data_conf.get("target_dataset_path") or data_conf["dataset_path"])
        return str(data_conf["dataset_path"])

    @staticmethod
    def _split_filelist_path(data_conf: Dict[str, Any], split_key: str) -> str:
        """Path of the split CSV: `<data.{source|target}_splits_dir>/<split>.csv`,
        defaulting to `<dataset root>/splits/<split>.csv`."""
        split_name = str(data_conf[split_key]).strip()
        if not split_name:
            raise ValueError(f"data.{split_key} must be non-empty.")
        role = "target" if str(split_key).startswith("target_") else "source"
        splits_dir = data_conf.get(f"{role}_splits_dir")
        if not splits_dir:
            splits_dir = Path(CellViTHeadTrainer._domain_root(data_conf, split_key)) / "splits"
        return str(Path(splits_dir) / f"{split_name}.csv")

    def __init__(
        self,
        model: nn.Module,
        cellvit_model: nn.Module,
        loss_fn: Callable,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        device: str,
        logger: logging.Logger,
        logdir: Union[Path, str],
        num_classes: int,
        experiment_config: dict,
        early_stopping: EarlyStopping = None,
        mixed_precision: bool = False,
    ):
        super().__init__(
            model=model,
            loss_fn_dict=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger,
            logdir=logdir,
            experiment_config=experiment_config,
            early_stopping=early_stopping,
            mixed_precision=mixed_precision,
        )

        self.loss_fn = loss_fn
        self.num_classes = int(num_classes)

        self.cellvit_model = cellvit_model
        self.cellvit_model.eval()

        data_conf = self.experiment_config.get("data", {})
        self.target_val_overlay_dir: str = data_conf.get("target_val_overlay_dir", "")
        self.target_val_split: str = data_conf.get("target_val_split")
        self.dataset_path: str = data_conf.get("dataset_path", "")
        self.target_dataset_path: str = str(
            data_conf.get("target_dataset_path") or self.dataset_path
        )
        eval_mode_raw = str(data_conf.get("evaluation", "metric")).strip().lower()
        if eval_mode_raw not in {"metric", "plot"}:
            raise ValueError("data.evaluation must be either 'metric' or 'plot'.")
        self.target_eval_mode: str = eval_mode_raw
        self.logger.info(f"Target validation evaluation mode: {self.target_eval_mode}")
        self.visualizer = CellClassificationVisualizer(
            denormalize=bool(data_conf.get("overlay_denormalize", True)),
            norm_mean=as_rgb_tuple(
                data_conf.get("overlay_norm_mean", (0.485, 0.456, 0.406))
            ),
            norm_std=as_rgb_tuple(
                data_conf.get("overlay_norm_std", (0.229, 0.224, 0.225))
            ),
            show_titles=bool(data_conf.get("overlay_show_titles", False)),
        )

        uda = self.experiment_config.get("uda", {})
        if uda is None:
            uda = {}
        if not isinstance(uda, dict):
            raise ValueError("uda must be a mapping if provided.")
        self.use_uda: bool = bool(uda.get("enable", False))
        self.current_epoch: int = 0
        self.total_epochs: int = 1

        self.warmup_epochs: int = 0
        self.lambda_u: float = 0.0
        self.ema_alpha: float = 0.0
        self.pseudo_thresh: float = 0.0
        self.num_tta: int = 1
        self.token_noise_weak_mean: float = 0.0
        self.token_noise_weak_std: float = 0.0
        self.token_noise_strong_mean: float = 0.0
        self.token_noise_strong_std: float = 0.0
        self.token_noise_weak_std_vec: Optional[torch.Tensor] = None
        self.token_noise_strong_std_vec: Optional[torch.Tensor] = None
        self.feature_dropout_p: float = 0.0
        self.teacher_model: Optional[nn.Module] = None

        def _require_uda(name: str):
            if name not in uda:
                raise ValueError(f"uda.{name} is required when uda.enable=true")
            return uda[name]

        if self.use_uda:
            self.warmup_epochs = int(uda.get("warmup_epochs", 0))

            self.lambda_u = float(_require_uda("lambda_u"))
            self.ema_alpha = float(_require_uda("ema_alpha"))
            self.pseudo_thresh = float(_require_uda("pseudo_thresh"))
            self.num_tta = int(_require_uda("num_tta"))
            self.token_noise_weak_mean = float(
                _require_uda("token_noise_weak_mean")
            )
            self.token_noise_weak_std = float(_require_uda("token_noise_weak_std"))
            self.token_noise_strong_mean = float(
                _require_uda("token_noise_strong_mean")
            )
            self.token_noise_strong_std = float(
                _require_uda("token_noise_strong_std")
            )
            self.feature_dropout_p = float(_require_uda("feature_dropout_p"))

            self.teacher_model = copy.deepcopy(self.model).to(self.device)
            self._sync_teacher_from_student()

        self.loss_avg_tracker = AverageMeter("Total_Loss", ":.4f")
        self.sup_loss_tracker = AverageMeter("Sup_Loss", ":.4f")
        self.ce_loss_tracker = AverageMeter("CE_Loss", ":.4f")
        self.unsup_loss_tracker = AverageMeter("Unsup_Loss", ":.4f")
        self.mask_rate_tracker = AverageMeter("Mask_Rate", ":.4f")
        self.weight_rate_tracker = AverageMeter("Weight_Mean", ":.4f")

        if self.num_classes <= 2:
            self.auroc_func = AUROC(task="binary")
            self.acc_func = Accuracy(task="binary")
            self.average_prec_func = AveragePrecision(task="binary")
        else:
            self.auroc_func = AUROC(task="multiclass", num_classes=self.num_classes)
            self.acc_func = Accuracy(task="multiclass", num_classes=self.num_classes)
            self.average_prec_func = AveragePrecision(
                task="multiclass", num_classes=self.num_classes
            )

        self.cached_dataset = {}

        self.train_dataset_hash: str = ""
        self.val_dataset_hash: str = ""
        self.tgt_dataset_hash: str = ""
        self.tgt_val_dataset_hash: str = ""

        self.random_generator = torch.Generator()
        self.random_generator.manual_seed(int(experiment_config.get("random_seed")))

        self.logger.info("Labeled-cell extraction mode: CellViT coordinate matching")
        matching_conf = self.experiment_config.get("data", {}).get("matching", {})
        if matching_conf is None:
            matching_conf = {}
        if not isinstance(matching_conf, dict):
            raise ValueError("data.matching must be a mapping if provided.")
        self.match_distance = int(matching_conf.get("match_distance", 12))
        self.logger.info(
            "Pipeline instance matching: distance_radius=%d",
            self.match_distance,
        )
        self.restrict_val_matching_to_center_patch: bool = bool(
            matching_conf.get("restrict_val_matching_to_center_patch", False)
        )
        self.val_matching_center_patch_size: int = int(
            matching_conf.get("val_matching_center_patch_size", 256)
        )
        if self.val_matching_center_patch_size <= 0:
            raise ValueError(
                "data.matching.val_matching_center_patch_size must be > 0."
            )
        if self.restrict_val_matching_to_center_patch:
            self.logger.info(
                "Validation matching restricted to center %dx%d region.",
                self.val_matching_center_patch_size,
                self.val_matching_center_patch_size,
            )
        class_map = data_conf["class_map"]
        self.final_eval_map = {
            "pred_map": class_map["target_pred_mapping"],
            "gt_map": class_map["target_eval_mapping"],
        }
        self.final_eval_skip_loss = True
        self._final_eval_mode: bool = False
    def _embedding_dataloader(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
    ) -> DataLoader:
        kwargs = {
            "batch_size": int(
                batch_size or self.experiment_config["training"]["batch_size"]
            ),
            "shuffle": bool(shuffle),
            "num_workers": 0,
            "worker_init_fn": BaseExperiment.seed_worker,
        }
        if shuffle:
            kwargs["generator"] = self.random_generator
        return DataLoader(dataset, **kwargs)

    def _paired_domain_batch_count(self, source_count: int, target_count: int) -> int:
        if source_count <= 0 or target_count <= 0:
            raise RuntimeError(
                "Source and target loaders must both contain at least one batch."
            )
        return source_count

    def _paired_embedding_batches(
        self,
        source_dataloader: DataLoader,
        target_dataloader: DataLoader,
    ):
        num_batches = self._paired_domain_batch_count(
            len(source_dataloader), len(target_dataloader)
        )
        tgt_iter = iter(target_dataloader)
        train_loop = tqdm.tqdm(
            itertools.islice(enumerate(source_dataloader), num_batches),
            total=num_batches,
        )
        for batch_idx, src_batch in train_loop:
            try:
                tgt_batch = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(target_dataloader)
                tgt_batch = next(tgt_iter)
            yield batch_idx, src_batch, tgt_batch, num_batches

    def _sync_teacher_from_student(self) -> None:
        """Keep teacher in sync with current student weights."""
        if self.teacher_model is None:
            return
        self.teacher_model.load_state_dict(self.model.state_dict(), strict=True)
        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)

    def resume_checkpoint(self, checkpoint: dict, just_load_model: bool = False):
        """Resume checkpoint while keeping teacher/student aligned."""
        super().resume_checkpoint(checkpoint, just_load_model=just_load_model)
        self._sync_teacher_from_student()

    def _align_final_token_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Check that the token width matches the classifier input (`fc1.in_features`)."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        expected = int(self.model.fc1.in_features)
        if x.shape[-1] != expected:
            raise RuntimeError(
                f"[TokenDim/Final] token_dim mismatch: got {x.shape[-1]}, expected {expected}."
            )
        return x

    @contextmanager
    def _classifier_eval_mode(self):
        was_training = self.model.training
        self.model.eval()
        try:
            yield
        finally:
            if was_training:
                self.model.train()

    @contextmanager
    def _autocast_context(self):
        if self.mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                yield
        else:
            yield

    def _optimizer_step_due(self, batch_idx: int, num_batches: int) -> bool:
        return (
            self.accum_iter == 1
            or (batch_idx + 1) % self.accum_iter == 0
            or (batch_idx + 1) == num_batches
        )

    def _backward_and_step(
        self,
        loss: torch.Tensor,
        batch_idx: int,
        num_batches: int,
        *,
        update_teacher: bool = False,
    ) -> None:
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if not self._optimizer_step_due(batch_idx, num_batches):
            return

        if self.mixed_precision:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
        self.model.zero_grad()
        if update_teacher:
            self._ema_update_teacher()

    def _calculate_hashes(self):
        """Calculate cache hashes using the classifier trainer's hash recipe."""
        conf = self.experiment_config
        data_conf = conf["data"]
        cellvit_path = conf["cellvit_path"]
        normalize_stains_train = data_conf.get("normalize_stains_train", False)
        hash_info = data_conf.get("hash_info")

        def _dataset_hash(filelist: str, dataset_part: str, dataset_path: str) -> str:
            hash_str = (
                f"{dataset_path}_{filelist}_{cellvit_path}_"
                f"stain_{normalize_stains_train}_{dataset_part}"
            )
            if hash_info is not None:
                hash_str = f"{hash_str}_{hash_info}"
            hasher = hashlib.sha256()
            hasher.update(hash_str.encode("utf-8"))
            return hasher.hexdigest()

        source_train_file = self._split_filelist_path(
            data_conf, "source_train_split"
        )
        source_val_file = self._split_filelist_path(data_conf, "source_val_split")
        target_train_file = self._split_filelist_path(
            data_conf, "target_train_split"
        )
        target_val_file = self._split_filelist_path(data_conf, "target_val_split")

        source_root = self._domain_root(data_conf, "source_train_split")
        target_root = self._domain_root(data_conf, "target_train_split")
        self.train_dataset_hash = _dataset_hash(source_train_file, "train", source_root)
        self.val_dataset_hash = _dataset_hash(source_val_file, "val", source_root)
        self.tgt_dataset_hash = _dataset_hash(target_train_file, self.TGT_TRAIN, target_root)
        self.tgt_val_dataset_hash = _dataset_hash(target_val_file, self.TGT_VAL, target_root)

    def _log_token_range(self, name: str, cells: List[dict]) -> None:
        min_v = None
        max_v = None
        for cell in cells:
            tok = cell.get("token", None)
            if tok is None:
                continue
            if not torch.is_tensor(tok):
                tok = torch.as_tensor(tok)
            tmin = tok.min()
            tmax = tok.max()
            min_v = tmin if min_v is None else torch.minimum(min_v, tmin)
            max_v = tmax if max_v is None else torch.maximum(max_v, tmax)

        if min_v is None or max_v is None:
            self.logger.info(f"[{name}] Token range: no tokens")
            return
        self.logger.info(
            f"[{name}] Token range: min={float(min_v):.4f}, max={float(max_v):.4f}"
        )

    def _cache_dir(self) -> Path:
        cache_dir_config = self.experiment_config.get("training", {}).get("cache_dir")
        if cache_dir_config:
            return Path(cache_dir_config).expanduser().resolve()
        dataset_path = Path(self.experiment_config["data"]["dataset_path"])
        return dataset_path / "cache"

    def _cache_path(self, dataset_part: str) -> Path:
        hash_by_part = {
            self.SRC_TRAIN: self.train_dataset_hash,
            self.SRC_VAL: self.val_dataset_hash,
            self.TGT_TRAIN: self.tgt_dataset_hash,
            self.TGT_VAL: self.tgt_val_dataset_hash,
        }
        if dataset_part not in hash_by_part:
            raise NotImplementedError("Unknown set")
        return self._cache_dir() / f"{hash_by_part[dataset_part]}.h5"

    def _test_cache_exists(self, dataset_part: str) -> bool:
        """Test if a classifier-style HDF5 cache exists."""
        cache_path = self._cache_path(dataset_part)
        if cache_path.exists():
            return True
        return False

    @staticmethod
    def _empty_detection_metrics() -> Dict[str, List[float]]:
        return {"f1s": [], "precs": [], "recs": []}

    @staticmethod
    def _summarize_detection_metrics(
        det_metrics: Dict[str, List[float]]
    ) -> Dict[str, float]:
        f1s = np.asarray(det_metrics.get("f1s", []), dtype=np.float32)
        precs = np.asarray(det_metrics.get("precs", []), dtype=np.float32)
        recs = np.asarray(det_metrics.get("recs", []), dtype=np.float32)
        return {
            "F1": float(np.mean(f1s)) if f1s.size > 0 else 0.0,
            "Prec": float(np.mean(precs)) if precs.size > 0 else 0.0,
            "Rec": float(np.mean(recs)) if recs.size > 0 else 0.0,
            "Count": int(f1s.size),
        }

    def _cache_results(self, extracted_cells: List, dataset_part: str) -> None:
        """Cache extracted cells with the classifier trainer's HDF5 schema."""
        cache_dir = self._cache_dir()
        cache_dir.mkdir(exist_ok=True, parents=True)
        cache_path = self._cache_path(dataset_part)
        dataset_hash = cache_path.stem
        self.logger.info(f"Caching dataset {dataset_hash} to disk...")
        training_conf = self.experiment_config.get("training", {})
        cache_compression = training_conf.get("cache_compression", "gzip")
        if cache_compression in (None, "", "none", "None"):
            cache_compression = None
        cache_chunk_size = int(training_conf.get("cache_chunk_size", 8192))
        cache_chunk_len = min(max(cache_chunk_size, 1), max(len(extracted_cells), 1))
        cache_1d_kwargs = {
            "chunks": (cache_chunk_len,),
            "compression": cache_compression,
        }
        cache_2d_kwargs = {
            "chunks": (cache_chunk_len, 2),
            "compression": cache_compression,
        }

        with h5py.File(cache_path, "w") as f:
            images = f.create_dataset(
                "images",
                shape=(len(extracted_cells),),
                dtype=h5py.string_dtype(),
                **cache_1d_kwargs,
            )
            coords = f.create_dataset(
                "coords",
                shape=(len(extracted_cells), 2),
                dtype=h5py.special_dtype(vlen=np.float64),
                **cache_2d_kwargs,
            )
            types = f.create_dataset(
                "types",
                shape=(len(extracted_cells),),
                dtype="i",
                **cache_1d_kwargs,
            )
            tokens = f.create_dataset(
                "tokens",
                shape=(len(extracted_cells),),
                dtype=h5py.special_dtype(vlen=np.float64),
            )
            for i, cell in tqdm.tqdm(
                enumerate(extracted_cells), total=len(extracted_cells)
            ):
                images[i] = cell["image"]
                coords[i] = cell["coords"]
                types[i] = cell["type"]
                tokens[i] = cell["token"].numpy()

    def _load_from_cache(self, dataset_part: str) -> List:
        """Load extracted cells from a classifier-style HDF5 cache."""
        cache_path = self._cache_path(dataset_part)
        extracted_cells = []

        f = h5py.File(cache_path, "r")
        images = f["images"][:]
        coords = f["coords"][:]
        types = f["types"][:]
        tokens = f["tokens"][:]
        f.close()

        for image, coord, cell_type, token in zip(images, coords, types, tokens):
            cell = {
                "image": image.decode("utf-8"),
                "coords": coord[0],
                "type": torch.tensor(cell_type),
                "token": torch.tensor(token).type(torch.float32),
            }
            extracted_cells.append(cell)
        self.logger.info(f"Loaded dataset from cache: {str(cache_path)}")
        return extracted_cells

    @torch.no_grad()
    def _ema_update_teacher(self) -> None:
        if self.teacher_model is None:
            return
        a = self.ema_alpha
        for tp, sp in zip(self.teacher_model.parameters(), self.model.parameters()):
            tp.data.mul_(a).add_(sp.data, alpha=(1.0 - a))

    def _gaussian_noise(self, x: torch.Tensor, std, mean: float = 0.0) -> torch.Tensor:
        std_t = torch.as_tensor(std, device=x.device, dtype=x.dtype).abs()
        return x + torch.randn_like(x) * std_t + mean

    def _feature_dropout(self, x: torch.Tensor, p: float) -> torch.Tensor:
        if p <= 0:
            return x
        if p >= 1:
            return torch.zeros_like(x)
        mask = (torch.rand_like(x) > p).to(x.dtype)
        return x * mask / (1 - p)

    def _weak_aug_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self._gaussian_noise(
            x, self.token_noise_weak_std_vec, mean=self.token_noise_weak_mean
        )

    def _strong_aug_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Sample Gaussian noise or feature dropout."""
        ops = [
            lambda tensor: self._gaussian_noise(
                tensor,
                self.token_noise_strong_std_vec,
                mean=self.token_noise_strong_mean,
            )
        ]
        if self.feature_dropout_p > 0:
            ops.append(
                lambda tensor: self._feature_dropout(
                    tensor, self.feature_dropout_p
                )
            )
        op_index = torch.randint(0, len(ops), (1,), device=x.device).item()
        return ops[op_index](x)

    def _initialize_token_augmentation_stats(self, cells: List[dict]) -> None:
        if (
            self.token_noise_weak_std_vec is not None
            and self.token_noise_strong_std_vec is not None
        ):
            return

        tokens = []
        for cell in cells:
            token = cell.get("token")
            if token is None:
                continue
            tokens.append(token if torch.is_tensor(token) else torch.as_tensor(token))

        if not tokens:
            raise RuntimeError(
                "Pseudo-label training requires target cells with valid tokens."
            )

        token_stack = torch.stack(tokens, dim=0)
        p10 = torch.quantile(token_stack, 0.1, dim=0)
        p90 = torch.quantile(token_stack, 0.9, dim=0)
        range_vec = (p90 - p10) / 2.0
        self.token_noise_weak_std_vec = (range_vec * self.token_noise_weak_std).to(
            self.device
        )
        self.token_noise_strong_std_vec = (range_vec * self.token_noise_strong_std).to(
            self.device
        )

    def fit(
        self,
        epochs: int,
        source_train_dataloader: DataLoader,
        source_val_dataloader: DataLoader,
        eval_every: int = 1,
        target_train_dataloader: DataLoader = None,
        target_val_dataloader: DataLoader = None,
        patience: int = 20,
        early_stopping_min_delta: float = 0.01,
        export_name: str = "",
        classifier: str = "",
    ):
        self.total_epochs = int(epochs)

        self._calculate_hashes()
        self.logger.info(
            f"Starting training, epochs={epochs}, UDA={self.use_uda}"
        )

        patience = int(patience) if patience is not None else 20
        min_delta = float(early_stopping_min_delta)
        best_metric = float("-inf")
        bad_count = 0
        best_epoch = None
        best_ckpt_name = "model_best_retrain.pth" if (not self.use_uda) else None
        dataset_name = str(export_name)
        training_conf = self.experiment_config.get("training", {})
        export_dir_value = training_conf.get("classifier_export_dir")
        export_dir = (
            Path(export_dir_value).expanduser()
            if export_dir_value
            else self.logdir / "checkpoints" / "exported_classifier"
        )
        export_dir = export_dir.resolve()
        export_dir.mkdir(parents=True, exist_ok=True)
        seed = self.experiment_config.get("random_seed")
        seed_suffix = f"_seed{seed}" if seed is not None else ""
        final_ckpt_path = str(Path(export_dir) / f"{dataset_name}{seed_suffix}.pth")
        final_epoch_ran = 0
        ran_any_epoch = False
        for epoch in range(self.start_epoch, epochs):
            ran_any_epoch = True
            final_epoch_ran = epoch
            self.logger.info(f"Epoch: {epoch+1}/{epochs}")
            self._reset_cuda_peak_memory()
            train_scalar_metrics, _ = self.train_epoch(
                epoch,
                source_train_dataloader=source_train_dataloader,
                target_train_dataloader=target_train_dataloader,
            )
            self._log_cuda_peak_memory(prefix="Train")
            if ((epoch + 1) % eval_every) == 0:
                self._reset_cuda_peak_memory()
                val_scalar_metrics, _, _, _ = self.validation_epoch(
                    epoch, source_val_dataloader, dataset_part=self.SRC_VAL
                )
                self._log_cuda_peak_memory(prefix="Val")
                metric_for_es = (
                    train_scalar_metrics.get("UDA/Train/MaskRate")
                    if self.use_uda
                    else -val_scalar_metrics.get("Loss/Validation")
                )
                metric_for_es = float(metric_for_es)
                improved = metric_for_es > (best_metric + min_delta)
                if improved:
                    best_metric = metric_for_es
                    bad_count = 0
                    best_epoch = epoch
                    if best_ckpt_name is not None:
                        self.save_checkpoint(epoch, best_ckpt_name)
                        self.logger.info(
                            f"New best checkpoint - epoch={epoch+1}, metric={best_metric:.6f}"
                        )
                else:
                    bad_count += 1
                    self.logger.info(
                        f"EarlyStop counter: {bad_count}/{patience} "
                        f"(metric={metric_for_es:.6f}, best={best_metric:.6f}, min_delta={min_delta:.6f})"
                    )
                    if bad_count >= patience:
                        if best_ckpt_name is not None:
                            self.logger.info(
                                "Performing early stopping (restore best checkpoint)."
                            )
                        else:
                            self.logger.info(
                                "Performing early stopping (keep last epoch weights)."
                            )
                        break

            self.scheduler.step()

        if best_ckpt_name is not None:
            best_ckpt_path = self.logdir / "checkpoints" / best_ckpt_name
            if best_ckpt_path.exists():
                self.logger.info(
                    f"Using best checkpoint for export: {str(best_ckpt_path)}"
                )
                best_ckpt = torch.load(str(best_ckpt_path), map_location=self.device)
                best_state = best_ckpt.get("model_state_dict", best_ckpt)
                self.model.load_state_dict(best_state, strict=True)
                self.model.to(self.device)
                export_epoch = int(best_ckpt.get("epoch", final_epoch_ran))
                self.save_checkpoint(export_epoch, final_ckpt_path)
            else:
                self.logger.warning(
                    "Best checkpoint not found at %s. "
                    "This can happen when no training epoch/eval produced an improved checkpoint.",
                    str(best_ckpt_path),
                )
                if ran_any_epoch or not classifier:
                    self.logger.warning(
                        "Exporting current model weights to %s instead.",
                        str(final_ckpt_path),
                    )
                    self.save_checkpoint(final_epoch_ran, final_ckpt_path)
                else:
                    self.logger.warning(
                        "No training epoch ran; final evaluation will fall back to classifier_path=%s.",
                        str(classifier),
                    )
        else:
            self.save_checkpoint(final_epoch_ran, final_ckpt_path)

        if target_val_dataloader is not None:
            ckpt_to_load = final_ckpt_path
            if (not Path(ckpt_to_load).exists()) and classifier:
                ckpt_to_load = classifier
            target_eval_step = (
                (best_epoch + 1) if best_epoch is not None else (final_epoch_ran + 1)
            )
            self.evaluate_target(
                target_val_dataloader,
                checkpoint_path=ckpt_to_load,
                step=target_eval_step,
            )

    def evaluate_target(
        self,
        target_val_dataloader: DataLoader,
        checkpoint_path: Optional[str] = None,
        step: int = 0,
    ) -> Dict[str, float]:
        """Final target-domain evaluation (`TargetValFinal`) using the class mappings of the config.

        Loads `checkpoint_path` into the classifier head when given, otherwise evaluates
        the current weights. `fit()` calls this after training; `run.py --mode evaluate`
        calls it directly on an exported head.
        """
        if checkpoint_path is not None:
            self.logger.info(
                f"[TargetValFinal] Loading final checkpoint: {str(checkpoint_path)}"
            )
            ckpt = torch.load(str(checkpoint_path), map_location=self.device)
            state = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state, strict=True)
            self.model.to(self.device)

        self._final_eval_mode = True
        try:
            t_val_scalar, _, _, _ = self.validation_epoch(
                0, target_val_dataloader, dataset_part=self.TGT_VAL
            )
        finally:
            self._final_eval_mode = False
        target_final_scalar = {
            k: v
            for k, v in t_val_scalar.items()
            if not k.startswith("Loss/Validation")
        }
        wandb.log(
            {f"TargetValFinal/{k}": v for k, v in target_final_scalar.items()},
            step=int(step),
        )
        tval_lines = [
            f"  {k.replace('/Validation', '')}={v:.4f}"
            for k, v in target_final_scalar.items()
            if isinstance(v, (float, int))
        ]
        self.logger.info("[TargetValFinal]\n" + "\n".join(tval_lines))
        return target_final_scalar

    @torch.no_grad()
    def extract_and_cache(
        self,
        source_train_dataloader: DataLoader,
        source_val_dataloader: DataLoader,
        target_train_dataloader: Optional[DataLoader] = None,
        target_val_dataloader: Optional[DataLoader] = None,
        overwrite: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """Run CellViT segmentation + cell-token extraction for every split and write the HDF5 caches.

        Uses the same per-batch routines as `train_epoch`/`validation_epoch`
        (`get_cellvit_result_labeled` for labeled splits, `get_cellvit_result_unlabeled`
        for the unlabeled target-train split) and the same cache paths, so later
        training/evaluation runs load these caches instead of re-extracting.
        Returns the detection scores measured during extraction, per split.
        """
        self._calculate_hashes()
        postprocessor = self._make_postprocessor()
        summary: Dict[str, Dict[str, float]] = {}
        labeled = [
            (self.SRC_TRAIN, source_train_dataloader),
            (self.SRC_VAL, source_val_dataloader),
        ]
        if target_val_dataloader is not None and self.target_eval_mode == "metric":
            labeled.append((self.TGT_VAL, target_val_dataloader))

        for dataset_part, dataloader in labeled:
            if self._test_cache_exists(dataset_part) and not overwrite:
                self.logger.info(
                    f"[{dataset_part}] cache exists, skipping: {self._cache_path(dataset_part)}"
                )
                continue
            self.logger.info(f"Extracting cells [{dataset_part}]")
            extracted_cells: List[dict] = []
            f1s, precs, recs = [], [], []
            for _, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
                images, cell_gt_batch, types_batch, image_names, offsets = batch
                cells, bf1, bprec, brec = self.get_cellvit_result_labeled(
                    images=images,
                    cell_gt_batch=cell_gt_batch,
                    types_batch=types_batch,
                    image_names=image_names,
                    postprocessor=postprocessor,
                    offsets=offsets,
                    dataset_part=dataset_part,
                )
                extracted_cells += cells
                f1s += bf1
                precs += bprec
                recs += brec
            det = self._summarize_detection_metrics(
                {"f1s": f1s, "precs": precs, "recs": recs}
            )
            det["Cells"] = len(extracted_cells)
            summary[dataset_part] = det
            self.logger.info(
                f"[{dataset_part}] extraction det-metrics - F1={det['F1']:.3f}, "
                f"Prec={det['Prec']:.3f}, Rec={det['Rec']:.3f}, Cells={len(extracted_cells)}"
            )
            self._cache_results(extracted_cells, dataset_part)

        if self.use_uda and target_train_dataloader is not None:
            if self._test_cache_exists(self.TGT_TRAIN) and not overwrite:
                self.logger.info(
                    f"[{self.TGT_TRAIN}] cache exists, skipping: {self._cache_path(self.TGT_TRAIN)}"
                )
            else:
                self.logger.info("Extracting TARGET training cells (unlabeled)")
                extracted_cells_tgt: List[dict] = []
                for _, batch in tqdm.tqdm(
                    enumerate(target_train_dataloader), total=len(target_train_dataloader)
                ):
                    images, _, _, image_names, offsets = batch
                    extracted_cells_tgt += self.get_cellvit_result_unlabeled(
                        images=images,
                        image_names=image_names,
                        postprocessor=postprocessor,
                        offsets=offsets,
                        pred_class_map=self.final_eval_map.get("pred_map"),
                    )
                self.logger.info(f"[TARGET] extracted cells={len(extracted_cells_tgt)}")
                summary[self.TGT_TRAIN] = {"Cells": len(extracted_cells_tgt)}
                self._cache_results(extracted_cells_tgt, self.TGT_TRAIN)
        return summary

    def _make_postprocessor(self) -> DetectionCellPostProcessorCupy:
        nr_types = getattr(self.cellvit_model, "num_nuclei_classes", None)
        if nr_types is None:
            nr_types = int(
                self.experiment_config.get("data", {}).get("num_nuclei_classes", 6)
            )
        return DetectionCellPostProcessorCupy(wsi=None, nr_types=int(nr_types))

    def apply_softmax_reorder(self, predictions: dict) -> dict:
        predictions["nuclei_binary_map"] = F.softmax(
            predictions["nuclei_binary_map"], dim=1
        )
        predictions["nuclei_type_map"] = F.softmax(
            predictions["nuclei_type_map"], dim=1
        )
        predictions["nuclei_type_map"] = predictions["nuclei_type_map"].permute(
            0, 2, 3, 1
        )
        predictions["nuclei_binary_map"] = predictions["nuclei_binary_map"].permute(
            0, 2, 3, 1
        )
        predictions["hv_map"] = predictions["hv_map"].permute(0, 2, 3, 1)
        return predictions

    def extract_tokens(
        self,
        cell_pred_dict: dict,
        predictions: dict,
        image_size: List[int],
    ) -> List[List[torch.Tensor]]:
        patch_size = int(getattr(self.cellvit_model, "patch_size", 16))
        batch_tokens = []
        for patch_idx, patch_cell_pred_dict in enumerate(cell_pred_dict):
            extracted_cell_tokens = []
            patch_tokens = predictions["tokens"][patch_idx]
            for cell in patch_cell_pred_dict.values():
                bbox = cell["bbox"]
                bb_index = bbox / patch_size
                bb_index[0, :] = np.floor(bb_index[0, :])
                bb_index[1, :] = np.ceil(bb_index[1, :])
                bb_index = bb_index.astype(np.uint8)
                cell_token = patch_tokens[
                    :,
                    bb_index[0, 0] : bb_index[1, 0],
                    bb_index[0, 1] : bb_index[1, 1],
                ]
                cell_token = torch.mean(
                    rearrange(cell_token, "D H W -> (H W) D"), dim=0
                )
                extracted_cell_tokens.append(cell_token.detach().cpu())
            batch_tokens.append(extracted_cell_tokens)

        return batch_tokens

    def _drop_ignored_cells(self, cells: List[dict], context: str) -> List[dict]:
        if len(cells) == 0:
            return cells

        kept: List[dict] = []
        dropped = 0
        for cell in cells:
            if "type" not in cell:
                raise KeyError(f"[{context}] cell record missing 'type'.")
            cell_type = int(cell["type"])
            if cell_type < 0:
                dropped += 1
                continue
            kept.append(cell)

        if dropped > 0:
            self.logger.info(
                f"[{context}] dropped {dropped} cells with type < 0 (ignore labels)."
            )
        return kept

    def _should_restrict_matching_to_center_patch(
        self, dataset_part: Optional[str]
    ) -> bool:
        return bool(
            self.restrict_val_matching_to_center_patch
            and dataset_part in {self.SRC_VAL, self.TGT_VAL, "val", "target_val"}
        )

    @staticmethod
    def _points_in_bbox(
        points: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> np.ndarray:
        if points.size == 0:
            return np.zeros((0,), dtype=bool)
        x_min, y_min, x_max, y_max = bbox
        return (
            (points[:, 0] >= x_min)
            & (points[:, 0] < x_max)
            & (points[:, 1] >= y_min)
            & (points[:, 1] < y_max)
        )

    def _center_patch_bbox(
        self,
        image_shape: Tuple[int, int],
        offset: Tuple[int, int],
    ) -> Tuple[float, float, float, float]:
        height, width = int(image_shape[0]), int(image_shape[1])
        size = min(int(self.val_matching_center_patch_size), height, width)
        x0 = float((width - size) / 2.0)
        y0 = float((height - size) / 2.0)
        ox, oy = offset if offset is not None else (0, 0)
        return (
            x0 + float(ox),
            y0 + float(oy),
            x0 + float(size) + float(ox),
            y0 + float(size) + float(oy),
        )

    def _restrict_matching_arrays_to_bbox(
        self,
        pred_centroids: np.ndarray,
        true_centroids: np.ndarray,
        cell_types: np.ndarray,
        patch_token: torch.Tensor,
        bbox: Tuple[float, float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
        n_true = min(int(len(true_centroids)), int(len(cell_types)))
        true_centroids = true_centroids[:n_true]
        cell_types = cell_types[:n_true]

        gt_keep = self._points_in_bbox(true_centroids, bbox)
        pred_keep = self._points_in_bbox(pred_centroids, bbox)
        pred_indices = np.nonzero(pred_keep)[0].astype(np.int64)
        if len(pred_indices) == 0:
            filtered_tokens = patch_token[:0]
        elif torch.is_tensor(patch_token):
            filtered_tokens = patch_token[pred_indices.tolist()]
        else:
            filtered_tokens = [patch_token[int(idx)] for idx in pred_indices.tolist()]

        return (
            pred_centroids[pred_keep],
            true_centroids[gt_keep],
            cell_types[gt_keep],
            filtered_tokens,
        )

    def get_cellvit_result_labeled(
        self,
        images: torch.Tensor,
        cell_gt_batch: List,
        types_batch: List,
        image_names: List,
        postprocessor: DetectionCellPostProcessorCupy,
        offsets: Optional[List[Tuple[int, int]]] = None,
        dataset_part: Optional[str] = None,
    ) -> Tuple[List[dict], List[float], List[float], List[float]]:
        extracted_cells: List[dict] = []
        f1s, precs, recs = [], [], []

        if offsets is None:
            offsets = [(0, 0)] * len(images)

        images = images.to(self.device)
        with self._autocast_context():
            predictions = self.cellvit_model.forward(images, retrieve_tokens=True)

        predictions = self.apply_softmax_reorder(predictions)
        instance_map, cell_pred_dict = postprocessor.post_process_batch(predictions)
        tokens = self.extract_tokens(
            cell_pred_dict,
            predictions,
            list(images.shape[2:]),
        )
        with self._classifier_eval_mode():
            distance_matching_used = 0
            for (
                _,
                pred_dict,
                true_centroids,
                cell_types,
                patch_token,
                image_name,
                offset,
            ) in zip(
                instance_map,
                cell_pred_dict,
                cell_gt_batch,
                types_batch,
                tokens,
                image_names,
                offsets,
            ):
                pred_centroids = np.array(
                    [v["centroid"] for v in pred_dict.values()], dtype=np.float32
                )
                true_centroids = np.array(true_centroids, dtype=np.float32)

                ox, oy = offset if offset is not None else (0, 0)
                if pred_centroids.size > 0:
                    pred_centroids = pred_centroids + np.array(
                        [ox, oy], dtype=np.float32
                    )
                if true_centroids.size > 0:
                    true_centroids = true_centroids + np.array(
                        [ox, oy], dtype=np.float32
                    )

                cell_types_arr = np.asarray(cell_types, dtype=np.int64).reshape(-1)
                if self._should_restrict_matching_to_center_patch(dataset_part):
                    center_bbox = self._center_patch_bbox(
                        image_shape=tuple(images.shape[2:]),
                        offset=(ox, oy),
                    )
                    (
                        pred_centroids,
                        true_centroids,
                        cell_types_arr,
                        patch_token,
                    ) = self._restrict_matching_arrays_to_bbox(
                        pred_centroids=pred_centroids,
                        true_centroids=true_centroids,
                        cell_types=cell_types_arr,
                        patch_token=patch_token,
                        bbox=center_bbox,
                    )

                if len(true_centroids) > 0 and len(pred_centroids) > 0:
                    paired, unpaired_true, unpaired_pred = pair_coordinates(
                        true_centroids, pred_centroids, self.match_distance
                    )
                    distance_matching_used += 1

                    for pair in paired:
                        gt_type = int(cell_types_arr[pair[0]])
                        if gt_type < 0:
                            continue
                        extracted_cells.append(
                            {
                                "image": image_name,
                                "coords": pred_centroids[pair[1]],
                                "type": gt_type,
                                "token": patch_token[pair[1]],
                            }
                        )

                    f1_d, prec_d, rec_d = cell_detection_scores(
                        paired_true=paired[:, 0],
                        paired_pred=paired[:, 1],
                        unpaired_true=unpaired_true,
                        unpaired_pred=unpaired_pred,
                    )
                    f1s.append(f1_d)
                    precs.append(prec_d)
                    recs.append(rec_d)
            if distance_matching_used > 0:
                self.logger.info(
                    "[%s] pipeline matching: distance_patches=%d, radius=%d",
                    str(dataset_part),
                    int(distance_matching_used),
                    int(self.match_distance),
                )
        return extracted_cells, f1s, precs, recs

    def get_cellvit_result_unlabeled(
        self,
        images: torch.Tensor,
        image_names: List,
        postprocessor: DetectionCellPostProcessorCupy,
        offsets: Optional[List[Tuple[int, int]]] = None,
        overlay_dir: Optional[Path] = None,
        pred_class_map: Optional[dict] = None,
    ) -> List[dict]:
        extracted_cells: List[dict] = []
        vis_dir: Optional[Path] = None
        if overlay_dir is not None:
            vis_dir = Path(str(overlay_dir) + "_vis")
            vis_dir.mkdir(parents=True, exist_ok=True)
        if offsets is None:
            offsets = [(0, 0)] * len(images)

        images = images.to(self.device)
        with self._autocast_context():
            predictions = self.cellvit_model.forward(images, retrieve_tokens=True)

        predictions = self.apply_softmax_reorder(predictions)
        instance_map, cell_pred_dict = postprocessor.post_process_batch(predictions)
        tokens = self.extract_tokens(
            cell_pred_dict,
            predictions,
            list(images.shape[2:]),
        )
        with self._classifier_eval_mode():
            if overlay_dir is not None:
                for b_idx in range(instance_map.shape[0]):
                    inst = instance_map[b_idx].cpu().numpy().astype(np.int32)
                    type_map = self.visualizer.build_type_map(
                        inst_map=inst,
                        cell_predictions=cell_pred_dict[b_idx],
                        patch_tokens=tokens[b_idx],
                        model=self.model,
                        device=self.device,
                        align_tokens=self._align_final_token_dim,
                        pred_class_map=pred_class_map,
                    )
                    self.visualizer.save_segmentation_maps(
                        inst, type_map, overlay_dir, image_names[b_idx]
                    )
                    if vis_dir is not None:
                        out_png = vis_dir / f"{image_names[b_idx]}_overlay.png"
                        self.visualizer.save_type_overlay(
                            images[b_idx], type_map, out_png, inst_map=inst
                        )

            for pred_dict, patch_token, image_name, offset in zip(
                cell_pred_dict, tokens, image_names, offsets
            ):
                pred_centroids = np.array(
                    [v["centroid"] for v in pred_dict.values()],
                    dtype=np.float32,
                )
                ox, oy = offset if offset is not None else (0, 0)
                if pred_centroids.size > 0:
                    pred_centroids = pred_centroids + np.array(
                        [ox, oy], dtype=np.float32
                    )

                for i, _ in enumerate(pred_dict.values()):
                    if i >= len(patch_token):
                        break
                    extracted_cells.append(
                        {
                            "image": image_name,
                            "coords": pred_centroids[i]
                            if pred_centroids.size > 0
                            else np.array([0.0, 0.0], dtype=np.float32),
                            "type": -1,
                            "token": patch_token[i],
                        }
                    )
        return extracted_cells

    def train_epoch(
        self,
        epoch: int,
        source_train_dataloader: DataLoader,
        target_train_dataloader: DataLoader = None,
    ) -> Tuple[dict, dict]:
        self.current_epoch = int(epoch)
        da_start_time: Optional[float] = None
        postprocessor = self._make_postprocessor()

        src_cache_key = "Train-Cells"
        if epoch == 0:
            if self._test_cache_exists(self.SRC_TRAIN):
                extracted_cells_src = self._load_from_cache(self.SRC_TRAIN)
            else:
                self.logger.info("Extracting SOURCE training cells")
                extracted_cells_src: List[dict] = []
                f1s, precs, recs = [], [], []
                with torch.no_grad():
                    for _, batch in tqdm.tqdm(
                        enumerate(source_train_dataloader),
                        total=len(source_train_dataloader),
                    ):
                        images, cell_gt_batch, types_batch, image_names, offsets = batch
                        cells, bf1, bprec, brec = self.get_cellvit_result_labeled(
                            images=images,
                            cell_gt_batch=cell_gt_batch,
                            types_batch=types_batch,
                            image_names=image_names,
                            postprocessor=postprocessor,
                            offsets=offsets,
                            dataset_part=self.SRC_TRAIN,
                        )
                        extracted_cells_src += cells
                        f1s += bf1
                        precs += bprec
                        recs += brec
                if len(f1s) > 0:
                    self.logger.info(
                        f"[SOURCE] extraction det-metrics - F1={np.mean(np.array(f1s)):.3f}, "
                        f"Prec={np.mean(np.array(precs)):.3f}, Rec={np.mean(np.array(recs)):.3f}, "
                        f"Cells={len(extracted_cells_src)}"
                    )
                self._cache_results(extracted_cells_src, self.SRC_TRAIN)
            self.cached_dataset[src_cache_key] = extracted_cells_src
        else:
            extracted_cells_src = self.cached_dataset[src_cache_key]

        extracted_cells_src = self._drop_ignored_cells(
            extracted_cells_src,
            context=self.SRC_TRAIN,
        )
        if epoch == 0:
            self._log_token_range("SourceTrain", extracted_cells_src)

        tgt_cache_key = "TargetTrain-Cells"
        extracted_cells_tgt: List[dict] = []

        if self.use_uda and target_train_dataloader is not None:
            if epoch == 0:
                if self._test_cache_exists(self.TGT_TRAIN):
                    extracted_cells_tgt = self._load_from_cache(self.TGT_TRAIN)
                else:
                    self.logger.info("Extracting TARGET training cells (unlabeled)")
                    with torch.no_grad():
                        for _, batch in tqdm.tqdm(
                            enumerate(target_train_dataloader),
                            total=len(target_train_dataloader),
                        ):
                            images, _, _, image_names, offsets = batch
                            extracted_cells_tgt += self.get_cellvit_result_unlabeled(
                                images=images,
                                image_names=image_names,
                                postprocessor=postprocessor,
                                offsets=offsets,
                                pred_class_map=self.final_eval_map.get("pred_map"),
                            )
                    self.logger.info(
                        f"[TARGET] extracted cells={len(extracted_cells_tgt)}"
                    )
                    self._cache_results(extracted_cells_tgt, self.TGT_TRAIN)
                self.cached_dataset[tgt_cache_key] = extracted_cells_tgt
            else:
                extracted_cells_tgt = self.cached_dataset[tgt_cache_key]

        if self.use_uda and da_start_time is None:
            self._reset_cuda_peak_memory()
            da_start_time = time.perf_counter()

        if (
            self.use_uda
            and target_train_dataloader is not None
            and not extracted_cells_tgt
        ):
            raise RuntimeError("[target_train] No cells extracted for UDA training.")

        if epoch == 0 and self.use_uda:
            self._log_token_range("TargetTrain", extracted_cells_tgt)

        if self.use_uda:
            self._initialize_token_augmentation_stats(extracted_cells_tgt)

        src_embed_ds = BaseCellEmbeddingDataset(extracted_cells_src)
        src_embed_dl = self._embedding_dataloader(src_embed_ds, shuffle=True)

        tgt_embed_dl = None
        if self.use_uda and target_train_dataloader is not None:
            tgt_embed_ds = _UnlabeledEmbeddingDataset(extracted_cells_tgt)
            tgt_embed_dl = self._embedding_dataloader(tgt_embed_ds, shuffle=True)

        self.model.train()
        if self.teacher_model is not None:
            self.teacher_model.eval()

        self.loss_avg_tracker.reset()
        self.sup_loss_tracker.reset()
        self.ce_loss_tracker.reset()
        self.unsup_loss_tracker.reset()
        self.mask_rate_tracker.reset()
        self.weight_rate_tracker.reset()

        predictions, probabilities, gt = [], [], []

        if tgt_embed_dl is None:
            train_loop = tqdm.tqdm(enumerate(src_embed_dl), total=len(src_embed_dl))
            for batch_idx, src_batch in train_loop:
                out = self.train_step(src_batch, batch_idx, len(src_embed_dl))
                predictions.append(out["predictions"])
                probabilities.append(out["probabilities"])
                gt.append(out["gt"])
        else:
            for batch_idx, src_batch, tgt_batch, num_batches in self._paired_embedding_batches(
                src_embed_dl, tgt_embed_dl
            ):
                out = self.train_step_uda(
                    src_batch, tgt_batch, batch_idx, num_batches
                )

                predictions.append(out["predictions"])
                probabilities.append(out["probabilities"])
                gt.append(out["gt"])

        predictions = torch.cat(predictions, dim=0).detach().cpu()
        probabilities = torch.cat(probabilities, dim=0).detach().cpu()
        gt = torch.cat(gt, dim=0).detach().cpu()

        acc_score = np.float32(self.acc_func(predictions, gt))

        if self.num_classes <= 2:
            auroc_score = np.float32(self.auroc_func(probabilities[:, 1], gt))
            ap_score = np.float32(self.average_prec_func(probabilities[:, 1], gt))
            f1_per_class = tmf.f1_score(
                predictions, gt, task="binary", average=None
            ).detach()
            precision_per_class = tmf.precision(
                predictions, gt, task="binary", average=None
            ).detach()
            recall_per_class = tmf.recall(
                predictions, gt, task="binary", average=None
            ).detach()
        else:
            auroc_score = np.float32(self.auroc_func(probabilities, gt))
            ap_score = np.float32(self.average_prec_func(probabilities, gt))
            f1_per_class = tmf.f1_score(
                predictions,
                gt,
                task="multiclass",
                num_classes=self.num_classes,
                average=None,
            ).detach()
            precision_per_class = tmf.precision(
                predictions,
                gt,
                task="multiclass",
                num_classes=self.num_classes,
                average=None,
            ).detach()
            recall_per_class = tmf.recall(
                predictions,
                gt,
                task="multiclass",
                num_classes=self.num_classes,
                average=None,
            ).detach()

        macro_f1 = float(torch.mean(f1_per_class).cpu().item())
        macro_precision = float(torch.mean(precision_per_class).cpu().item())
        macro_recall = float(torch.mean(recall_per_class).cpu().item())

        scalar_metrics = {
            "Loss/Train": float(self.loss_avg_tracker.avg),
            "Loss/Train/Sup": float(self.sup_loss_tracker.avg),
            "Loss/Train/CE": float(self.ce_loss_tracker.avg),
            "Loss/Train/Unsup": float(self.unsup_loss_tracker.avg),
            "UDA/Train/MaskRate": float(self.mask_rate_tracker.avg),
            "UDA/Train/WeightMean": float(self.weight_rate_tracker.avg),
            "F1-Score/Train/Macro": macro_f1,
            "Accuracy/Train": acc_score,
            "AUROC/Train": auroc_score,
            "Average-Precision/Train": ap_score,
            "Precision/Train/Macro": macro_precision,
            "Recall/Train/Macro": macro_recall,
        }
        for idx, (f1c, pc, rc) in enumerate(
            zip(f1_per_class, precision_per_class, recall_per_class)
        ):
            scalar_metrics[f"F1-Score/Train/Class_{idx}"] = float(f1c)
            scalar_metrics[f"Precision/Train/Class_{idx}"] = float(pc)
            scalar_metrics[f"Recall/Train/Class_{idx}"] = float(rc)

        da_seconds = (
            max(0.0, time.perf_counter() - da_start_time)
            if self.use_uda and da_start_time is not None
            else 0.0
        )
        scalar_metrics["UDA/DomainAdaptationTimeSeconds"] = float(da_seconds)
        if self.use_uda:
            self.logger.info(
                f"[UDAStats] DomainAdaptationTime seconds={da_seconds:.3f}"
            )
            if torch.cuda.is_available():
                alloc_mb = torch.cuda.max_memory_allocated(self.device) / (1024.0**2)
                reserved_mb = torch.cuda.max_memory_reserved(self.device) / (
                    1024.0**2
                )
                self.logger.info(
                    "[UDAStats] DomainAdaptationPeakMemory "
                    f"alloc_mb={alloc_mb:.1f} reserved_mb={reserved_mb:.1f}"
                )

        self.logger.info(
            f"Train - Loss:{self.loss_avg_tracker.avg:.4f} "
            f"Sup:{self.sup_loss_tracker.avg:.4f} CE:{self.ce_loss_tracker.avg:.4f} "
            f"Unsup:{self.unsup_loss_tracker.avg:.4f} "
            f"Mask:{self.mask_rate_tracker.avg:.4f} Weight:{self.weight_rate_tracker.avg:.4f} "
            f"F1:{macro_f1:.4f} Acc:{acc_score:.4f} AUROC:{auroc_score:.4f} AP:{ap_score:.4f}"
        )

        return scalar_metrics, None

    def train_step(
        self, batch: Any, batch_idx: int, num_batches: int
    ) -> Dict[str, torch.Tensor]:
        cell_tokens = self._align_final_token_dim(batch[0].to(self.device))
        cell_types = batch[2].to(self.device)
        if hasattr(self.loss_fn, "to"):
            self.loss_fn.to(self.device)

        with self._autocast_context():
            logits = self.model.forward(cell_tokens)
            loss_sup = self.loss_fn(logits, cell_types)

        self._backward_and_step(
            loss_sup,
            batch_idx,
            num_batches,
            update_teacher=self.teacher_model is not None,
        )

        zero = torch.tensor(0.0, device=self.device)
        self.loss_avg_tracker.update(loss_sup.detach())
        self.sup_loss_tracker.update(loss_sup.detach())
        self.ce_loss_tracker.update(loss_sup.detach())
        self.unsup_loss_tracker.update(zero)
        self.mask_rate_tracker.update(zero)
        self.weight_rate_tracker.update(zero)

        probs = F.softmax(logits, dim=1)
        return {
            "gt": cell_types,
            "probabilities": probs,
            "predictions": probs.argmax(dim=1),
        }

    @torch.no_grad()
    def _build_pseudo_targets(
        self, x_u: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teacher pseudo-labels, confidence mask and TTA-agreement weight for a target batch."""
        if self.teacher_model is None:
            raise RuntimeError(
                "Pseudo-label training requires an initialized teacher model."
            )

        num_tta = int(self.num_tta) if int(self.num_tta) > 0 else 5

        teacher_probs = F.softmax(self.teacher_model.forward(x_u), dim=1)
        confidence, pseudo = teacher_probs.max(dim=1)
        tta_predictions = []
        for _ in range(num_tta):
            teacher_input = self._weak_aug_tokens(x_u)
            augmented_probs = F.softmax(
                self.teacher_model.forward(teacher_input), dim=1
            )
            tta_predictions.append(augmented_probs.argmax(dim=1))

        prediction_stack = torch.stack(tta_predictions, dim=0)
        weight = (prediction_stack == pseudo.unsqueeze(0)).float().mean(dim=0)
        weight = torch.where(weight < 0.5, torch.zeros_like(weight), weight)

        mask = confidence.ge(float(self.pseudo_thresh))
        return pseudo, mask, weight

    def train_step_uda(
        self,
        src_batch: Any,
        tgt_batch: Any,
        batch_idx: int,
        num_batches: int,
    ) -> Dict[str, torch.Tensor]:
        x_s = self._align_final_token_dim(src_batch[0].to(self.device))
        y_s = src_batch[2].to(self.device)

        x_u = self._align_final_token_dim(tgt_batch.to(self.device))
        x_us = self._strong_aug_tokens(x_u)

        if hasattr(self.loss_fn, "to"):
            self.loss_fn.to(self.device)

        in_warmup = self.warmup_epochs > 0 and self.current_epoch < self.warmup_epochs
        lambda_u_eff = 0.0 if in_warmup else float(self.lambda_u)

        with self._autocast_context():
            logits_s = self.model.forward(x_s)
            loss_sup = self.loss_fn(logits_s, y_s)
            pseudo, mask, weight = self._build_pseudo_targets(x_u)
            logits_u = self.model.forward(x_us)
            if mask.any():
                per_sample_loss = F.cross_entropy(logits_u, pseudo, reduction="none")
                sample_weight = (weight * mask.float()).detach()
                loss_unsup = (per_sample_loss * sample_weight).sum() / (
                    sample_weight.sum() + 1e-6
                )
            else:
                loss_unsup = logits_u.sum() * 0.0
            total_loss = loss_sup + lambda_u_eff * loss_unsup

        self._backward_and_step(
            total_loss,
            batch_idx,
            num_batches,
            update_teacher=True,
        )

        zero = torch.tensor(0.0, device=self.device)
        mask_rate = mask.float().mean().detach() if mask.numel() > 0 else zero
        weight_mean = weight.mean().detach() if weight.numel() > 0 else zero
        self.loss_avg_tracker.update(total_loss.detach())
        self.sup_loss_tracker.update(loss_sup.detach())
        self.ce_loss_tracker.update(loss_sup.detach())
        self.unsup_loss_tracker.update(loss_unsup.detach())
        self.mask_rate_tracker.update(mask_rate)
        self.weight_rate_tracker.update(weight_mean)

        probs_s = F.softmax(logits_s, dim=1)
        return {
            "gt": y_s,
            "probabilities": probs_s,
            "predictions": probs_s.argmax(dim=1),
        }

    def validation_epoch(
        self, epoch: int, val_dataloader: DataLoader, dataset_part: str = "source_val"
    ):
        postprocessor = self._make_postprocessor()
        is_target_unlabeled_eval_mode = (
            dataset_part == self.TGT_VAL and self.target_eval_mode == "plot"
        )
        overlay_dir = (
            resolve_overlay_dir(
                self.target_val_overlay_dir,
                self.target_dataset_path,
                self.target_val_split,
            )
            if dataset_part == self.TGT_VAL
            else None
        )
        cache_key = {
            self.SRC_VAL: "Val-Cells",
            self.TGT_VAL: "TargetVal-Cells",
            self.SRC_TRAIN: "Train-Cells",
            self.TGT_TRAIN: "TargetTrain-Cells",
        }.get(dataset_part, f"{dataset_part}-Cells")

        if is_target_unlabeled_eval_mode:
            self.logger.info(
                "Target validation evaluation=plot: generating overlays without classification metrics."
            )
            with torch.no_grad():
                for _, batch in tqdm.tqdm(
                    enumerate(val_dataloader), total=len(val_dataloader)
                ):
                    images, _, _, image_names, offsets = batch
                    self.get_cellvit_result_unlabeled(
                        images=images,
                        image_names=image_names,
                        postprocessor=postprocessor,
                        offsets=offsets,
                        overlay_dir=overlay_dir,
                        pred_class_map=self.final_eval_map.get("pred_map"),
                    )

            scalar_metrics = {
                "Detection-F1/Validation": 0.0,
                "Detection-Precision/Validation": 0.0,
                "Detection-Recall/Validation": 0.0,
                "Detection-Count/Validation": 0,
                "F1-Score/Validation/Macro": 0.0,
                "Precision/Validation/Macro": 0.0,
                "Recall/Validation/Macro": 0.0,
                "Loss/Validation": 0.0,
            }
            return (
                scalar_metrics,
                None,
                0.0,
                {"predictions": None, "probabilities": None, "gt": None},
            )

        extracted_cells: List[dict] = []
        f1s, precs, recs = [], [], []
        det_metrics = self._empty_detection_metrics()

        if epoch == 0:
            dataset_cache_exists = self._test_cache_exists(dataset_part)
            if dataset_cache_exists:
                extracted_cells = self._load_from_cache(dataset_part)
            else:
                self.logger.info(f"Extracting validation cells [{dataset_part}]")
                with torch.no_grad():
                    for _, batch in tqdm.tqdm(
                        enumerate(val_dataloader), total=len(val_dataloader)
                    ):
                        images, cell_gt_batch, types_batch, image_names, offsets = batch

                        cells, bf1, bprec, brec = self.get_cellvit_result_labeled(
                            images=images,
                            cell_gt_batch=cell_gt_batch,
                            types_batch=types_batch,
                            image_names=image_names,
                            postprocessor=postprocessor,
                            offsets=offsets,
                            dataset_part=dataset_part,
                        )
                        extracted_cells += cells
                        f1s += bf1
                        precs += bprec
                        recs += brec

                det_metrics = {"f1s": f1s, "precs": precs, "recs": recs}
                if len(f1s) > 0:
                    self.logger.info(
                        f"[{dataset_part}] extraction det-metrics - "
                        f"F1={np.mean(np.array(f1s)):.3f}, "
                        f"Prec={np.mean(np.array(precs)):.3f}, "
                        f"Rec={np.mean(np.array(recs)):.3f}, "
                        f"Cells={len(extracted_cells)}"
                    )
                self._cache_results(extracted_cells, dataset_part)

            self.cached_dataset[cache_key] = extracted_cells
        else:
            extracted_cells = self.cached_dataset[cache_key]

        extracted_cells = self._drop_ignored_cells(
            extracted_cells,
            context=f"validation[{dataset_part}]",
        )
        if epoch == 0:
            self._log_token_range(f"{dataset_part}", extracted_cells)

        is_target_val = dataset_part == self.TGT_VAL
        val_embed_ds = BaseCellEmbeddingDataset(extracted_cells)
        val_embed_dl = self._embedding_dataloader(
            val_embed_ds, batch_size=256, shuffle=False
        )

        if len(val_embed_ds) == 0:
            raise RuntimeError(f"[{dataset_part}] No cells extracted for validation.")

        with torch.no_grad():
            self.model.eval()
            self.loss_avg_tracker.reset()
            self.ce_loss_tracker.reset()

            predictions, probabilities, gt = [], [], []
            val_loop = tqdm.tqdm(enumerate(val_embed_dl), total=len(val_embed_dl))
            for _, batch in val_loop:
                out = self.validation_step(batch)
                predictions.append(out["predictions"])
                probabilities.append(out["probabilities"])
                gt.append(out["gt"])

        predictions = torch.cat(predictions, dim=0).detach().cpu()
        probabilities = torch.cat(probabilities, dim=0).detach().cpu()
        gt = torch.cat(gt, dim=0).detach().cpu()
        if is_target_val:
            pred_map = self.final_eval_map.get("pred_map", {})
            gt_map = self.final_eval_map.get("gt_map", {})
            all_vals = list(pred_map.values()) + list(gt_map.values())
            mapped_num_classes = (
                int(max(all_vals)) + 1 if all_vals else int(self.num_classes)
            )
            default_class = max(mapped_num_classes - 1, 0)

            def _map_labels(
                labels: torch.Tensor, label_map: dict, allow_negative: bool = False
            ) -> torch.Tensor:
                if not label_map:
                    return labels
                mapped = torch.full_like(labels, default_class)
                if allow_negative:
                    mapped[labels < 0] = -1
                for k, v in label_map.items():
                    ik = int(k)
                    iv = int(v)
                    if allow_negative and iv < 0:
                        mapped[labels == ik] = -1
                        continue
                    if 0 <= iv < int(mapped_num_classes):
                        mapped[labels == ik] = iv
                return mapped

            mapped_pred = _map_labels(predictions, pred_map)
            mapped_gt = (
                _map_labels(gt, gt_map, allow_negative=True)
                if gt.numel() > 0 and int(gt.max().item()) >= mapped_num_classes
                else gt
            )
            predictions = mapped_pred
            gt = mapped_gt
            valid_mask = gt >= 0
            if valid_mask.numel() > 0 and not bool(torch.all(valid_mask)):
                predictions = predictions[valid_mask]
                gt = gt[valid_mask]

            if gt.numel() == 0:
                raise RuntimeError(
                    f"Val[{dataset_part}] all samples were filtered by gt_map."
                )

            if mapped_num_classes <= 2:
                f1_per_class = tmf.f1_score(
                    predictions, gt, task="binary", average=None
                ).detach()
                precision_per_class = tmf.precision(
                    predictions, gt, task="binary", average=None
                ).detach()
                recall_per_class = tmf.recall(
                    predictions, gt, task="binary", average=None
                ).detach()
            else:
                f1_per_class = tmf.f1_score(
                    predictions,
                    gt,
                    task="multiclass",
                    num_classes=mapped_num_classes,
                    average=None,
                ).detach()
                precision_per_class = tmf.precision(
                    predictions,
                    gt,
                    task="multiclass",
                    num_classes=mapped_num_classes,
                    average=None,
                ).detach()
                recall_per_class = tmf.recall(
                    predictions,
                    gt,
                    task="multiclass",
                    num_classes=mapped_num_classes,
                    average=None,
                ).detach()
        else:
            if self.num_classes <= 2:
                f1_per_class = tmf.f1_score(
                    predictions, gt, task="binary", average=None
                ).detach()
                precision_per_class = tmf.precision(
                    predictions, gt, task="binary", average=None
                ).detach()
                recall_per_class = tmf.recall(
                    predictions, gt, task="binary", average=None
                ).detach()
            else:
                f1_per_class = tmf.f1_score(
                    predictions,
                    gt,
                    task="multiclass",
                    num_classes=self.num_classes,
                    average=None,
                ).detach()
                precision_per_class = tmf.precision(
                    predictions,
                    gt,
                    task="multiclass",
                    num_classes=self.num_classes,
                    average=None,
                ).detach()
                recall_per_class = tmf.recall(
                    predictions,
                    gt,
                    task="multiclass",
                    num_classes=self.num_classes,
                    average=None,
                ).detach()

        macro_f1 = float(torch.mean(f1_per_class).cpu().item())
        macro_precision = float(torch.mean(precision_per_class).cpu().item())
        macro_recall = float(torch.mean(recall_per_class).cpu().item())
        det_summary = self._summarize_detection_metrics(det_metrics)

        scalar_metrics = {
            "Detection-F1/Validation": float(det_summary["F1"]),
            "Detection-Precision/Validation": float(det_summary["Prec"]),
            "Detection-Recall/Validation": float(det_summary["Rec"]),
            "Detection-Count/Validation": int(det_summary["Count"]),
        }
        for idx in range(int(f1_per_class.numel())):
            scalar_metrics[f"F1-Score/Validation/Class_{idx}"] = float(
                f1_per_class[idx].cpu().item()
            )
        scalar_metrics["F1-Score/Validation/Macro"] = macro_f1
        for idx in range(int(precision_per_class.numel())):
            scalar_metrics[f"Precision/Validation/Class_{idx}"] = float(
                precision_per_class[idx].cpu().item()
            )
        scalar_metrics["Precision/Validation/Macro"] = macro_precision
        for idx in range(int(recall_per_class.numel())):
            scalar_metrics[f"Recall/Validation/Class_{idx}"] = float(
                recall_per_class[idx].cpu().item()
            )
        scalar_metrics["Recall/Validation/Macro"] = macro_recall
        scalar_metrics["Loss/Validation"] = float(self.loss_avg_tracker.avg)

        val_results = {
            "predictions": predictions,
            "probabilities": probabilities,
            "gt": gt,
            "classification": {
                "F1_Macro": macro_f1,
                "Precision_Macro": macro_precision,
                "Recall_Macro": macro_recall,
                "Cell_Count": int(gt.numel()),
            },
        }

        self.logger.info(
            f"Val[{dataset_part}] - F1:{macro_f1:.4f} "
            f"Precision:{macro_precision:.4f} Recall:{macro_recall:.4f}"
        )

        return scalar_metrics, None, float(macro_f1), val_results

    def validation_step(self, batch: Any):
        cell_tokens = self._align_final_token_dim(batch[0].to(self.device))
        cell_types = batch[2].to(self.device)
        skip_loss = self._final_eval_mode and self.final_eval_skip_loss

        with self._autocast_context():
            logits = self.model.forward(cell_tokens)
            loss = (
                torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
                if skip_loss
                else self.loss_fn(logits, cell_types)
            )

        self.loss_avg_tracker.update(loss.detach())
        self.ce_loss_tracker.update(loss.detach())
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)
        return {"gt": cell_types, "probabilities": probs, "predictions": pred}
