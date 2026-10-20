# -*- coding: utf-8 -*-
# CellUDA: patch-level cell-detection dataset (images + centroid/type labels) for source and target domains.
# Modified from CellViT++ (https://github.com/TIO-IKIM/CellViT-plus-plus) — changes: derived from cellvit/training/datasets/detection_dataset.py; adds split file lists, label mapping with ignore labels (<0), unlabeled mode for target data, padding to 1024x1024 and crop offsets.
import csv
import os
from pathlib import Path
from typing import Dict, Optional, Union, Tuple, List

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset
import tqdm
class DetectionDataset(Dataset):
    """
    Patch-level dataset that returns:
      - image tensor [C,H,W]
      - detections: list[(x,y)] in PATCH coordinates
      - types: list[int] aligned with detections
      - img_name: str
      - (optional) offset: (x0,y0) crop origin in original image coordinates
    """

    def __init__(
        self,
        dataset_path: Union[Path, str],
        split: str,
        filelist_path: Union[Path, str] = None,
        label_mapping: Optional[Dict[int, int]] = None,
        no_label: bool = False,
        normalize_mean: Optional[Tuple[float, float, float]] = None,
        normalize_std: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        super().__init__()
        self.label_mapping = label_mapping
        self.no_label = bool(no_label)
        self.normalize_mean = tuple(normalize_mean or (0.485, 0.456, 0.406))
        self.normalize_std = tuple(normalize_std or (0.229, 0.224, 0.225))

        self.dataset_path = Path(dataset_path)
        self.split = split
        self.image_path = self.dataset_path / self.split / "images"
        self.annotation_path = self.dataset_path / self.split / "labels"

        self.images = [
            f
            for f in sorted(self.image_path.glob("*"))
            if f.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ]

        if filelist_path is not None:
            selected = []
            with open(filelist_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) == 0:
                        continue
                    selected.append(row[0])
            selected = set(selected)
            self.images = [f for f in self.images if f.stem in selected]

        if self.no_label:
            self.annotations = [None] * len(self.images)
        else:
            self.annotations = []
            for img_path in self.images:
                img_name = img_path.stem
                self.annotations.append(self.annotation_path / f"{img_name}.csv")

        self.cache_images: Dict[str, Image.Image] = {}
        self.cache_annotations: Dict[str, List[List[str]]] = {}
        self._base_transform = self._build_base_transform()

    def cache_dataset(self) -> None:
        """Cache the dataset in memory."""
        for img_path, annot_path in tqdm.tqdm(
            zip(self.images, self.annotations), total=len(self.images)
        ):
            img = Image.open(img_path).convert("RGB")
            self.cache_images[img_path.stem] = img

            if self.no_label or annot_path is None:
                cell_annot = []
            else:
                with open(annot_path, "r") as f:
                    reader = csv.reader(f)
                    cell_annot = list(reader)
            self.cache_annotations[img_path.stem] = cell_annot

    def __len__(self) -> int:
        return len(self.images)

    @staticmethod
    def _parse_annotation_rows(
        cell_annot: List[List[str]],
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        detections: List[Tuple[int, int]] = []
        types: List[int] = []
        for row in cell_annot:
            x, y, t = row[0], row[1], row[2]
            xi = int(float(x))
            yi = int(float(y))
            ti = int(float(t))
            detections.append((xi, yi))
            types.append(ti)

        return detections, types

    def _build_base_transform(self) -> A.Compose:
        transforms = [
            A.Normalize(mean=self.normalize_mean, std=self.normalize_std),
            ToTensorV2(),
        ]
        return A.Compose(
            transforms,
            keypoint_params=A.KeypointParams(
                format="xy",
                label_fields=["keypoint_labels"],
                remove_invisible=True,
            ),
        )

    def __getitem__(self, index: int):
        img_path = self.images[index]
        img_name = img_path.stem
        offset = (0, 0)

        img_pil = self.cache_images[img_name]
        cell_annot = self.cache_annotations[img_name]

        detections, types = self._parse_annotation_rows(cell_annot)
        if self.label_mapping is not None:
            kept_det: List[Tuple[int, int]] = []
            kept_types: List[int] = []
            for (xy, t_raw) in zip(detections, types):
                mapped = int(self.label_mapping.get(int(t_raw), int(t_raw)))
                if mapped < 0:
                    continue
                kept_det.append(xy)
                kept_types.append(mapped)
            detections, types = kept_det, kept_types

        img = np.array(img_pil).astype(np.uint8)
        h, w, _ = img.shape
        target_h = 1024
        target_w = 1024
        pad_bottom = target_h - h
        pad_right = target_w - w
        if pad_bottom > 0 or pad_right > 0:
            img = np.pad(
                img,
                ((0, pad_bottom), (0, pad_right), (0, 0)),
                mode="constant",
                constant_values=255,
            )

        out = self._base_transform(image=img, keypoints=detections, keypoint_labels=types)
        img_t = out["image"]
        detections = list(out["keypoints"])
        types = list(out.get("keypoint_labels", types))
        return img_t, detections, types, img_name, offset


    @staticmethod
    def collate_batch(batch: List[Tuple]):
        imgs, detections_list, types_list, names, offsets = zip(*batch)
        imgs = torch.stack(imgs, dim=0)
        return imgs, list(detections_list), list(types_list), list(names), list(offsets)
