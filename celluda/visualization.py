# -*- coding: utf-8 -*-
# CellUDA: qualitative overlays of predicted cell types (used with `data.evaluation: plot`).
import colorsys
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont


TYPE_PALETTE = np.array(
    [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
        [255, 0, 255],
        [255, 128, 0],
        [128, 0, 255],
        [0, 128, 255],
        [128, 255, 0],
    ],
    dtype=np.uint8,
)
TYPE_OVERLAY_ALPHA = 0.3


def as_rgb_tuple(value: Any) -> Tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Expected RGB triple, got: {value!r}")
    return (float(value[0]), float(value[1]), float(value[2]))


def resolve_overlay_dir(
    configured_dir: Union[Path, str],
    dataset_path: Union[Path, str],
    target_val_split: str,
) -> Path:
    if configured_dir:
        return Path(configured_dir)
    if dataset_path:
        return Path(dataset_path) / f"{target_val_split}_overlay"
    return Path("runtime/target_val_overlay")


class CellClassificationVisualizer:
    def __init__(
        self,
        denormalize: bool = True,
        norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        norm_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        show_titles: bool = False,
    ) -> None:
        self.denormalize = bool(denormalize)
        self.norm_mean = as_rgb_tuple(norm_mean)
        self.norm_std = as_rgb_tuple(norm_std)
        self.show_titles = bool(show_titles)

    def _to_uint8_image(self, image: torch.Tensor) -> np.ndarray:
        if torch.is_tensor(image):
            image_cpu = image.detach().cpu()
            if image_cpu.ndim == 3:
                image_cpu = image_cpu.permute(1, 2, 0)
            elif image_cpu.ndim == 2:
                image_cpu = image_cpu.unsqueeze(-1)
            image_np = image_cpu.numpy()
        else:
            image_np = np.asarray(image)
            if (
                image_np.ndim == 3
                and image_np.shape[0] in (1, 3)
                and image_np.shape[-1] not in (1, 3)
            ):
                image_np = np.transpose(image_np, (1, 2, 0))

        if image_np.ndim == 2:
            image_np = np.repeat(image_np[:, :, None], 3, axis=2)
        if image_np.shape[-1] == 1:
            image_np = np.repeat(image_np, 3, axis=-1)
        elif image_np.shape[-1] > 3:
            image_np = image_np[:, :, :3]

        if np.issubdtype(image_np.dtype, np.floating):
            image_float = image_np.astype(np.float32)
            if self.denormalize and image_float.shape[-1] == 3:
                mean = np.asarray(self.norm_mean, dtype=np.float32).reshape(1, 1, 3)
                std = np.asarray(self.norm_std, dtype=np.float32).reshape(1, 1, 3)
                image_float = np.clip(image_float * std + mean, 0.0, 1.0)
                return (image_float * 255.0).round().astype(np.uint8)

            min_value = float(image_float.min())
            max_value = float(image_float.max())
            if max_value <= 1.0 and min_value >= 0.0:
                image_float = image_float * 255.0
            elif max_value - min_value > 1e-6:
                image_float = (image_float - min_value) / (max_value - min_value)
                image_float = image_float * 255.0
            else:
                image_float = np.zeros_like(image_float)
            return np.clip(image_float, 0, 255).round().astype(np.uint8)

        return np.clip(image_np, 0, 255).astype(np.uint8)

    @staticmethod
    def _type_color(cell_type: int) -> np.ndarray:
        if int(cell_type) <= len(TYPE_PALETTE):
            return TYPE_PALETTE[int(cell_type) - 1]
        hue = (int(cell_type) * 0.61803398875) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
        return np.array(
            [int(red * 255), int(green * 255), int(blue * 255)], dtype=np.uint8
        )

    @staticmethod
    def predict_types(
        model: nn.Module,
        patch_tokens: List[torch.Tensor],
        device: Union[str, torch.device],
        align_tokens: Callable[[torch.Tensor], torch.Tensor],
    ) -> np.ndarray:
        if not patch_tokens:
            return np.zeros((0,), dtype=np.int64)

        tokens = [
            token
            if torch.is_tensor(token)
            else torch.as_tensor(token, dtype=torch.float32)
            for token in patch_tokens
        ]
        token_batch = align_tokens(torch.stack(tokens, dim=0).to(device))
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                predictions = model.forward(token_batch).argmax(dim=1)
        finally:
            model.train(was_training)
        return predictions.detach().cpu().numpy()

    def build_type_map(
        self,
        inst_map: np.ndarray,
        cell_predictions: Mapping,
        patch_tokens: List[torch.Tensor],
        model: nn.Module,
        device: Union[str, torch.device],
        align_tokens: Callable[[torch.Tensor], torch.Tensor],
        pred_class_map: Optional[Mapping] = None,
    ) -> np.ndarray:
        type_map = np.zeros(inst_map.shape, dtype=np.int32)
        if not cell_predictions:
            return type_map

        inst_ids = list(cell_predictions.keys())
        predictions = self.predict_types(model, patch_tokens, device, align_tokens)
        if len(predictions) != len(inst_ids):
            raise RuntimeError(
                "[TypeMap] token count mismatch: "
                f"preds={len(predictions)}, inst={len(inst_ids)}."
            )

        for inst_id, prediction in zip(inst_ids, predictions):
            if int(inst_id) <= 0:
                continue
            mapped = int(prediction)
            if pred_class_map:
                mapped = int(pred_class_map.get(mapped, mapped))
            type_map[inst_map == int(inst_id)] = mapped
        return type_map

    def save_type_overlay(
        self,
        image: torch.Tensor,
        type_map: np.ndarray,
        out_path: Path,
        inst_map: Optional[np.ndarray] = None,
        contour_thickness: int = 2,
    ) -> None:
        image_np = self._to_uint8_image(image)
        if image_np.shape[:2] != type_map.shape:
            image_np = np.asarray(
                Image.fromarray(image_np).resize(
                    (type_map.shape[1], type_map.shape[0]), Image.BILINEAR
                )
            )

        overlay = image_np.copy()
        if inst_map is not None and inst_map.shape == type_map.shape:
            inst_ids = np.unique(inst_map)
            for inst_id in inst_ids[inst_ids > 0]:
                inst_mask = (inst_map == int(inst_id)).astype(np.uint8)
                inst_types = type_map[inst_mask > 0]
                inst_types = inst_types[inst_types > 0]
                if not inst_types.size:
                    continue
                unique_types, counts = np.unique(inst_types, return_counts=True)
                cell_type = int(unique_types[np.argmax(counts)])
                color = tuple(int(value) for value in self._type_color(cell_type))
                contours, _ = cv2.findContours(
                    inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    cv2.drawContours(
                        overlay, contours, -1, color, int(contour_thickness)
                    )
        else:
            mask = type_map > 0
            if mask.any():
                color_mask = np.zeros_like(overlay, dtype=np.uint8)
                for cell_type in np.unique(type_map):
                    if int(cell_type) == 0:
                        continue
                    color_mask[type_map == cell_type] = self._type_color(int(cell_type))
                overlay[mask] = (
                    overlay[mask] * (1.0 - TYPE_OVERLAY_ALPHA)
                    + color_mask[mask] * TYPE_OVERLAY_ALPHA
                ).astype(np.uint8)

        panel = self._compose_panel(image_np, overlay)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(panel).save(out_path)

    def _compose_panel(
        self,
        original: np.ndarray,
        overlay: np.ndarray,
        gap_px: int = 100,
        top_bar_px: int = 64,
    ) -> np.ndarray:
        if overlay.shape[:2] != original.shape[:2]:
            overlay = np.asarray(
                Image.fromarray(overlay).resize(
                    (original.shape[1], original.shape[0]), Image.BILINEAR
                )
            )

        height, width = original.shape[:2]
        gap_px = max(100, int(gap_px))
        top_bar_px = max(40, int(top_bar_px)) if self.show_titles else 0
        canvas = np.full(
            (height + top_bar_px, (width * 2) + gap_px, 3), 255, dtype=np.uint8
        )
        right_x = width + gap_px
        canvas[top_bar_px : top_bar_px + height, :width] = original
        canvas[top_bar_px : top_bar_px + height, right_x : right_x + width] = overlay

        if not self.show_titles:
            return canvas

        panel = Image.fromarray(canvas)
        draw = ImageDraw.Draw(panel)
        font = ImageFont.load_default()
        for text, center_x in (
            ("Original", width // 2),
            ("CellUDA", right_x + width // 2),
        ):
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.text(
                (
                    int(center_x - text_width / 2),
                    max(0, int((top_bar_px - text_height) / 2)),
                ),
                text,
                fill=(0, 0, 0),
                font=font,
            )
        return np.asarray(panel)

    @staticmethod
    def save_segmentation_maps(
        inst_map: np.ndarray,
        type_map: np.ndarray,
        out_dir: Path,
        image_name: str,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        base = Path(str(image_name)).stem
        Image.fromarray(type_map.astype(np.uint8), mode="L").save(
            out_dir / f"{base}_type.png"
        )

        inst_max = int(inst_map.max()) if inst_map.size else 0
        if inst_max <= np.iinfo(np.uint16).max:
            Image.fromarray(inst_map.astype(np.uint16), mode="I;16").save(
                out_dir / f"{base}_inst.png"
            )
        else:
            Image.fromarray(inst_map.astype(np.int32), mode="I").save(
                out_dir / f"{base}_inst.tiff"
            )
