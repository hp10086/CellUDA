#!/usr/bin/env python3
"""Convert the official CoNSeP / BRCA-M2C / PanopTILs downloads into the `data/` layout used by CellUDA.

  python scripts/prepare_data.py --dataset consep    --raw ~/raw/CoNSeP
  python scripts/prepare_data.py --dataset brca_m2c  --raw ~/raw/Dataset-BRCA-M2C
  python scripts/prepare_data.py --dataset panoptils --raw ~/raw/PanopTILs

`--raw` is the extracted download: for CoNSeP the folder that contains `Train/` and `Test/`
(each with `Images/` and `Labels/`); for BRCA-M2C the repository folder that contains
`images/` and `labels/`; for PanopTILs a folder that contains the downloaded PanopTILs folders
(the training release and the validation release, any nesting) - the script looks for
`rgbs/<stem>.png` and `csv/<stem>.csv` below it. For every stem listed in
configs/splits/<dataset>/{train,test}.csv the script writes

  data/<dataset>/<split>/images/<stem>.png     8-bit RGB, 40x, 1024x1024 (padded with white)
  data/<dataset>/<split>/labels/<stem>.csv     no header, one cell per line: x,y,type

CoNSeP: images (1000x1000 px, 40x) are padded with white to 1024x1024; the centroids
(`inst_centroid`) and types (`inst_type`, 1-7) of each .mat file are written as x,y,type.
BRCA-M2C: the 20x patches are upscaled by 2 (LANCZOS) to 40x and padded with white to
1024x1024; the official label rows `y x class` are written as 2*x,2*y,class (class 1-3).
PanopTILs: the official 1024x1024 tiles (0.25 MPP) are copied unchanged; for every nucleus
polygon of the official csv the centroid is written with the class id of its `group`
(1 CancerEpithelium, 2 NormalEpithelium, 3 TILsCell, 4 ActiveTILsCell, 5 StromalCellNOS,
6 ActiveStromalCellNOS, 7 OtherCell, 8 UnknownOrAmbiguousCell).
"""

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("consep", "brca_m2c", "panoptils")
PANOPTILS_GROUPS = {
    "CancerEpithelium": 1,
    "NormalEpithelium": 2,
    "TILsCell": 3,
    "ActiveTILsCell": 4,
    "StromalCellNOS": 5,
    "ActiveStromalCellNOS": 6,
    "OtherCell": 7,
    "UnknownOrAmbiguousCell": 8,
}


def read_stems(dataset: str, split: str) -> list:
    path = ROOT / "configs" / "splits" / dataset / f"{split}.csv"
    with open(path, newline="") as f:
        return [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]


def find_dir(base: Path, *names: str) -> Path:
    """Sub-directory of `base` matching one of `names`, case-insensitively."""
    wanted = {n.lower() for n in names}
    for p in sorted(base.iterdir()):
        if p.is_dir() and p.name.lower() in wanted:
            return p
    raise FileNotFoundError(f"none of {names} found in {base}")


def find_file(directory: Path, stem: str, suffixes: tuple) -> Path:
    for suffix in suffixes:
        p = directory / f"{stem}{suffix}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{stem}{suffixes[0]} not found in {directory}")


PATCH_SIZE = 1024


def save_padded(image: Image.Image, path: Path) -> None:
    """Pad the image with white to PATCH_SIZE x PATCH_SIZE (bottom/right) and save it as PNG."""
    image = image.convert("RGB")
    if image.width < PATCH_SIZE or image.height < PATCH_SIZE:
        canvas = Image.new("RGB", (max(image.width, PATCH_SIZE), max(image.height, PATCH_SIZE)), (255, 255, 255))
        canvas.paste(image, (0, 0))
        image = canvas
    image.save(path)


def write_labels(path: Path, rows) -> int:
    rows = [(int(x), int(y), int(t)) for x, y, t in rows]
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return len(rows)


def out_dirs(out: Path, dataset: str, split: str):
    img_dir = out / dataset / split / "images"
    lab_dir = out / dataset / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, lab_dir


def prepare_consep(raw: Path, out: Path) -> None:
    import scipy.io as sio

    for split, src in (("train", "Train"), ("test", "Test")):
        src_dir = find_dir(raw, src)
        img_src = find_dir(src_dir, "Images")
        lab_src = find_dir(src_dir, "Labels")
        img_dir, lab_dir = out_dirs(out, "consep", split)
        n_cells = 0
        stems = read_stems("consep", split)
        for stem in stems:
            save_padded(Image.open(find_file(img_src, stem, (".png",))), img_dir / f"{stem}.png")
            mat = sio.loadmat(find_file(lab_src, stem, (".mat",)))
            xy = np.rint(np.asarray(mat["inst_centroid"], dtype=float)).astype(int)
            types = np.asarray(mat["inst_type"]).reshape(-1).astype(int)
            if len(xy) != len(types):
                raise ValueError(f"{stem}.mat: {len(xy)} centroids but {len(types)} types")
            n_cells += write_labels(lab_dir / f"{stem}.csv", zip(xy[:, 0], xy[:, 1], types))
        print(f"consep/{split}: {len(stems)} images, {n_cells} cells -> {img_dir.parent}")


def prepare_brca_m2c(raw: Path, out: Path) -> None:
    img_src = find_dir(raw, "images")
    lab_src = find_dir(raw, "labels")
    for split in ("train", "test"):
        img_dir, lab_dir = out_dirs(out, "brca_m2c", split)
        n_cells = 0
        stems = read_stems("brca_m2c", split)
        for stem in stems:
            image = Image.open(find_file(img_src, stem, (".png",))).convert("RGB")
            image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
            save_padded(image, img_dir / f"{stem}.png")
            label_path = find_file(lab_src, f"{stem}_gt_class_coords", (".txt",))
            yxc = np.loadtxt(label_path, dtype=int, ndmin=2)
            n_cells += write_labels(lab_dir / f"{stem}.csv", ((2 * x, 2 * y, c) for y, x, c in yxc))
        print(f"brca_m2c/{split}: {len(stems)} images, {n_cells} cells -> {img_dir.parent}")


def _polygon_centroid(xs, ys):
    n = len(xs)
    if n == 1:
        return xs[0], ys[0]
    if n == 2:
        return (xs[0] + xs[1]) / 2.0, (ys[0] + ys[1]) / 2.0
    area = cx = cy = 0.0
    for i in range(n):
        j = (i + 1) % n
        cross = xs[i] * ys[j] - xs[j] * ys[i]
        area += cross
        cx += (xs[i] + xs[j]) * cross
        cy += (ys[i] + ys[j]) * cross
    area *= 0.5
    if area == 0:
        return sum(xs) / n, sum(ys) / n
    return cx / (6.0 * area), cy / (6.0 * area)


def _index_by_stem(raw: Path, dirname: str, suffix: str, stems: set) -> dict:
    index = {}
    for root, _, files in os.walk(raw, followlinks=True):
        if Path(root).name.lower() != dirname:
            continue
        for name in files:
            p = Path(root) / name
            if p.suffix.lower() == suffix and p.stem in stems:
                if p.stem in index:
                    raise FileNotFoundError(
                        f"{dirname}/{p.name} found twice under {raw} ({index[p.stem]} and {p}); "
                        "point --raw at one training and one validation release only"
                    )
                index[p.stem] = p
    if not index:
        raise FileNotFoundError(f"no {dirname}/*{suffix} files found under {raw}")
    return index


def prepare_panoptils(raw: Path, out: Path) -> None:
    stems = {split: read_stems("panoptils", split) for split in ("train", "test")}
    wanted = set(stems["train"]) | set(stems["test"])
    images = _index_by_stem(raw, "rgbs", ".png", wanted)
    labels = _index_by_stem(raw, "csv", ".csv", wanted)
    for split in ("train", "test"):
        img_dir, lab_dir = out_dirs(out, "panoptils", split)
        n_cells = 0
        for stem in stems[split]:
            if stem not in images or stem not in labels:
                raise FileNotFoundError(f"{stem}: rgbs/{stem}.png or csv/{stem}.csv not found under {raw}")
            shutil.copyfile(images[stem], img_dir / f"{stem}.png")
            rows = []
            with open(labels[stem], newline="") as f:
                for row in csv.DictReader(f):
                    group = (row.get("group") or "").strip()
                    if group not in PANOPTILS_GROUPS:
                        raise ValueError(f"{stem}.csv: unknown nucleus group '{group}'")
                    xs = [float(v) for v in row.get("coords_x", "").split(",") if v.strip()]
                    ys = [float(v) for v in row.get("coords_y", "").split(",") if v.strip()]
                    if len(xs) != len(ys) or not xs:
                        raise ValueError(f"{stem}.csv: bad polygon coordinates")
                    cx, cy = _polygon_centroid(xs, ys)
                    rows.append((cx, cy, PANOPTILS_GROUPS[group]))
            with open(lab_dir / f"{stem}.csv", "w", newline="") as f:
                csv.writer(f).writerows(rows)
            n_cells += len(rows)
        print(f"panoptils/{split}: {len(stems[split])} images, {n_cells} cells -> {img_dir.parent}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--raw", required=True, type=Path, help="extracted official download (see above)")
    parser.add_argument("--out", type=Path, default=ROOT / "data", help="output root")
    args = parser.parse_args()
    raw = args.raw.expanduser().resolve()
    if not raw.is_dir():
        parser.error(f"--raw {raw} is not a directory")
    try:
        {"consep": prepare_consep, "brca_m2c": prepare_brca_m2c, "panoptils": prepare_panoptils}[args.dataset](raw, args.out.expanduser())
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
