# checkpoints/

Nothing in this directory is tracked by git (see `.gitignore`).

| Path | What | How to get it |
|---|---|---|
| `CellViT-SAM-H-x40-AMP.pth` | Frozen CellViT-SAM-H segmentation model (2.8 GB) | Download from the [CellViT++ authors' Google Drive, folder `SAM`](https://drive.google.com/drive/folders/10g3Bf2khWiIgkiQscD7F7bPTfVF3-5-g); see README.md "Checkpoints" |
| `source_heads/<dataset>_seed1.pth` | Source-only classifier head (step 1) | `python run.py --config configs/source_<dataset>.yaml` |
| `adapted/<source>_to_<target>_seed1.pth` | CellUDA-adapted head (step 2) | `python run.py --config configs/<source>_to_<target>.yaml` |
