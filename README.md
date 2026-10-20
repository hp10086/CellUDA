# CellUDA: Unsupervised Domain Adaptation for Cell Classification in Histopathology

Official code for **"CellUDA: Unsupervised Domain Adaptation for Cell Classification in Histopathology" (WACV 2027)**.

## 1. Installation

```bash
git clone https://github.com/hp10086/CellUDA.git CellUDA && cd CellUDA
cd CellUDA

conda env create -f environment.yaml
conda activate celluda

python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0
python -m pip install -r requirements.txt
```

## 2. Data preparation

**1. Download** the datasets from their providers to `~/raw/`:


| Dataset                           | Download                                                                                                                                                                        |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CoNSeP (Graham et al., 2019)      | [TIA Centre, University of Warwick](https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/) (see also the [HoVer-Net repository](https://github.com/vqdang/hover_net#datasets)) |
| BRCA-M2C (Abousamra et al., 2021) | [github.com/TopoXLab/Dataset-BRCA-M2C](https://github.com/TopoXLab/Dataset-BRCA-M2C)                                                                                            |
| PanopTILs (Amgad et al., 2022)    | [sites.google.com/view/panoptils](https://sites.google.com/view/panoptils)                                                                                                      |

**2. Convert** them into the `data/` folder of the repository with `scripts/prepare_data.py`:

```bash
python scripts/prepare_data.py --dataset consep    --raw ~/raw/CoNSeP  
python scripts/prepare_data.py --dataset brca_m2c  --raw ~/raw/Dataset-BRCA-M2C  
python scripts/prepare_data.py --dataset panoptils --raw ~/raw/PanopTILs   
```

The script converts the annotations of all three datasets into the same detection format (one `x,y,type` row per nucleus: centroid and cell type), upscales images and coordinates to 40× (0.25 µm/px) where necessary, pads images smaller than 1024 × 1024 with white to 1024 × 1024, and writes only the images of the paper's train/test splits, provided in [`configs/splits/`](configs/splits/):

```text
data/
├── consep/
│   ├── train/
│   │   ├── images/   27 .png
│   │   └── labels/   27 .csv
│   └── test/
│       ├── images/   14 .png
│       └── labels/   14 .csv
├── brca_m2c/
│   ├── train/
│   │   ├── images/   80 .png
│   │   └── labels/   80 .csv
│   └── test/
│       ├── images/   30 .png
│       └── labels/   30 .csv
├── panoptils/
│   ├── train/
│   │   ├── images/   1709 .png
│   │   └── labels/   1709 .csv
│   └── test/
│       ├── images/   1317 .png
│       └── labels/   1317 .csv
└── your_own_data/  
    ├── train/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

## 3. Checkpoints

**CellViT-SAM-H** (frozen segmentation model): download `CellViT-SAM-H-x40-AMP.pth` from the
[CellViT++ repository](https://github.com/TIO-IKIM/CellViT-plus-plus#model-checkpoints) (Google Drive folder [`SAM`](https://drive.google.com/drive/folders/10g3Bf2khWiIgkiQscD7F7bPTfVF3-5-g)) and place it at `checkpoints/CellViT-SAM-H-x40-AMP.pth`.

**Classifier heads**: `run.py` writes the source head to `checkpoints/source_heads/` and the adapted head to `checkpoints/adapted/`.

## 4. Usage

**Step 1 — source-only head**:

```bash
python run.py --config configs/source_consep.yaml --gpu 0      # -> checkpoints/source_heads/consep_seed1.pth
python run.py --config configs/source_brca.yaml --gpu 0        # -> checkpoints/source_heads/brca_m2c_seed1.pth
python run.py --config configs/source_panoptils.yaml --gpu 0   # -> checkpoints/source_heads/panoptils_seed1.pth
```

**Step 2 — CellUDA adaptation**:

```bash
python run.py --config configs/template.yaml --gpu 0               # your own target data: fill in the placeholders first; any of the three datasets above can be the source
# python run.py --config configs/consep_to_brca.yaml --gpu 0       # -> checkpoints/adapted/consep_to_brca_m2c_seed1.pth
# python run.py --config configs/brca_to_consep.yaml --gpu 0       # -> checkpoints/adapted/brca_m2c_to_consep_seed1.pth
# python run.py --config configs/panoptils_to_consep.yaml --gpu 0  # -> checkpoints/adapted/panoptils_to_consep_seed1.pth
# python run.py --config configs/consep_to_panoptils.yaml --gpu 0  # -> checkpoints/adapted/consep_to_panoptils_seed1.pth
```

For an unannotated target dataset, set `data.evaluation: plot` in the config to draw overlays of the predicted cell types.

## 5. License

* Our code (`celluda/`, `run.py`, `scripts/`, `configs/`): Apache License 2.0 (`LICENSE`).
* `third_party/cellvit/` is derived from [CellViT++](https://github.com/TIO-IKIM/CellViT-plus-plus) and keeps its license (Apache 2.0 with the Commons Clause): **non-commercial use only**, and **any use of this repository must cite CellViT and CellViT++** (section 6). Full terms:
  `third_party/cellvit/LICENSE` and `NOTICE`.
* Datasets and the CellViT-SAM-H weights: see their providers (sections 2 and 3).

## 6. Citation

If you use this code, please cite CellUDA **and** CellViT / CellViT++:

```bibtex
@inproceedings{zhao2027celluda,
  title     = {CellUDA: Unsupervised Domain Adaptation for Cell Classification in Histopathology},
  author    = {Zhao, Haipeng and Abousamra, Shahira and Xu, Meilong and Ganguly, Aniruddha and Miller, Michael L. and Chen, Chao},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year      = {2027}
}
```

```bibtex
@article{horst2026cellvit++,
  title={CellViT++: Energy-efficient and adaptive cell segmentation and classification using foundation models},
  author={Fabian H{\"o}rst and Moritz Rempe and Helmut Becker and Lukas Heine and Julius Keyl and Jens Kleesiek},
  journal={Computer methods and programs in biomedicine},
  year={2026},
  volume={277},
  pages={109206},
  url={https://api.semanticscholar.org/CorpusID:285402366}
}

@ARTICLE{Horst2024,
  title    = "{CellViT}: Vision Transformers for precise cell segmentation and
              classification",
  author   = "H{\"o}rst, Fabian and Rempe, Moritz and Heine, Lukas and Seibold,
              Constantin and Keyl, Julius and Baldini, Giulia and Ugurel, Selma
              and Siveke, Jens and Gr{\"u}nwald, Barbara and Egger, Jan and
              Kleesiek, Jens",
  journal  = "Med. Image Anal.",
  volume   =  94,
  pages    = "103143",
  month    =  may,
  year     =  2024,
}
```

## 7. Acknowledgements

This repository builds on [CellViT++](https://github.com/TIO-IKIM/CellViT-plus-plus) by Fabian Hörst et al. and on [Segment Anything](https://github.com/facebookresearch/segment-anything) by Meta AI.
