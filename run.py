#!/usr/bin/env python3
"""CellUDA entry point. Everything is driven by one YAML config from configs/.

  python run.py --config configs/source_consep.yaml                  # no uda block      -> train the source-only head
  python run.py --config configs/consep_to_brca.yaml                 # uda.enable: true  -> CellUDA adaptation + final target evaluation
  python run.py --config configs/consep_to_brca.yaml --mode extract  # only segment + cache cell embeddings (reports detection F1)
  python run.py --config configs/consep_to_brca.yaml --mode evaluate --classifier_path checkpoints/adapted/consep_to_brca_m2c_seed1.pth

Run from the repository root; all paths in the configs are relative to it. `--help` lists the overrides.
"""

from celluda.cli import main

if __name__ == "__main__":
    main()
