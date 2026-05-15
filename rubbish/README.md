# Rubbish Archive

This directory stores local-only files that are not needed for the minimal runnable project tree.

Archived categories:

- `heavy_assets/`: local datasets, model weights, and HuggingFace/COMET caches.
- `generated_outputs/`: raw translations, per-direction metric JSON files, smoke outputs, historical archives, and logs.
- `machine_state/`: AutoDL local databases and machine-specific logs.
- `cache/`: pytest cache, Python bytecode cache, and notebook checkpoints.
- `historical_setup/`: one-off AutoDL prompt/setup files and local plugin config.

These files are intentionally ignored by git. To rerun the full evaluation locally, restore the required assets to their expected root paths:

- `rubbish/heavy_assets/data` -> `data`
- `rubbish/heavy_assets/models` -> `models`
- `rubbish/heavy_assets/root-cache` -> `root-cache`
