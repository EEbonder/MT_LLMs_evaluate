# Rubbish Archive

This directory stores local-only files that are not needed for the minimal runnable project tree.

Archived categories:

- `heavy_assets/`: large local caches or temporary assets that are not needed in the minimal repo tree.
- `generated_outputs/`: raw translations, per-direction metric JSON files, smoke outputs, historical archives, and logs.
- `machine_state/`: AutoDL local databases and machine-specific logs.
- `cache/`: pytest cache, Python bytecode cache, and notebook checkpoints.
- `historical_setup/`: one-off AutoDL prompt/setup files and local plugin config.

These files are intentionally ignored by git.

Important: `data/` and `models/` are expected to stay at the project root for local reruns, but they are ignored by git and are excluded from the public repository.

If local caches were archived, restore them to their expected root paths before a full offline rerun:

- `rubbish/heavy_assets/root-cache` -> `root-cache`
