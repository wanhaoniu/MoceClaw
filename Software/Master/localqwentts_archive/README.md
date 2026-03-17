## localqwentts Archive

This directory stores reproducible snapshots of the local `localqwentts`
Conda environment.

Use the snapshot script to export:

- a minimal Conda history file
- a full Conda environment file
- an explicit Conda package lockfile
- `pip freeze`
- Python/Torch/runtime metadata
- basic Hugging Face / ModelScope cache inventory

### Create a snapshot

```bash
bash Software/Master/localqwentts_archive/snapshot_localqwentts.sh
```

Override the target environment if needed:

```bash
LOCALQWENTTS_ENV_NAME=myenv bash Software/Master/localqwentts_archive/snapshot_localqwentts.sh
```

### Output

Snapshots are written under:

```text
Software/Master/localqwentts_archive/snapshots/<env>-<timestamp>/
```

### Suggested restore order on a new Linux system

1. Install Conda or Miniconda.
2. Recreate the environment from `conda_env_history.yml` first.
3. If exact package pinning is needed on the same platform, use `conda_list_explicit.txt`.
4. Use `pip_freeze.txt` only for missing pip-only packages.
5. Re-download model weights from Hugging Face or copy your cache separately.
