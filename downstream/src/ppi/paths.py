"""Project path helpers with config/env overrides."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def get_project_root() -> Path:
    """Return repository root (two levels above src/ppi)."""
    return Path(__file__).resolve().parents[2]


def _resolve_path(value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    return Path(os.path.expanduser(value)).resolve()


def load_paths_config() -> Dict[str, Any]:
    cfg_path = get_project_root() / "configs" / "paths.json"
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_path(key: str, env_key: str, default_rel: str) -> Path:
    cfg = load_paths_config()
    if env_key in os.environ:
        return _resolve_path(os.environ[env_key])
    if key in cfg:
        return _resolve_path(str(cfg[key]))
    return (get_project_root() / default_rel).resolve()


def get_data_root() -> Path:
    """Default dataset root used by training scripts."""
    return _get_path("data_root", "PPI_DATA_ROOT", "data/DeepPPIPS")


def get_embedding_dir() -> Path:
    """Default directory for per-protein embeddings."""
    return _get_path("embedding_dir", "PPI_EMB_DIR", "data/embeddings")


def get_outputs_root() -> Path:
    """Root directory for runs/results/visualizations."""
    return _get_path("outputs_root", "PPI_OUTPUTS_DIR", "outputs")


def get_vertex_residue_dir() -> Path:
    """Default directory for vertex residue mapping JSON files."""
    return _get_path("vertex_residue_dir", "PPI_VERTEX_RESIDUE_DIR", "data/vertex_residue")


def get_interim_preprocess_dir() -> Path:
    """Default directory for preprocessing intermediate files."""
    return _get_path("interim_preprocess_dir", "PPI_INTERIM_PREPROCESS_DIR", "data/interim/preprocess")
