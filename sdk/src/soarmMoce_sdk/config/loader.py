# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import importlib.resources as resources
import yaml


def _default_config_path() -> Path:
    pkg = "soarmMoce_sdk"
    res = resources.files(pkg) / "resources" / "configs" / "soarm_moce.yaml"
    # as_file handles zip/installed packages
    with resources.as_file(res) as p:
        return Path(p)


def _resolve_path(base_dir: Path, value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    if value.startswith("pkg://"):
        return value
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return str(p)


def _fallback_pkg_path(rel_path: str) -> str:
    return f"pkg://soarmMoce_sdk/{rel_path}"


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else _default_config_path()
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    base_dir = cfg_path.parent.resolve()

    # Resolve relative paths against config directory
    urdf_path = _resolve_path(base_dir, cfg.get("urdf", {}).get("path"))
    calib_path = _resolve_path(base_dir, cfg.get("calibration", {}).get("path"))
    mesh_path = _resolve_path(base_dir, cfg.get("meshes", {}).get("path"))

    if not urdf_path:
        urdf_path = _fallback_pkg_path("resources/urdf/soarmoce_urdf.urdf")
    if mesh_path is None:
        mesh_path = _fallback_pkg_path("resources/meshes")

    cfg.setdefault("urdf", {})["path"] = urdf_path
    cfg.setdefault("calibration", {})["path"] = calib_path or ""
    cfg.setdefault("meshes", {})["path"] = mesh_path

    return cfg
