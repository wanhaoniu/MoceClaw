# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json


def load_calibration_json(path: str) -> Dict[str, Any]:
    fpath = Path(path)
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)
