# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
import xml.etree.ElementTree as ET
import numpy as np


def _parse_floats(attr: Optional[str], n: int, default: float = 0.0) -> np.ndarray:
    if not attr:
        return np.full((n,), default, dtype=float)
    parts = attr.replace(",", " ").split()
    if len(parts) < n:
        parts = parts + [str(default)] * (n - len(parts))
    return np.array([float(x) for x in parts[:n]], dtype=float)


@dataclass
class Joint:
    name: str
    jtype: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    limit_lower: float
    limit_upper: float


class RobotModel:
    def __init__(self, urdf_path: Path, base_link: Optional[str] = None, end_link: Optional[str] = None):
        self.urdf_path = Path(urdf_path)
        self.joints: Dict[str, Joint] = {}
        self.links: List[str] = []
        self._child_map: Dict[str, List[Joint]] = {}
        self._parent_map: Dict[str, Joint] = {}

        self._load_urdf()
        self.base_link, self.end_link, self.chain_joints = self._build_chain(base_link, end_link)
        self.active_joints = [j for j in self.chain_joints if j.jtype in ("revolute", "continuous", "prismatic")]
        self.dof = len(self.active_joints)
        self.joint_names = [j.name for j in self.active_joints]
        self.joint_limits = [(j.limit_lower, j.limit_upper) for j in self.active_joints]

    def _load_urdf(self) -> None:
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")
        tree = ET.parse(str(self.urdf_path))
        root = tree.getroot()

        self.links = [link.get("name") for link in root.findall("link") if link.get("name")]

        for joint_node in root.findall("joint"):
            jname = joint_node.get("name")
            jtype = joint_node.get("type", "fixed")
            parent_node = joint_node.find("parent")
            child_node = joint_node.find("child")
            if jname is None or parent_node is None or child_node is None:
                continue
            parent = parent_node.get("link")
            child = child_node.get("link")
            if parent is None or child is None:
                continue

            origin_node = joint_node.find("origin")
            if origin_node is not None:
                xyz = _parse_floats(origin_node.get("xyz"), 3, 0.0)
                rpy = _parse_floats(origin_node.get("rpy"), 3, 0.0)
            else:
                xyz = np.zeros(3, dtype=float)
                rpy = np.zeros(3, dtype=float)

            axis_node = joint_node.find("axis")
            axis = _parse_floats(axis_node.get("xyz") if axis_node is not None else None, 3, 0.0)
            if np.linalg.norm(axis) < 1e-12:
                axis = np.array([1.0, 0.0, 0.0], dtype=float)
            else:
                axis = axis / np.linalg.norm(axis)

            limit_node = joint_node.find("limit")
            if jtype in ("revolute", "prismatic"):
                if limit_node is None:
                    lower, upper = -math.inf, math.inf
                else:
                    lower = float(limit_node.get("lower", "-inf"))
                    upper = float(limit_node.get("upper", "inf"))
            elif jtype == "continuous":
                lower, upper = -math.inf, math.inf
            else:
                lower, upper = 0.0, 0.0

            joint = Joint(
                name=jname,
                jtype=jtype,
                parent=parent,
                child=child,
                origin_xyz=xyz,
                origin_rpy=rpy,
                axis=axis,
                limit_lower=lower,
                limit_upper=upper,
            )
            self.joints[jname] = joint
            self._child_map.setdefault(parent, []).append(joint)
            self._parent_map[child] = joint

    def _infer_base_link(self) -> str:
        children = set(self._parent_map.keys())
        roots = [link for link in self.links if link not in children]
        if not roots:
            roots = list(self._child_map.keys())
        if not roots:
            raise RuntimeError("Failed to infer base link from URDF")
        return roots[0]

    def _build_chain(self, base_link: Optional[str], end_link: Optional[str]) -> Tuple[str, str, List[Joint]]:
        base = base_link or self._infer_base_link()
        if base not in self.links:
            raise ValueError(f"Base link not found: {base}")

        if end_link:
            chain = self._path_from_base(base, end_link)
            if chain is None:
                raise ValueError(f"End link not reachable from base: {end_link}")
            return base, end_link, chain

        best_chain: List[Joint] = []
        best_end = base

        def dfs(link: str, joints_path: List[Joint]):
            nonlocal best_chain, best_end
            if link not in self._child_map:
                if len(joints_path) > len(best_chain):
                    best_chain = list(joints_path)
                    best_end = link
                return
            for j in self._child_map[link]:
                dfs(j.child, joints_path + [j])

        dfs(base, [])
        if not best_chain:
            raise RuntimeError("Failed to build a joint chain from base")
        return base, best_end, best_chain

    def _path_from_base(self, base: str, end: str) -> Optional[List[Joint]]:
        stack = [(base, [])]
        visited = set()
        while stack:
            link, path = stack.pop()
            if link == end:
                return path
            if link in visited:
                continue
            visited.add(link)
            for j in self._child_map.get(link, []):
                stack.append((j.child, path + [j]))
        return None
