#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Show the current SOARM URDF in RViz2 and/or Gazebo (ROS2 Humble).

Examples:
  python3 show_urdf_in_ros.py
  python3 show_urdf_in_ros.py --mode rviz
  python3 show_urdf_in_ros.py --mode gazebo
  python3 show_urdf_in_ros.py --urdf /abs/path/to/robot.urdf --mode both
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional


def _repo_root() -> Path:
    # .../SO102/Software/Master/Tools/show_urdf_in_ros.py -> repo root
    return Path(__file__).resolve().parents[4]


def _default_urdf_candidates() -> List[Path]:
    root = _repo_root()
    return [
        root / "SO102" / "Urdf" / "urdf" / "soarmoce_purple.urdf",
        root / "SO102" / "Urdf" / "urdf" / "soarmoce_urdf.urdf",
        root / "SO102" / "Soarm101" / "SO101" / "so101_new_calib.urdf",
        root / "SO102" / "Software" / "Master" / "so101.urdf",
    ]


def _select_urdf(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"URDF file not found: {p}")
        return p

    for p in _default_urdf_candidates():
        if p.exists():
            return p
    raise FileNotFoundError("No usable URDF found from default candidates.")


def _cmd_exists(name: str) -> bool:
    return shutil.which(name) is not None


def _run_quiet(cmd: List[str]) -> int:
    try:
        return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    except FileNotFoundError:
        return 127


def _ros2_pkg_exists(pkg: str) -> bool:
    return _run_quiet(["ros2", "pkg", "prefix", pkg]) == 0


class ProcGroup:
    def __init__(self) -> None:
        self.procs: List[subprocess.Popen] = []

    def start(self, cmd: List[str], name: str) -> subprocess.Popen:
        print(f"[START] {name}: {' '.join(cmd)}")
        p = subprocess.Popen(cmd, preexec_fn=os.setsid)
        self.procs.append(p)
        return p

    def stop_all(self) -> None:
        for p in reversed(self.procs):
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except Exception:
                    pass
        deadline = time.time() + 3.0
        for p in reversed(self.procs):
            if p.poll() is None:
                wait_left = max(0.0, deadline - time.time())
                try:
                    p.wait(timeout=wait_left)
                except Exception:
                    try:
                        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                    except Exception:
                        pass


def _write_rsp_params_yaml(urdf_path: Path, out_path: Path) -> None:
    xml = urdf_path.read_text(encoding="utf-8")
    xml_indented = "\n".join(f"      {line}" for line in xml.splitlines())
    text = (
        "/**:\n"
        "  ros__parameters:\n"
        "    use_sim_time: false\n"
        "    robot_description: |\n"
        f"{xml_indented}\n"
    )
    out_path.write_text(text, encoding="utf-8")


def _normalize_urdf_mesh_paths(src_urdf: Path, dst_urdf: Path, staged_mesh_dir: Optional[Path] = None) -> None:
    """
    Convert mesh filenames to ROS2-friendly URIs.
    - absolute/relative file path -> file://...
    - package://..., file://..., http(s)://... kept unchanged
    """
    tree = ET.parse(str(src_urdf))
    root = tree.getroot()
    converted = 0
    missing = 0

    if staged_mesh_dir is not None:
        staged_mesh_dir.mkdir(parents=True, exist_ok=True)

    staged = 0

    for i, mesh in enumerate(root.findall(".//mesh")):
        fn = mesh.attrib.get("filename", "").strip()
        if not fn:
            continue
        low = fn.lower()
        if low.startswith("package://") or low.startswith("file://") or low.startswith("http://") or low.startswith("https://"):
            continue
        p = Path(fn)
        if not p.is_absolute():
            p = (src_urdf.parent / p).resolve()
        else:
            p = p.resolve()
        if not p.exists():
            missing += 1
        target = p
        # Some setups are sensitive to upper-case .STL; stage as lower-case .stl.
        if staged_mesh_dir is not None and p.exists() and p.suffix == ".STL":
            staged_name = f"{i:03d}_{p.stem}.stl"
            staged_path = staged_mesh_dir / staged_name
            if not staged_path.exists():
                try:
                    os.symlink(str(p), str(staged_path))
                except Exception:
                    shutil.copy2(str(p), str(staged_path))
            target = staged_path
            staged += 1

        mesh.set("filename", target.as_uri())
        converted += 1

    tree.write(str(dst_urdf), encoding="utf-8", xml_declaration=True)
    print(
        f"[INFO] Normalized URDF mesh paths: converted={converted}, missing={missing}, staged_lowercase_stl={staged}"
    )


def _write_rviz2_config(out_path: Path) -> None:
    text = """Panels:
  - Class: rviz_common/Displays
    Name: Displays
  - Class: rviz_common/Selection
    Name: Selection
  - Class: rviz_common/Tool Properties
    Name: Tool Properties
  - Class: rviz_common/Views
    Name: Views
  - Class: rviz_common/Time
    Name: Time
Visualization Manager:
  Class: ""
  Displays:
    - Class: rviz_default_plugins/Grid
      Enabled: true
      Name: Grid
      Plane: XY
      Plane Cell Count: 10
      Cell Size: 0.05
      Color: 160; 160; 164
      Line Style:
        Value: Lines
        Line Width: 0.02
    - Class: rviz_default_plugins/RobotModel
      Enabled: true
      Name: RobotModel
      Description Source: Topic
      Description Topic:
        Value: /robot_description
      Visual Enabled: true
      Collision Enabled: false
  Enabled: true
  Global Options:
    Fixed Frame: base
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz_default_plugins/Interact
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
    - Class: rviz_default_plugins/FocusCamera
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 1.2
      Focal Point:
        X: 0
        Y: 0
        Z: 0.2
      Pitch: 0.45
      Yaw: 0.9
Window Geometry:
  Width: 1400
  Height: 900
"""
    out_path.write_text(text, encoding="utf-8")


def _require_ros2_tools(mode: str) -> None:
    if not _cmd_exists("ros2"):
        raise EnvironmentError(
            "Command `ros2` not found. Please source ROS2 first, e.g. "
            "`source /opt/ros/humble/setup.bash`."
        )

    required_pkgs = ["robot_state_publisher"]
    if mode in ("rviz", "both"):
        required_pkgs += ["joint_state_publisher_gui", "rviz2"]
    if mode in ("gazebo", "both"):
        required_pkgs += ["gazebo_ros"]

    missing = [p for p in required_pkgs if not _ros2_pkg_exists(p)]
    if missing:
        raise EnvironmentError(
            "Missing ROS2 package(s): "
            + ", ".join(missing)
            + ". Install with apt (Humble), e.g. "
            "`sudo apt install ros-humble-rviz2 ros-humble-joint-state-publisher-gui "
            "ros-humble-robot-state-publisher ros-humble-gazebo-ros-pkgs`."
        )


def _spawn_gazebo_entity(urdf_path: Path, model_name: str, retries: int = 20, delay_s: float = 1.0) -> None:
    cmd = [
        "ros2",
        "run",
        "gazebo_ros",
        "spawn_entity.py",
        "-entity",
        model_name,
        "-file",
        str(urdf_path),
    ]
    last_rc = 1
    for i in range(retries):
        print(f"[INFO] Gazebo spawn attempt {i + 1}/{retries}")
        rc = subprocess.run(cmd).returncode
        last_rc = rc
        if rc == 0:
            return
        time.sleep(delay_s)
    raise RuntimeError(f"Failed to spawn URDF into Gazebo after {retries} attempts (last rc={last_rc}).")


def main() -> int:
    parser = argparse.ArgumentParser(description="Show SOARM URDF in RViz2/Gazebo (ROS2 Humble).")
    parser.add_argument(
        "--mode",
        choices=["rviz", "gazebo", "both"],
        default="both",
        help="What to launch.",
    )
    parser.add_argument(
        "--urdf",
        default="",
        help="URDF absolute/relative path. Empty means auto-select current SOARM URDF.",
    )
    parser.add_argument(
        "--model-name",
        default="soarmoce",
        help="Gazebo model name when spawning.",
    )
    parser.add_argument(
        "--gazebo-spawn-wait-sec",
        type=float,
        default=5.0,
        help="Initial wait time before first spawn attempt after Gazebo launch.",
    )
    args = parser.parse_args()

    try:
        _require_ros2_tools(args.mode)
        urdf_path = _select_urdf(args.urdf if args.urdf else None)
    except Exception as e:
        print(f"[ERROR] {e}")
        return 2

    print(f"[INFO] URDF: {urdf_path}")
    group = ProcGroup()
    tmp_dir = Path(tempfile.mkdtemp(prefix="soarm_ros2_urdf_"))
    runtime_urdf = tmp_dir / "runtime_robot.urdf"
    staged_mesh_dir = tmp_dir / "meshes"
    rsp_params = tmp_dir / "rsp_params.yaml"
    rviz_cfg = tmp_dir / "rviz2_auto.rviz"
    _normalize_urdf_mesh_paths(urdf_path, runtime_urdf, staged_mesh_dir=staged_mesh_dir)
    _write_rsp_params_yaml(runtime_urdf, rsp_params)
    _write_rviz2_config(rviz_cfg)

    def _cleanup(_sig=None, _frame=None):
        print("\n[INFO] Shutting down...")
        group.stop_all()
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    try:
        group.start(
            [
                "ros2",
                "run",
                "robot_state_publisher",
                "robot_state_publisher",
                "--ros-args",
                "--params-file",
                str(rsp_params),
            ],
            "robot_state_publisher",
        )

        if args.mode in ("rviz", "both"):
            group.start(
                ["ros2", "run", "joint_state_publisher_gui", "joint_state_publisher_gui"],
                "joint_state_publisher_gui",
            )
            group.start(
                ["ros2", "run", "rviz2", "rviz2", "-d", str(rviz_cfg)],
                "rviz2",
            )

        if args.mode in ("gazebo", "both"):
            group.start(["ros2", "launch", "gazebo_ros", "gazebo.launch.py"], "gazebo_ros")
            time.sleep(max(0.0, float(args.gazebo_spawn_wait_sec)))
            _spawn_gazebo_entity(runtime_urdf, args.model_name)

        print("[INFO] Running. Press Ctrl+C to exit.")
        while True:
            for p in group.procs:
                rc = p.poll()
                if rc is not None and rc != 0:
                    raise RuntimeError(f"A subprocess exited with code {rc}.")
            time.sleep(0.5)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1
    finally:
        group.stop_all()
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
