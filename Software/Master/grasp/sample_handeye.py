#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified hand-eye sampling for eye-to-hand and eye-in-hand.
Saves paired T_base_ee (base->ee) and T_cam_board (board->cam) on key press.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import cv2

try:
    import pyrealsense2 as rs
except Exception as e:
    print("[ERROR] pyrealsense2 not available:", e)
    sys.exit(1)


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from real_test.robot_interface import RobotInterface, pose6_to_T  # noqa: E402


DEFAULT_SERIALS = {
    "eye_in_hand": "408322071062",
    "eye_to_hand": "050222072285",
}


def _parse_args():
    parser = argparse.ArgumentParser(description="Sample hand-eye calibration pairs (UR + D435).")
    parser.add_argument("--mode", choices=["eye_to_hand", "eye_in_hand"], default="eye_to_hand")
    parser.add_argument("--robot_mode", choices=["mock", "real"], default="mock")
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--enable_control", action="store_true", help="Use RTDEControlInterface when robot_mode=real.")
    parser.add_argument("--mock_pose6", default="", help="Comma-separated pose6 for mock.")
    parser.add_argument("--mock_path", default="", help="JSON file with list of pose6 for mock.")
    parser.add_argument("--mock_loop", action="store_true", help="Loop mock path poses.")
    parser.add_argument("--camera", default="external", help="Camera label (e.g., external/wrist).")
    parser.add_argument(
        "--serial",
        default="",
        help="RealSense serial to select when multiple cameras are connected (empty=auto by mode/camera).",
    )

    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)

    parser.add_argument("--dictionary", default="DICT_5X5_100")
    parser.add_argument("--squaresX", type=int, default=11)
    parser.add_argument("--squaresY", type=int, default=8)
    parser.add_argument("--squareLength", type=float, default=0.020)
    parser.add_argument("--markerLength", type=float, default=0.015)
    parser.add_argument(
        "--legacy_pattern",
        "--legacy",
        dest="legacy_pattern",
        default="auto",
        choices=["auto", "true", "false"],
        help="ChArUco legacy pattern (OpenCV <4.6). Use auto to try both.",
    )
    parser.add_argument("--auto_swap", action="store_true", help="Auto-try swapped squaresX/squaresY.")
    parser.add_argument("--axis_length", type=float, default=0.05)
    parser.add_argument("--pose_min_corners", type=int, default=4)
    parser.add_argument("--save_min_corners", type=int, default=20)

    parser.add_argument("--out", default="", help="Output JSONL path.")
    parser.add_argument("--no_raw_pose6", action="store_true", help="Do not save raw pose6.")
    parser.add_argument("--no_view", action="store_true", help="Disable OpenCV view window.")
    parser.add_argument("--debug", action="store_true", help="Print extra debug info.")
    return parser.parse_args()


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float64
    )


def _rot_angle_deg(R: np.ndarray) -> float:
    tr = float(np.trace(R))
    cos_theta = max(-1.0, min(1.0, (tr - 1.0) * 0.5))
    return float(np.degrees(np.arccos(cos_theta)))


def _get_dictionary(dict_name: str):
    if not hasattr(cv2.aruco, dict_name):
        raise ValueError(f"Unknown dictionary: {dict_name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))


def _create_charuco_board(squares_x, squares_y, square_len, marker_len, dictionary):
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        return cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_len, marker_len, dictionary)
    return cv2.aruco.CharucoBoard((squares_x, squares_y), square_len, marker_len, dictionary)


def _create_detector_params():
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        return cv2.aruco.DetectorParameters_create()
    return cv2.aruco.DetectorParameters()


def _create_charuco_params(K: np.ndarray, D: np.ndarray):
    if hasattr(cv2.aruco, "CharucoParameters_create"):
        cp = cv2.aruco.CharucoParameters_create()
    elif hasattr(cv2.aruco, "CharucoParameters"):
        cp = cv2.aruco.CharucoParameters()
    else:
        return None
    cp.cameraMatrix = K
    cp.distCoeffs = D
    if hasattr(cp, "tryRefineMarkers"):
        cp.tryRefineMarkers = True
    if hasattr(cp, "minMarkers"):
        cp.minMarkers = 1
    return cp


def _detect_pose_charuco(
    gray: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    board,
    dictionary,
    params,
    charuco_params,
    min_pose_corners: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int, Optional[List], Optional[np.ndarray]]:
    charuco_corners = charuco_ids = None
    marker_corners = marker_ids = None

    if hasattr(cv2.aruco, "CharucoDetector"):
        try:
            if charuco_params is not None:
                detector = cv2.aruco.CharucoDetector(
                    board, detectorParams=params, charucoParams=charuco_params
                )
            else:
                detector = cv2.aruco.CharucoDetector(board, detectorParams=params)
        except TypeError:
            if charuco_params is not None:
                detector = cv2.aruco.CharucoDetector(board, charuco_params, params)
            else:
                detector = cv2.aruco.CharucoDetector(board, params)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    else:
        if hasattr(cv2.aruco, "detectMarkers"):
            marker_corners, marker_ids, _rejected = cv2.aruco.detectMarkers(
                gray, dictionary, parameters=params
            )
        else:
            detector = cv2.aruco.ArucoDetector(dictionary, params)
            marker_corners, marker_ids, _rejected = detector.detectMarkers(gray)
        if marker_ids is None or len(marker_ids) == 0:
            return None, None, None, 0, marker_corners, marker_ids
        if hasattr(cv2.aruco, "interpolateCornersCharuco"):
            charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, board, cameraMatrix=K, distCoeffs=D
            )

    count = 0 if charuco_ids is None else len(charuco_ids)
    if count < min_pose_corners:
        return None, None, None, count, marker_corners, marker_ids

    if hasattr(cv2.aruco, "estimatePoseCharucoBoard"):
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, board, K, D, None, None
        )
        if not ok:
            return None, None, None, count, marker_corners, marker_ids
    else:
        if hasattr(board, "getChessboardCorners"):
            obj_pts = np.asarray(board.getChessboardCorners(), dtype=np.float64)
        else:
            obj_pts = np.asarray(board.chessboardCorners, dtype=np.float64)
        ids_flat = np.asarray(charuco_ids).flatten().astype(int)
        obj_sel = obj_pts[ids_flat].reshape(-1, 3)
        img_sel = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
        ok, rvec, tvec = cv2.solvePnP(obj_sel, img_sel, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, None, None, count, marker_corners, marker_ids

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return rvec, tvec, T, count, marker_corners, marker_ids


def _capture_color_frame(pipeline, align) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = pipeline.wait_for_frames()
    if align is not None:
        frames = align.process(frames)
    color_frame = frames.get_color_frame()
    if not color_frame:
        raise RuntimeError("Failed to get color frame")
    color = np.asanyarray(color_frame.get_data())
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    K = np.array(
        [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    D = np.array(intr.coeffs, dtype=np.float64).reshape(1, -1)
    return color, K, D


def _select_serial(mode: str, camera: str, serial: str) -> str:
    if serial:
        return serial
    cam = (camera or "").lower()
    if cam in ("external", "ext", "eye_to_hand"):
        return DEFAULT_SERIALS.get("eye_to_hand", "")
    if cam in ("wrist", "eye_in_hand", "hand", "inner"):
        return DEFAULT_SERIALS.get("eye_in_hand", "")
    return DEFAULT_SERIALS.get(mode, "")


def _parse_pose6(pose6_str: str) -> Optional[List[float]]:
    if not pose6_str:
        return None
    parts = [p.strip() for p in pose6_str.split(",")]
    if len(parts) != 6:
        raise ValueError("mock_pose6 must have 6 comma-separated values")
    return [float(p) for p in parts]


def main():
    args = _parse_args()

    if args.out:
        out_path = args.out
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join("real_test", f"handeye_samples_{args.mode}_{stamp}.jsonl")
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    mock_pose6 = _parse_pose6(args.mock_pose6) if args.mock_pose6 else None
    robot = RobotInterface(
        mode=args.robot_mode,
        ip=args.ip,
        mock_pose6=mock_pose6,
        mock_path=args.mock_path or None,
        mock_loop=args.mock_loop,
        enable_control=args.enable_control,
    )

    try:
        dictionary = _get_dictionary(args.dictionary)
    except ValueError as e:
        print("[ERROR]", e)
        sys.exit(1)

    params = _create_detector_params()

    has_legacy = hasattr(cv2.aruco.CharucoBoard, "setLegacyPattern")
    if args.legacy_pattern == "auto" and has_legacy:
        legacy_modes = [False, True]
    elif args.legacy_pattern == "true":
        legacy_modes = [True]
    else:
        legacy_modes = [False]

    square_pairs = [(args.squaresX, args.squaresY)]
    if args.auto_swap and (args.squaresX != args.squaresY):
        square_pairs.append((args.squaresY, args.squaresX))

    pipeline = rs.pipeline()
    config = rs.config()
    selected_serial = _select_serial(args.mode, args.camera, args.serial)
    if selected_serial:
        config.enable_device(selected_serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    device = profile.get_device()
    try:
        camera_serial = device.get_info(rs.camera_info.serial_number)
    except Exception:
        camera_serial = ""
    try:
        camera_name = device.get_info(rs.camera_info.name)
    except Exception:
        camera_name = ""

    # Warm up
    for _ in range(10):
        pipeline.wait_for_frames()

    last_T_base_ee = None
    last_T_cam_board = None

    if not args.no_view:
        cv2.namedWindow("charuco", cv2.WINDOW_NORMAL)

    try:
        with open(out_path, "a", encoding="utf-8") as f:
            while True:
                color, K, D = _capture_color_frame(pipeline, align)
                gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                charuco_params = _create_charuco_params(K, D)

                best = {
                    "rvec": None,
                    "tvec": None,
                    "T": None,
                    "count": 0,
                    "corners": None,
                    "ids": None,
                    "sx": args.squaresX,
                    "sy": args.squaresY,
                    "legacy": None,
                }

                for sx, sy in square_pairs:
                    for legacy in legacy_modes:
                        board = _create_charuco_board(
                            sx, sy, args.squareLength, args.markerLength, dictionary
                        )
                        if has_legacy:
                            board.setLegacyPattern(bool(legacy))
                        rvec, tvec, T, count, corners, ids = _detect_pose_charuco(
                            gray,
                            K,
                            D,
                            board,
                            dictionary,
                            params,
                            charuco_params,
                            args.pose_min_corners,
                        )
                        if args.debug:
                            print(
                                f"[DEBUG] squaresX={sx} squaresY={sy} legacy={legacy} corners={count}"
                            )
                        if count > best["count"]:
                            best.update(
                                {
                                    "rvec": rvec,
                                    "tvec": tvec,
                                    "T": T,
                                    "count": count,
                                    "corners": corners,
                                    "ids": ids,
                                    "sx": sx,
                                    "sy": sy,
                                    "legacy": bool(legacy) if has_legacy else None,
                                }
                            )
                        if T is not None:
                            best.update(
                                {
                                    "rvec": rvec,
                                    "tvec": tvec,
                                    "T": T,
                                    "count": count,
                                    "corners": corners,
                                    "ids": ids,
                                    "sx": sx,
                                    "sy": sy,
                                    "legacy": bool(legacy) if has_legacy else None,
                                }
                            )
                            break
                    if best["T"] is not None:
                        break

                annotated = color.copy()
                if best["ids"] is not None and len(best["ids"]) > 0:
                    cv2.aruco.drawDetectedMarkers(annotated, best["corners"], best["ids"])
                if best["T"] is not None:
                    cv2.drawFrameAxes(
                        annotated, K, D, best["rvec"], best["tvec"], args.axis_length
                    )

                msg = f"corners={best['count']} (save>= {args.save_min_corners})"
                cv2.putText(
                    annotated,
                    msg,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if best["count"] >= args.save_min_corners else (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    "press 's' to save, 'q' to quit",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                if not args.no_view:
                    cv2.imshow("charuco", annotated)
                    key = cv2.waitKey(1) & 0xFF
                else:
                    key = 255

                if key == ord("q"):
                    break

                if key == ord("s"):
                    if best["T"] is None or best["count"] < args.save_min_corners:
                        print(
                            f"[SKIP] corners={best['count']} (need >= {args.save_min_corners})"
                        )
                        continue
                    pose6 = robot.get_tcp_pose6()
                    print(f"[TCP] pose6: {list(map(float, pose6))}")
                    T_base_ee = pose6_to_T(pose6)
                    T_cam_board = best["T"]

                    ts = time.time()
                    rec = {
                        "timestamp": ts,
                        "mode": args.mode,
                        "camera": args.camera,
                        "requested_serial": selected_serial,
                        "camera_name": camera_name,
                        "camera_serial": camera_serial,
                        "T_base_ee": T_base_ee.tolist(),
                        "T_cam_board": T_cam_board.tolist(),
                        "n_charuco_corners": int(best["count"]),
                        "corners": int(best["count"]),
                    }
                    if not args.no_raw_pose6:
                        rec["raw_pose6"] = list(map(float, pose6))
                    rec["squaresX"] = int(best["sx"])
                    rec["squaresY"] = int(best["sy"])
                    rec["legacyPattern"] = best["legacy"]

                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    print(f"[OK] Saved sample to {out_path} (corners={best['count']})")

                    if last_T_base_ee is not None and last_T_cam_board is not None:
                        A = np.linalg.inv(last_T_base_ee) @ T_base_ee
                        B = np.linalg.inv(last_T_cam_board) @ T_cam_board
                        ang_A = _rot_angle_deg(A[:3, :3])
                        ang_B = _rot_angle_deg(B[:3, :3])
                        print(f"[CHECK] rot angle A={ang_A:.3f} deg, B={ang_B:.3f} deg")

                    last_T_base_ee = T_base_ee
                    last_T_cam_board = T_cam_board
    finally:
        pipeline.stop()
        robot.close()
        if not args.no_view:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
