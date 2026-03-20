#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline hand-eye calibration (eye-to-hand or eye-in-hand) from JSONL samples.
Uses OpenCV calibrateHandEye and outputs the requested extrinsic.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np
import cv2


def _parse_args():
    parser = argparse.ArgumentParser(description="Solve hand-eye from JSONL samples.")
    parser.add_argument("--samples", required=True, help="JSONL samples path.")
    parser.add_argument("--mode", choices=["eye_to_hand", "eye_in_hand"], default="eye_to_hand")
    parser.add_argument(
        "--method",
        default="TSAI",
        choices=["TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"],
        help="OpenCV hand-eye method.",
    )
    parser.add_argument(
        "--eye_to_hand_convention",
        choices=["btg", "gtb", "auto"],
        default="auto",
        help=(
            "How to build gripper2base for eye_to_hand. "
            "btg uses T_base_ee directly, gtb uses inv(T_base_ee), auto tries both and picks lower error."
        ),
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output JSON path. Default depends on --mode.",
    )
    parser.add_argument("--report", default="", help="Optional JSON report path.")
    return parser.parse_args()


def _load_samples(path: str) -> List[dict]:
    samples: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return samples


def _to_T(mat) -> np.ndarray:
    T = np.asarray(mat, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("Expected 4x4 matrix")
    return T


def _split_rt(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R = T[:3, :3].astype(np.float64)
    t = T[:3, 3].reshape(3, 1).astype(np.float64)
    return R, t


def _quat_from_R(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _R_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    R = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return R


def _avg_rotation(Rs: List[np.ndarray]) -> np.ndarray:
    if len(Rs) == 1:
        return Rs[0]
    q0 = _quat_from_R(Rs[0])
    qs = []
    for R in Rs:
        q = _quat_from_R(R)
        if np.dot(q, q0) < 0.0:
            q = -q
        qs.append(q)
    q_mean = np.mean(np.stack(qs, axis=0), axis=0)
    q_mean = q_mean / np.linalg.norm(q_mean)
    return _R_from_quat(q_mean)


def _rot_angle_deg(R: np.ndarray) -> float:
    tr = float(np.trace(R))
    cos_theta = max(-1.0, min(1.0, (tr - 1.0) * 0.5))
    return float(np.degrees(np.arccos(cos_theta)))


def _stats(values: List[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
    }


def _collect_rt(samples: List[dict], invert_gripper: bool) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], int]:
    R_gripper2base: List[np.ndarray] = []
    t_gripper2base: List[np.ndarray] = []
    R_target2cam: List[np.ndarray] = []
    t_target2cam: List[np.ndarray] = []
    used = 0
    for s in samples:
        if "T_base_ee" not in s or "T_cam_board" not in s:
            continue
        T_base_ee = _to_T(s["T_base_ee"])
        T_cam_board = _to_T(s["T_cam_board"])
        T_gripper2base = np.linalg.inv(T_base_ee) if invert_gripper else T_base_ee
        Rg, tg = _split_rt(T_gripper2base)
        Rt, tt = _split_rt(T_cam_board)
        R_gripper2base.append(Rg)
        t_gripper2base.append(tg)
        R_target2cam.append(Rt)
        t_target2cam.append(tt)
        used += 1
    return R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, used


def _solve_handeye(
    R_gripper2base: List[np.ndarray],
    t_gripper2base: List[np.ndarray],
    R_target2cam: List[np.ndarray],
    t_target2cam: List[np.ndarray],
    method: int,
) -> np.ndarray:
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, method=method
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_cam2gripper
    T[:3, 3] = np.asarray(t_cam2gripper).reshape(3)
    return T


def _eval_eye_to_hand(T_base_cam_ext: np.ndarray, samples: List[dict]) -> Tuple[dict, np.ndarray, np.ndarray]:
    T_pred_list = []
    for s in samples:
        if "T_base_ee" not in s or "T_cam_board" not in s:
            continue
        T_base_ee = _to_T(s["T_base_ee"])
        T_cam_board = _to_T(s["T_cam_board"])
        T_ee_board_pred = np.linalg.inv(T_base_ee) @ T_base_cam_ext @ T_cam_board
        T_pred_list.append(T_ee_board_pred)
    Rs_pred = [T[:3, :3] for T in T_pred_list]
    ts_pred = [T[:3, 3] for T in T_pred_list]
    R_pred_mean = _avg_rotation(Rs_pred)
    t_pred_mean = np.mean(np.stack(ts_pred, axis=0), axis=0)

    trans_errs = []
    rot_errs = []
    for T in T_pred_list:
        dt = np.linalg.norm(T[:3, 3] - t_pred_mean)
        dR = R_pred_mean.T @ T[:3, :3]
        da = _rot_angle_deg(dR)
        trans_errs.append(float(dt))
        rot_errs.append(float(da))
    report = {
        "trans_error_m": _stats(trans_errs),
        "rot_error_deg": _stats(rot_errs),
        "n_used": len(T_pred_list),
        "T_ee_board_mean": {"R": R_pred_mean.tolist(), "t": t_pred_mean.tolist()},
    }
    return report, R_pred_mean, t_pred_mean


def _eval_eye_in_hand(T_ee_cam: np.ndarray, samples: List[dict]) -> Tuple[dict, np.ndarray, np.ndarray]:
    T_pred_list = []
    for s in samples:
        if "T_base_ee" not in s or "T_cam_board" not in s:
            continue
        T_base_ee = _to_T(s["T_base_ee"])
        T_cam_board = _to_T(s["T_cam_board"])
        T_base_board_pred = T_base_ee @ T_ee_cam @ T_cam_board
        T_pred_list.append(T_base_board_pred)
    Rs_pred = [T[:3, :3] for T in T_pred_list]
    ts_pred = [T[:3, 3] for T in T_pred_list]
    R_pred_mean = _avg_rotation(Rs_pred)
    t_pred_mean = np.mean(np.stack(ts_pred, axis=0), axis=0)

    trans_errs = []
    rot_errs = []
    for T in T_pred_list:
        dt = np.linalg.norm(T[:3, 3] - t_pred_mean)
        dR = R_pred_mean.T @ T[:3, :3]
        da = _rot_angle_deg(dR)
        trans_errs.append(float(dt))
        rot_errs.append(float(da))
    report = {
        "trans_error_m": _stats(trans_errs),
        "rot_error_deg": _stats(rot_errs),
        "n_used": len(T_pred_list),
        "T_base_board_mean": {"R": R_pred_mean.tolist(), "t": t_pred_mean.tolist()},
    }
    return report, R_pred_mean, t_pred_mean


def main():
    args = _parse_args()
    if not args.out:
        if args.mode == "eye_in_hand":
            args.out = os.path.join("real_test", "T_ee_cam_wrist.json")
        else:
            args.out = os.path.join("real_test", "T_base_cam_ext.json")
    samples = _load_samples(args.samples)
    if len(samples) < 3:
        print("[ERROR] Need at least 3 samples.")
        sys.exit(1)

    # Mode semantics:
    # - eye_to_hand: camera fixed, board moves with ee => output T_base_cam_ext
    # - eye_in_hand: camera on ee, board fixed        => output T_ee_cam_wrist
    #
    # Conventions used in this script (as saved by the sampler):
    # - T_base_ee is ^bT_e (ee -> base; pose of ee in base)
    # - T_cam_board is ^cT_b (board -> cam)
    #
    method_map = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    method = method_map[args.method]

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out = {
        "mode": args.mode,
        "method": args.method,
    }

    if args.mode == "eye_in_hand":
        Rg, tg, Rt, tt, used = _collect_rt(samples, invert_gripper=False)
        if used < 3:
            print("[ERROR] Not enough valid samples with T_base_ee and T_cam_board.")
            sys.exit(1)
        T_X = _solve_handeye(Rg, tg, Rt, tt, method)
        T_ee_cam = T_X
        report, _, _ = _eval_eye_in_hand(T_ee_cam, samples)
        report["mode"] = args.mode
        out["n_samples"] = used
        out["T_ee_cam_wrist"] = T_ee_cam.tolist()
    else:
        candidates = []
        if args.eye_to_hand_convention in ("btg", "auto"):
            Rg, tg, Rt, tt, used = _collect_rt(samples, invert_gripper=False)
            if used < 3:
                print("[ERROR] Not enough valid samples with T_base_ee and T_cam_board.")
                sys.exit(1)
            T_X = _solve_handeye(Rg, tg, Rt, tt, method)
            T_base_cam_list = []
            for s in samples:
                if "T_base_ee" not in s:
                    continue
                T_base_ee = _to_T(s["T_base_ee"])
                T_base_cam_list.append(T_base_ee @ T_X)
            Rs = [T[:3, :3] for T in T_base_cam_list]
            ts = [T[:3, 3] for T in T_base_cam_list]
            R_mean = _avg_rotation(Rs)
            t_mean = np.mean(np.stack(ts, axis=0), axis=0)
            T_base_cam_ext = np.eye(4, dtype=np.float64)
            T_base_cam_ext[:3, :3] = R_mean
            T_base_cam_ext[:3, 3] = t_mean
            rep, _, _ = _eval_eye_to_hand(T_base_cam_ext, samples)
            candidates.append(("btg", used, T_base_cam_ext, rep))

        if args.eye_to_hand_convention in ("gtb", "auto"):
            Rg, tg, Rt, tt, used = _collect_rt(samples, invert_gripper=True)
            if used < 3:
                print("[ERROR] Not enough valid samples with T_base_ee and T_cam_board.")
                sys.exit(1)
            T_X = _solve_handeye(Rg, tg, Rt, tt, method)
            T_base_cam_ext = T_X
            rep, _, _ = _eval_eye_to_hand(T_base_cam_ext, samples)
            candidates.append(("gtb", used, T_base_cam_ext, rep))

        def _score(rep: dict) -> Tuple[float, float]:
            return (rep["rot_error_deg"]["mean"], rep["trans_error_m"]["mean"])

        if not candidates:
            print("[ERROR] No candidates computed for eye_to_hand.")
            sys.exit(1)
        candidates.sort(key=lambda c: _score(c[3]))
        chosen_conv, used, T_base_cam_ext, report = candidates[0]
        report["mode"] = args.mode
        report["eye_to_hand_convention"] = chosen_conv
        out["n_samples"] = used
        out["eye_to_hand_convention"] = chosen_conv
        out["T_base_cam_ext"] = T_base_cam_ext.tolist()
        if len(candidates) > 1:
            out["eye_to_hand_candidates"] = {
                c[0]: {
                    "trans_error_m": c[3]["trans_error_m"],
                    "rot_error_deg": c[3]["rot_error_deg"],
                    "n_used": c[3]["n_used"],
                }
                for c in candidates
            }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    print("[OK] Saved:", args.out)
    print("[REPORT] trans error (m):", report["trans_error_m"])
    print("[REPORT] rot error (deg):", report["rot_error_deg"])


if __name__ == "__main__":
    main()
