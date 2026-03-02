#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R


def rpy_to_quat(roll, pitch, yaw):
    # PyBullet uses (x,y,z,w)
    quat = R.from_euler("xyz", [roll, pitch, yaw]).as_quat()
    return quat.tolist()


def quat_angle_error(q_target_xyzw, q_actual_xyzw):
    r_t = R.from_quat(q_target_xyzw)
    r_a = R.from_quat(q_actual_xyzw)
    r_rel = r_t * r_a.inv()
    return r_rel.magnitude()


def sanitize_limits(ll, ul):
    if (not np.isfinite(ll)) or (not np.isfinite(ul)) or (ll > ul):
        return -np.pi, np.pi
    return float(ll), float(ul)


def build_joint_lists(robot_id, excluded_joint_names=None):
    movable_joint_indices = []
    movable_joint_names = []
    lower_limits = []
    upper_limits = []
    joint_ranges = []
    rest_poses = []
    excluded = set(excluded_joint_names or [])

    num_joints = p.getNumJoints(robot_id)
    for ji in range(num_joints):
        info = p.getJointInfo(robot_id, ji)
        joint_type = info[2]
        joint_name = info[1].decode("utf-8")
        ll = info[8]
        ul = info[9]

        if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC) and joint_name not in excluded:
            movable_joint_indices.append(ji)
            movable_joint_names.append(joint_name)

            ll, ul = sanitize_limits(ll, ul)

            lower_limits.append(float(ll))
            upper_limits.append(float(ul))
            joint_ranges.append(float(ul - ll) if ul > ll else float(2 * np.pi))
            rest_poses.append(float(0.5 * (ll + ul)))

    return (movable_joint_indices, movable_joint_names,
            lower_limits, upper_limits, joint_ranges, rest_poses)


def auto_detect_ee_link(robot_id, excluded_joint_names=None):
    num_joints = p.getNumJoints(robot_id)
    parents = set()
    all_links = set(range(num_joints))
    excluded = set(excluded_joint_names or [])
    for j in range(num_joints):
        info = p.getJointInfo(robot_id, j)
        parent_idx = info[16]
        if parent_idx >= 0:
            parents.add(parent_idx)

    leaf_links = sorted(list(all_links - parents))
    filtered_leaf_links = []
    for li in leaf_links:
        info = p.getJointInfo(robot_id, li)
        parent_joint_name = info[1].decode("utf-8")
        if parent_joint_name in excluded:
            continue
        filtered_leaf_links.append(li)

    if len(filtered_leaf_links) > 0:
        return filtered_leaf_links[-1]
    if len(leaf_links) == 0:
        return num_joints - 1 if num_joints > 0 else -1
    return leaf_links[-1]


def find_link_index_by_name(robot_id, link_name):
    num_joints = p.getNumJoints(robot_id)
    for j in range(num_joints):
        info = p.getJointInfo(robot_id, j)
        child_link_name = info[12].decode("utf-8")
        if child_link_name == link_name:
            return j
    return None


def find_joint_index_by_name(robot_id, joint_name):
    num_joints = p.getNumJoints(robot_id)
    for j in range(num_joints):
        info = p.getJointInfo(robot_id, j)
        if info[1].decode("utf-8") == joint_name:
            return j
    return None


def set_robot_joints(robot_id, movable_joint_indices, q):
    for idx, ji in enumerate(movable_joint_indices):
        p.resetJointState(robot_id, ji, float(q[idx]))


def get_link_pose_world(robot_id, link_index):
    st = p.getLinkState(robot_id, link_index, computeForwardKinematics=True)
    pos = st[4]  # worldLinkFramePosition
    orn = st[5]  # worldLinkFrameOrientation
    return np.array(pos), np.array(orn)


def world_to_base(robot_id, pos_w, quat_w):
    base_pos_w, base_quat_w = p.getBasePositionAndOrientation(robot_id)
    inv_p, inv_q = p.invertTransform(base_pos_w, base_quat_w)
    pos_b, quat_b = p.multiplyTransforms(inv_p, inv_q, pos_w.tolist(), quat_w.tolist())
    return np.array(pos_b), np.array(quat_b)


def quat_to_rpy(quat_xyzw):
    return R.from_quat(quat_xyzw).as_euler("xyz", degrees=False)


def draw_frame(pos, quat_xyzw, axis_len=0.12, life_time=0.0, line_width=2):
    rot = R.from_quat(quat_xyzw).as_matrix()
    o = np.array(pos)
    x = o + rot[:, 0] * axis_len
    y = o + rot[:, 1] * axis_len
    z = o + rot[:, 2] * axis_len
    p.addUserDebugLine(o, x, [1, 0, 0], lineWidth=line_width, lifeTime=life_time)
    p.addUserDebugLine(o, y, [0, 1, 0], lineWidth=line_width, lifeTime=life_time)
    p.addUserDebugLine(o, z, [0, 0, 1], lineWidth=line_width, lifeTime=life_time)


def solve_ik_once(robot_id, ee_link_index, target_pos_w, target_quat_w,
                  movable_joint_indices, lower_limits, upper_limits, joint_ranges, rest_poses,
                  max_iters=200, residual_threshold=1e-4):
    q_full = p.calculateInverseKinematics(
        robot_id,
        ee_link_index,
        targetPosition=target_pos_w,
        targetOrientation=target_quat_w,
        lowerLimits=lower_limits,
        upperLimits=upper_limits,
        jointRanges=joint_ranges,
        restPoses=rest_poses,
        maxNumIterations=max_iters,
        residualThreshold=residual_threshold
    )

    q = np.array(q_full[:len(movable_joint_indices)], dtype=np.float64)

    set_robot_joints(robot_id, movable_joint_indices, q)
    p.stepSimulation()
    pos_fk_w, quat_fk_w = get_link_pose_world(robot_id, ee_link_index)

    pos_err = float(np.linalg.norm(pos_fk_w - np.array(target_pos_w)))
    ori_err = float(quat_angle_error(np.array(target_quat_w), np.array(quat_fk_w)))

    return q, pos_fk_w, quat_fk_w, pos_err, ori_err


def parse_pose_input(line, deg=True):
    vals = [float(x) for x in line.strip().split()]
    if len(vals) != 6:
        raise ValueError("需要 6 个数：x y z roll pitch yaw")
    x, y, z, rr, pp, yy = vals
    if deg:
        rr, pp, yy = np.deg2rad([rr, pp, yy])
    return (float(x), float(y), float(z)), (float(rr), float(pp), float(yy))


def print_ee_pose_base(robot_id, ee_idx, use_deg=False, prefix="[EE@BASE]"):
    pos_w, quat_w = get_link_pose_world(robot_id, ee_idx)
    pos_b, quat_b = world_to_base(robot_id, pos_w, quat_w)
    rpy_b = quat_to_rpy(quat_b)
    if use_deg:
        rpy_show = np.rad2deg(rpy_b)
        unit = "deg"
    else:
        rpy_show = rpy_b
        unit = "rad"

    print(f"{prefix} xyz = ({pos_b[0]:.6f}, {pos_b[1]:.6f}, {pos_b[2]:.6f})  "
          f"rpy = ({rpy_show[0]:.6f}, {rpy_show[1]:.6f}, {rpy_show[2]:.6f}) [{unit}]")
    return pos_b, rpy_b


def teach_mode(robot_id, ee_idx,
               movable_joint_indices, movable_joint_names,
               lower_limits, upper_limits, joint_ranges, rest_poses,
               gripper_joint_index, gripper_lower, gripper_upper, gripper_init,
               use_deg: bool,
               pos_tol: float, ori_tol: float,
               teach_pos_range: float = 0.4):
    """
    Teach 模式：用滑块拖动末端目标位姿（xyz+rpy），机器人 IK 跟随。
    按 P：打印当前末端在基座系下 xyz+rpy
    按 Q：退出
    """
    # 初始用当前末端位姿作为 target
    ee_pos_w, ee_quat_w = get_link_pose_world(robot_id, ee_idx)
    ee_rpy_w = quat_to_rpy(ee_quat_w)

    # 给 xyz 设置一个“围绕当前值”的滑动范围
    x0, y0, z0 = ee_pos_w.tolist()
    pr = float(teach_pos_range)
    x_id = p.addUserDebugParameter("target_x", x0 - pr, x0 + pr, x0)
    y_id = p.addUserDebugParameter("target_y", y0 - pr, y0 + pr, y0)
    z_id = p.addUserDebugParameter("target_z", max(0.0, z0 - pr), z0 + pr, z0)

    if use_deg:
        r0, p0_, y0_ = np.rad2deg(ee_rpy_w).tolist()
        rr_id = p.addUserDebugParameter("target_roll(deg)", -180.0, 180.0, r0)
        pp_id = p.addUserDebugParameter("target_pitch(deg)", -180.0, 180.0, p0_)
        yy_id = p.addUserDebugParameter("target_yaw(deg)", -180.0, 180.0, y0_)
    else:
        r0, p0_, y0_ = ee_rpy_w.tolist()
        rr_id = p.addUserDebugParameter("target_roll(rad)", -np.pi, np.pi, r0)
        pp_id = p.addUserDebugParameter("target_pitch(rad)", -np.pi, np.pi, p0_)
        yy_id = p.addUserDebugParameter("target_yaw(rad)", -np.pi, np.pi, y0_)

    grip_id = None
    if gripper_joint_index is not None:
        grip_id = p.addUserDebugParameter("gripper(rad)", gripper_lower, gripper_upper, gripper_init)
        p.resetJointState(robot_id, gripper_joint_index, float(gripper_init))

    p.addUserDebugText(
        "TEACH MODE: sliders control EE target; gripper is independent; Press P=print, Q=quit",
        [0, 0, 1.2],
        textColorRGB=[1, 1, 1],
        textSize=1.2,
        lifeTime=0
    )

    last_print_time = 0.0

    while True:
        # keyboard events
        keys = p.getKeyboardEvents()
        if ord('q') in keys and (keys[ord('q')] & p.KEY_WAS_TRIGGERED):
            break
        if ord('Q') in keys and (keys[ord('Q')] & p.KEY_WAS_TRIGGERED):
            break

        # read target sliders
        tx = p.readUserDebugParameter(x_id)
        ty = p.readUserDebugParameter(y_id)
        tz = p.readUserDebugParameter(z_id)

        tr = p.readUserDebugParameter(rr_id)
        tp_ = p.readUserDebugParameter(pp_id)
        ty_ = p.readUserDebugParameter(yy_id)

        if use_deg:
            tr, tp_, ty_ = np.deg2rad([tr, tp_, ty_])

        target_pos_w = [tx, ty, tz]
        target_quat_w = rpy_to_quat(tr, tp_, ty_)

        # IK follow
        q, pos_fk_w, quat_fk_w, pos_err, ori_err = solve_ik_once(
            robot_id, ee_idx, target_pos_w, target_quat_w,
            movable_joint_indices, lower_limits, upper_limits, joint_ranges, rest_poses
        )

        if grip_id is not None:
            grip_target = float(p.readUserDebugParameter(grip_id))
            p.resetJointState(robot_id, gripper_joint_index, grip_target)
        else:
            grip_target = None

        # visualization
        p.removeAllUserDebugItems()
        # re-add message
        p.addUserDebugText(
            "TEACH MODE: sliders control EE target; gripper is independent; Press P=print, Q=quit",
            [0, 0, 1.2],
            textColorRGB=[1, 1, 1],
            textSize=1.2,
            lifeTime=0
        )
        draw_frame(target_pos_w, target_quat_w, axis_len=0.15, life_time=0.0, line_width=3)
        p.addUserDebugText("TARGET", target_pos_w, [1, 1, 1], textSize=1.2, lifeTime=0.0)

        draw_frame(pos_fk_w, quat_fk_w, axis_len=0.12, life_time=0.0, line_width=2)
        p.addUserDebugText("EE", pos_fk_w.tolist(), [1, 1, 0], textSize=1.2, lifeTime=0.0)

        reachable = (pos_err <= pos_tol) and (ori_err <= ori_tol)
        status = "REACHABLE" if reachable else "UNREACHABLE"
        p.addUserDebugText(
            f"{status}  pos_err={pos_err:.4f}m  ori_err={ori_err:.4f}rad",
            [0, 0, 1.1],
            textColorRGB=[0, 1, 0] if reachable else [1, 0.2, 0.2],
            textSize=1.1,
            lifeTime=0
        )
        if grip_target is not None:
            p.addUserDebugText(
                f"gripper={grip_target:.4f} rad",
                [0, 0, 1.0],
                textColorRGB=[0.8, 0.9, 1.0],
                textSize=1.0,
                lifeTime=0
            )

        # press P to print current EE pose in BASE frame
        if (ord('p') in keys and (keys[ord('p')] & p.KEY_WAS_TRIGGERED)) or \
           (ord('P') in keys and (keys[ord('P')] & p.KEY_WAS_TRIGGERED)):

            # 防止连续触发刷屏（可选）
            now = time.time()
            if now - last_print_time > 0.2:
                last_print_time = now
                print_ee_pose_base(robot_id, ee_idx, use_deg=use_deg, prefix="[TEACH PRINT]")

        p.stepSimulation()
        time.sleep(1.0 / 240.0)


def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--urdf", default='/home/sunyuan/Code/Soarm_Moce/SO102/Urdf/urdf/soarmoce_urdf.urdf', help="URDF 文件路径（本地路径）")
    #白色模型
    # ap.add_argument("--urdf", default='/home/sunyuan/Code/Soarm_Moce/SO102/Soarm101/SO101/so101_new_calib.urdf', help="URDF 文件路径（本地路径）")
    # 紫灰配色
    ap.add_argument("--urdf", default='/home/sunyuan/Code/Soarm_Moce/SO102/Urdf/urdf/soarmoce_purple.urdf', help="URDF 文件路径（本地路径）")

    ap.add_argument("--ee_link", default=None, help="末端 link 名称（可选；也支持填 joint 名称，如 wrist_roll）")
    ap.add_argument("--gripper_joint", default="gripper", help="夹爪关节名（独立控制，不参与IK）")
    ap.add_argument("--gripper_init", type=float, default=None, help="夹爪初始角度（rad，不填则取关节中值）")
    ap.add_argument("--fixed_base", default=1, help="将基座固定（机械臂通常需要）")

    ap.add_argument("--mode", choices=["ik", "teach"], default="teach",
                    help="ik: 命令行输入目标位姿求逆解；teach: 通过滑块拖动末端并读取 xyz+rpy")

    ap.add_argument("--deg", action="store_true", help="输入/显示 rpy 用“度”（默认建议用这个）")
    ap.add_argument("--rad", action="store_true", help="输入/显示 rpy 用“弧度”")

    ap.add_argument("--pos_tol", type=float, default=0.01, help="可达判断：位置误差阈值（m）")
    ap.add_argument("--ori_tol", type=float, default=0.15, help="可达判断：姿态误差阈值（rad）")

    ap.add_argument("--teach_pos_range", type=float, default=0.4,
                    help="teach 模式下 xyz 滑块范围：围绕初始末端位置的 ±range（m）")

    args = ap.parse_args()

    use_deg = True
    if args.rad:
        use_deg = False
    if args.deg:
        use_deg = True

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetDebugVisualizerCamera(cameraDistance=1.4, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.3])

    # p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    robot_id = p.loadURDF(args.urdf, basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1],
                          useFixedBase=bool(args.fixed_base))

    gripper_joint_index = find_joint_index_by_name(robot_id, args.gripper_joint)
    gripper_lower = None
    gripper_upper = None
    gripper_init = None
    if gripper_joint_index is None:
        print(f"[WARN] 没有找到夹爪关节: {args.gripper_joint}。将按纯 IK 机械臂运行。")
    else:
        gripper_info = p.getJointInfo(robot_id, gripper_joint_index)
        if gripper_info[2] not in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            print(f"[WARN] 夹爪关节 {args.gripper_joint} 不是可动关节，忽略独立控制。")
            gripper_joint_index = None
        else:
            gripper_lower, gripper_upper = sanitize_limits(gripper_info[8], gripper_info[9])
            if args.gripper_init is None:
                gripper_init = 0.5 * (gripper_lower + gripper_upper)
            else:
                gripper_init = float(np.clip(args.gripper_init, gripper_lower, gripper_upper))
            p.resetJointState(robot_id, gripper_joint_index, gripper_init)

    excluded_joint_names = [args.gripper_joint] if gripper_joint_index is not None else []
    movable_joint_indices, movable_joint_names, lower_limits, upper_limits, joint_ranges, rest_poses = build_joint_lists(
        robot_id, excluded_joint_names=excluded_joint_names
    )
    if len(movable_joint_indices) == 0:
        print("[ERROR] 没有检测到机械臂可动关节。请检查URDF和 gripper_joint 配置。")
        return

    # EE link
    if args.ee_link is not None:
        ee_idx = find_link_index_by_name(robot_id, args.ee_link)
        if ee_idx is None:
            # 兼容把 joint 名称当作 ee 输入（如: wrist_roll）
            ee_idx = find_joint_index_by_name(robot_id, args.ee_link)
        if ee_idx is None:
            print(f"[WARN] 没找到 link 名称: {args.ee_link}，改用自动检测。")
            ee_idx = auto_detect_ee_link(robot_id, excluded_joint_names=excluded_joint_names)
    else:
        # 默认优先使用 wrist_roll 对应的末端（joint index 也是 child link index）
        wrist_roll_idx = find_joint_index_by_name(robot_id, "wrist_roll")
        if wrist_roll_idx is not None:
            ee_idx = wrist_roll_idx
            print("[INFO] 默认 EE 使用 wrist_roll。")
        else:
            ee_idx = auto_detect_ee_link(robot_id, excluded_joint_names=excluded_joint_names)

    print("==== 可动关节列表 ====")
    for i, (ji, jn) in enumerate(zip(movable_joint_indices, movable_joint_names)):
        print(f"{i:02d}: jointIndex={ji:02d}  name={jn:>20s}  limit=[{lower_limits[i]: .3f}, {upper_limits[i]: .3f}]")
    info = p.getJointInfo(robot_id, ee_idx)
    print(f"\n[INFO] EE link index = {ee_idx}")
    print(f"[INFO] EE link name  = {info[12].decode('utf-8')}")
    if gripper_joint_index is not None:
        print(f"[INFO] GRIPPER joint = {args.gripper_joint} (index={gripper_joint_index}, "
              f"limit=[{gripper_lower:.3f}, {gripper_upper:.3f}], init={gripper_init:.3f})")

    # -------- TEACH MODE --------
    if args.mode == "teach":
        print("\n[TEACH MODE] 用滑块拖动末端目标位姿；按 P 打印末端在基座系下 xyz+rpy；按 Q 退出。\n")
        teach_mode(
            robot_id, ee_idx,
            movable_joint_indices, movable_joint_names,
            lower_limits, upper_limits, joint_ranges, rest_poses,
            gripper_joint_index, gripper_lower, gripper_upper, gripper_init,
            use_deg=use_deg,
            pos_tol=args.pos_tol,
            ori_tol=args.ori_tol,
            teach_pos_range=args.teach_pos_range
        )
        p.disconnect()
        return

    # -------- IK MODE (original) --------
    print("\n==== IK 输入格式 ====")
    if use_deg:
        print("请输入：x y z roll pitch yaw   (rpy单位=度)")
    else:
        print("请输入：x y z roll pitch yaw   (rpy单位=弧度)")
    print("输入 q 退出；输入 read 可读取当前末端 xyz+rpy(base)。")
    if gripper_joint_index is not None:
        print("输入 g <rad> 单独设置夹爪角度；输入 open / close 快速开合。")
    print("")

    gripper_target = gripper_init

    while True:
        p.removeAllUserDebugItems()

        line = input("Target pose> ").strip()
        if line.lower() in ("q", "quit", "exit"):
            break
        if line.lower() in ("read", "r"):
            print_ee_pose_base(robot_id, ee_idx, use_deg=use_deg, prefix="[READ]")
            if gripper_joint_index is not None:
                print(f"[READ] gripper = {gripper_target:.6f} rad")
            continue
        if gripper_joint_index is not None and line.lower().startswith("g "):
            try:
                g_cmd = float(line.split(maxsplit=1)[1])
            except Exception:
                print("[输入错误] gripper 指令格式应为: g <rad>")
                continue
            gripper_target = float(np.clip(g_cmd, gripper_lower, gripper_upper))
            p.resetJointState(robot_id, gripper_joint_index, gripper_target)
            print(f"[GRIPPER] set to {gripper_target:.6f} rad")
            continue
        if gripper_joint_index is not None and line.lower() == "open":
            gripper_target = float(gripper_upper)
            p.resetJointState(robot_id, gripper_joint_index, gripper_target)
            print(f"[GRIPPER] open -> {gripper_target:.6f} rad")
            continue
        if gripper_joint_index is not None and line.lower() == "close":
            gripper_target = float(gripper_lower)
            p.resetJointState(robot_id, gripper_joint_index, gripper_target)
            print(f"[GRIPPER] close -> {gripper_target:.6f} rad")
            continue

        try:
            target_pos, target_rpy = parse_pose_input(line, deg=use_deg)
            target_quat = rpy_to_quat(*target_rpy)
        except Exception as e:
            print(f"[输入错误] {e}")
            continue

        draw_frame(target_pos, target_quat, axis_len=0.15, life_time=0.0, line_width=3)
        p.addUserDebugText("TARGET", target_pos, textColorRGB=[1, 1, 1], textSize=1.2, lifeTime=0.0)

        q, pos_fk_w, quat_fk_w, pos_err, ori_err = solve_ik_once(
            robot_id, ee_idx, target_pos, target_quat,
            movable_joint_indices, lower_limits, upper_limits, joint_ranges, rest_poses
        )
        if gripper_joint_index is not None:
            p.resetJointState(robot_id, gripper_joint_index, gripper_target)

        draw_frame(pos_fk_w, quat_fk_w, axis_len=0.12, life_time=0.0, line_width=2)
        p.addUserDebugText("EE", pos_fk_w.tolist(), textColorRGB=[1, 1, 0], textSize=1.2, lifeTime=0.0)

        reachable = (pos_err <= args.pos_tol) and (ori_err <= args.ori_tol)

        print("\n==== IK 结果 ====")
        for name, val in zip(movable_joint_names, q.tolist()):
            print(f"{name:>25s} : {val: .6f} rad")
        if gripper_joint_index is not None:
            print(f"{args.gripper_joint:>25s} : {gripper_target: .6f} rad (independent)")
        print(f"\n[CHECK] pos_err = {pos_err:.6f} m   (tol={args.pos_tol})")
        print(f"[CHECK] ori_err = {ori_err:.6f} rad (tol={args.ori_tol})")

        if not reachable:
            print("\n[不可达] 请重新输入一个位姿。\n")
            for _ in range(60):
                p.stepSimulation()
                time.sleep(1.0 / 240.0)
            continue

        print("[可达] 已置于该逆解姿态。你也可以输入 read 读取当前末端 xyz+rpy(base)。\n")
        for _ in range(240):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    p.disconnect()


if __name__ == "__main__":
    main()
