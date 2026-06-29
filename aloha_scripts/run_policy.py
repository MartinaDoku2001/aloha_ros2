#!/usr/bin/env python3
"""
run_policy.py — Deploy a trained ACT policy on the real ALOHA robot.

Usage (from repo root):
    python snn_aloha/aloha_scripts/run_policy.py \
        --ckpt_dir checkpoints/aloha_pass_strawberry_0_kl_0 \
        --num_rollouts 5

Options:
    --temporal_agg      Use temporal aggregation (smoother, queries policy every step)
    --eval_ckpt NAME    Checkpoint filename inside ckpt_dir (default: policy_last.ckpt)
    --episode_len N     Override episode length from task config
    --save_video        Save first-camera video of each episode to --video_dir
    --video_dir PATH    Where to save videos (default: ./eval_videos)
    --task_name NAME    Override task name from checkpoint config

Press 'q' during an episode to end it early. Ctrl+C to exit.
"""

import argparse
import os
import sys
import pickle
import time
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pynput import keyboard

# ── Import core/models so 'from act import ModernACTPolicy' resolves correctly
REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
CORE_MODELS = REPO_ROOT / "core" / "models"
if str(CORE_MODELS) not in sys.path:
    sys.path.insert(0, str(CORE_MODELS))

from act import ModernACTPolicy

# ── Local imports (same directory as this script) ────────────────────────────
from constants import DT, FPS, TASK_CONFIGS, START_ARM_POSE
from constants import PUPPET_GRIPPER_JOINT_CLOSE, PUPPET_GRIPPER_JOINT_OPEN
from robot_utils import move_arms, move_grippers, torque_on
from real_env import make_real_env

from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    get_interbotix_global_node,
    robot_startup,
)


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def load_policy(ckpt_dir: str, eval_ckpt: str | None) -> tuple:
    """Load run_config, build policy, load weights, load dataset stats."""
    config_path = os.path.join(ckpt_dir, "run_config.pkl")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"run_config.pkl not found in {ckpt_dir}")
    with open(config_path, "rb") as f:
        run_config = pickle.load(f)

    policy_config = run_config["policy_config"]
    policy = ModernACTPolicy(policy_config)

    ckpt_name = eval_ckpt or "policy_last.ckpt"
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    status = policy.deserialize(torch.load(ckpt_path, map_location="cpu"))
    print(f"  Loaded: {ckpt_path}  status={status}")

    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    return run_config, policy, stats


# ── Image preprocessing ───────────────────────────────────────────────────────
def images_to_tensor(obs_images: dict, camera_names: list) -> torch.Tensor:
    """Stack camera images into a (1, N_cams, C, 224, 224) float32 CUDA tensor."""
    imgs = []
    for cam in camera_names:
        img = obs_images[cam]                  # (H, W, C) uint8
        imgs.append(img.transpose(2, 0, 1).astype(np.float32) / 255.0)
    arr = np.stack(imgs, axis=0)               # (N_cams, C, H, W)
    t = torch.from_numpy(arr).float().cuda().unsqueeze(0)  # (1, N_cams, C, H, W)
    if t.shape[-2:] != torch.Size([224, 224]):
        _B, _N, _C, _H, _W = t.shape
        t = F.interpolate(
            t.view(_B * _N, _C, _H, _W), size=(224, 224),
            mode='bilinear', align_corners=False
        ).view(_B, _N, _C, 224, 224)
    return t


# ── Opening ceremony ──────────────────────────────────────────────────────────
def opening_ceremony(env):
    """Move puppet arms to start pose then wait for Enter."""
    env.puppet_bot_left.core.robot_reboot_motors("single", "gripper", True)
    env.puppet_bot_right.core.robot_reboot_motors("single", "gripper", True)
    env.puppet_bot_left.core.robot_set_operating_modes("group", "arm", "position")
    env.puppet_bot_left.core.robot_set_operating_modes("single", "gripper", "current_based_position")
    env.puppet_bot_right.core.robot_set_operating_modes("group", "arm", "position")
    env.puppet_bot_right.core.robot_set_operating_modes("single", "gripper", "current_based_position")

    torque_on(env.puppet_bot_left)
    torque_on(env.puppet_bot_right)

    pose_with_flip = list(START_ARM_POSE[:6])
    pose_with_flip[1] = -pose_with_flip[1]
    pose_with_flip[2] = -pose_with_flip[2]

    move_arms(
        [env.puppet_bot_left, env.puppet_bot_right],
        [pose_with_flip, pose_with_flip],
        move_time=2.0,
    )
    move_grippers(
        [env.puppet_bot_left, env.puppet_bot_right],
        [PUPPET_GRIPPER_JOINT_CLOSE, PUPPET_GRIPPER_JOINT_CLOSE],
        move_time=0.5,
    )

    print("\nRobot at start pose. Press ENTER to begin (Ctrl+C to quit)...", flush=True)
    input()


# ── Temporal aggregation ──────────────────────────────────────────────────────
def temporal_agg_action(buffer: np.ndarray, t: int, k: float = 0.01) -> np.ndarray:
    """Average actions predicted for step t, weighting newer predictions higher."""
    col = buffer[:t + 1, t, :]
    populated = col[np.any(col != 0, axis=-1)]
    if len(populated) == 0:
        return col[-1]
    # reverse so index -1 = newest → gets highest weight exp(-k*0)
    exp_w = np.exp(-k * np.arange(len(populated))[::-1])
    exp_w /= exp_w.sum()
    return (populated * exp_w[:, None]).sum(axis=0)


# ── Episode loop ──────────────────────────────────────────────────────────────
_break_episode = False

def _on_press(key):
    global _break_episode
    try:
        if key.char == 'q':
            print("\n[INFO] 'q' pressed — ending episode early.")
            _break_episode = True
    except AttributeError:
        pass


def run_one_episode(env, policy, stats, camera_names, episode_len,
                    num_queries, temporal_agg, save_video):
    global _break_episode
    _break_episode = False

    qpos_mean   = stats["qpos_mean"]
    qpos_std    = stats["qpos_std"]
    action_mean = stats["action_mean"]
    action_std  = stats["action_std"]

    query_frequency = 1 if temporal_agg else num_queries

    agg_buffer = None
    if temporal_agg:
        agg_buffer = np.zeros((episode_len, episode_len + num_queries, 14))

    video_frames = []
    dt_history   = []
    action_chunk = None

    listener = keyboard.Listener(on_press=_on_press)
    listener.start()
    print(">> Press 'q' to end episode early. <<")

    # get fresh observation without moving the robot
    ts = env.reset(fake=True)

    with torch.inference_mode():
        for t in range(episode_len):
            if _break_episode:
                break

            t0 = time.time()

            # ── Observe ──────────────────────────────────────────────────────
            obs = ts.observation
            qpos_raw  = obs['qpos'].copy()
            qpos_norm = (qpos_raw - qpos_mean) / (qpos_std + 1e-8)
            qpos_t    = torch.from_numpy(qpos_norm).float().cuda().unsqueeze(0)
            curr_image = images_to_tensor(obs['images'], camera_names)

            if save_video:
                video_frames.append(obs['images'][camera_names[0]].copy())

            # ── Policy query ──────────────────────────────────────────────────
            if t % query_frequency == 0:
                out = policy(qpos_t, curr_image)
                action_chunk = out["action"] if isinstance(out, dict) else out
                # shape: (1, num_queries, 14)

            if temporal_agg:
                chunk_np = action_chunk[0, :, :14].cpu().numpy()
                agg_buffer[t, t:t + num_queries] = chunk_np
                raw_action = temporal_agg_action(agg_buffer, t)
            else:
                step_in_chunk = t % query_frequency
                raw_action = action_chunk[0, step_in_chunk, :14].cpu().numpy()

            # ── Denormalize and apply ─────────────────────────────────────────
            action = raw_action * action_std[:14] + action_mean[:14]

            if t % 30 == 0:
                print(f"  t={t:>4}  grip_L={action[6]:.4f}  grip_R={action[13]:.4f}"
                      f"  shoulder_L={action[1]:.3f}  elbow_L={action[2]:.3f}")

            # env.step() takes 14-dim master-convention action and handles the
            # shoulder/elbow sign flip for the puppet bots internally.
            ts = env.step(action)

            elapsed = time.time() - t0
            dt_history.append(elapsed)
            if elapsed < DT:
                time.sleep(DT - elapsed)

    listener.stop()

    if dt_history:
        print(f"  Avg freq: {1.0 / np.mean(dt_history):.1f} Hz  ({len(dt_history)} steps)")

    return video_frames


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    run_config, policy, stats = load_policy(args.ckpt_dir, args.eval_ckpt)

    task_name    = args.task_name or run_config["task"]
    temporal_agg = args.temporal_agg or run_config.get("temporal_agg", False)
    num_queries  = run_config["policy_config"].get("num_queries", 100)

    if task_name not in TASK_CONFIGS:
        raise KeyError(f"Task '{task_name}' not in TASK_CONFIGS. "
                       f"Available: {list(TASK_CONFIGS.keys())}")
    task_cfg     = TASK_CONFIGS[task_name]
    camera_names = task_cfg["camera_names"]
    episode_len  = args.episode_len or task_cfg["episode_len"]

    print(f"\nTask:         {task_name}")
    print(f"Episode len:  {episode_len}")
    print(f"Num queries:  {num_queries}")
    print(f"Temporal agg: {temporal_agg}")
    print(f"Cameras:      {camera_names}\n")

    policy.cuda()
    policy.eval()

    # ── Robot init ──────────────────────────────────────────────────────────
    try:
        global_node = get_interbotix_global_node()
    except Exception:
        global_node = None
    if global_node is None:
        global_node = create_interbotix_global_node()

    env = make_real_env(init_node=False, setup_robots=True)
    robot_startup(global_node)

    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    executor = MultiThreadedExecutor()
    for rec in [env.recorder_left, env.recorder_right, env.image_recorder]:
        if isinstance(rec, Node):
            executor.add_node(rec)
    threading.Thread(target=executor.spin, daemon=True).start()

    # ── Rollouts ────────────────────────────────────────────────────────────
    n_success = 0
    for rollout_id in range(args.num_rollouts):
        print(f"\n{'='*60}")
        print(f"Rollout {rollout_id + 1} / {args.num_rollouts}")
        print(f"{'='*60}")

        try:
            opening_ceremony(env)
        except KeyboardInterrupt:
            print("\nAborted.")
            break

        try:
            video_frames = run_one_episode(
                env, policy, stats, camera_names,
                episode_len, num_queries, temporal_agg,
                save_video=args.save_video,
            )
        except KeyboardInterrupt:
            print("\nEpisode interrupted.")
            video_frames = []

        if args.save_video and video_frames:
            try:
                import imageio
                os.makedirs(args.video_dir, exist_ok=True)
                vpath = os.path.join(args.video_dir,
                                     f"rollout_{task_name}_{rollout_id + 1:02d}.mp4")
                imageio.mimsave(vpath, video_frames, fps=FPS)
                print(f"  Video saved: {vpath}")
            except Exception as e:
                print(f"  Warning: could not save video: {e}")

        result = input("\nEpisode result — success? [y/n/q to quit]: ").strip().lower()
        if result == 'y':
            n_success += 1
            print("  Marked as SUCCESS")
        elif result == 'q':
            print("Exiting early.")
            rollout_id += 1
            break
        else:
            print("  Marked as FAILURE")

    total = rollout_id + 1
    if total > 0:
        print(f"\nFinal: {n_success}/{total} = {100.0 * n_success / total:.1f}% success")


def parse_args():
    p = argparse.ArgumentParser(
        description="Run a trained ACT policy on the real ALOHA robot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt_dir",     required=True,
                   help="Checkpoint directory (must contain run_config.pkl, dataset_stats.pkl)")
    p.add_argument("--task_name",    default=None,
                   help="Override task name from checkpoint")
    p.add_argument("--num_rollouts", type=int, default=5)
    p.add_argument("--eval_ckpt",    default=None,
                   help="Checkpoint filename (default: policy_last.ckpt)")
    p.add_argument("--temporal_agg", action="store_true")
    p.add_argument("--episode_len",  type=int, default=None)
    p.add_argument("--save_video",   action="store_true")
    p.add_argument("--video_dir",    default="./eval_videos")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
    import os as _os
    _os.exit(0)
