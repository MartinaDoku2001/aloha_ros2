#!/usr/bin/env python3
"""
run_policy.py — Deploy a trained ACT or SACT policy on the real ALOHA robot.

v3.3 — temporal ensembling + chunk smoothing to kill motion jitter.

WHAT WENT WRONG IN v2
---------------------
--async_infer used a Python thread. A launch-bound SNN (norse, T=32) spends
most of its forward pass in Python-level dispatch holding the GIL, so the
"background" thread starves the control loop. Chunks then arrive older than
num_queries steps, the executor clamps to "hold last action", and a job is
always in flight so the GIL never frees. Result: one action, long wait, repeat.

WHAT CHANGED
------------
  * --async_mode process   True parallelism. Inference runs in a separate
                           process with its own CUDA context and its own GIL.
                           This is the only async that works at 675 ms.
  * --async_mode thread    The old behaviour. Kept, but warns. Only sane if
                           your forward is <50 ms.
  * --async_mode off       Synchronous. Predictable periodic hitch. Boring and
                           reliable — use this to confirm the policy works.
  * Expired chunks are now REJECTED, not swapped in. A chunk whose age exceeds
    num_queries is useless; accepting it caused the permanent hold.
  * Per-episode diagnostics: control freq, inference latency, chunks accepted
    vs rejected, and how many steps were spent holding a stale action.

RECOMMENDED, in order
---------------------
1. Prove the policy works at all, accepting a visible hitch every 2 s:

    python3 snn_aloha/aloha_scripts/run_policy.py \
        --ckpt_dir checkpoints/aloha_pass_strawberry_0_tb_bs_4_kl_10_LIF_T32 \
        --num_rollouts 10 --save_video --video_dir VIDEO \
        --query_frequency 60 --blend_len 10 --async_mode off

2. Then try real async:

    ... --query_frequency 30 --blend_len 8 --async_mode process

3. Then actually fix the 675 ms. Everything above is a workaround.

Press 'q' during an episode to end it early. Ctrl+C to exit.
"""

import argparse
import os
import sys
import pickle
import time
import math
import threading
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
MODELS_ACT  = REPO_ROOT / "models" / "ACT"
MODELS_SACT = REPO_ROOT / "models" / "SACT"
SCRIPT_DIR  = Path(__file__).resolve().parent


def _ensure_on_path(p) -> None:
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)


# ══════════════════════════════════════════════════════════════════════════════
#  POLICY DETECTION / LOADING
# ══════════════════════════════════════════════════════════════════════════════
def _detect_policy_class(run_config: dict, state_dict: dict,
                         override: str | None = None) -> str:
    if override:
        u = override.upper()
        return "SACT" if ("SACT" in u or "SNN" in u or "SPIK" in u) else "ACT"

    pc = run_config.get("policy_config", {})
    for src in (run_config, pc):
        v = src.get("policy_class") or src.get("policy_type")
        if isinstance(v, str):
            u = v.upper()
            if "SACT" in u or "SNN" in u or "SPIK" in u:
                return "SACT"
            if "ACT" in u:
                return "ACT"

    joined = " ".join(state_dict.keys())
    if (".cross_attn." in joined) or (".self_attn.q_proj." in joined) or (".ff.linear" in joined):
        return "SACT"
    return "ACT"


def load_policy(ckpt_dir: str, eval_ckpt: str | None,
                policy_class: str | None = None) -> tuple:
    config_path = os.path.join(ckpt_dir, "run_config.pkl")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"run_config.pkl not found in {ckpt_dir}")
    with open(config_path, "rb") as f:
        run_config = pickle.load(f)
    policy_config = run_config["policy_config"]

    ckpt_name  = eval_ckpt or "policy_last.ckpt"
    ckpt_path  = os.path.join(ckpt_dir, ckpt_name)
    state_dict = torch.load(ckpt_path, map_location="cpu")

    kind = _detect_policy_class(run_config, state_dict, policy_class)
    print(f"  Policy class: {kind}")

    if kind == "SACT":
        _ensure_on_path(MODELS_ACT)
        _ensure_on_path(MODELS_SACT)
        from sact import ModernSACTPolicy
        policy = ModernSACTPolicy(policy_config)
    else:
        _ensure_on_path(MODELS_ACT)
        from act import ModernACTPolicy
        policy = ModernACTPolicy(policy_config)

    status = policy.deserialize(state_dict)
    print(f"  Loaded: {ckpt_path}  status={status}")

    with open(os.path.join(ckpt_dir, "dataset_stats.pkl"), "rb") as f:
        stats = pickle.load(f)

    return run_config, policy, stats


# ══════════════════════════════════════════════════════════════════════════════
#  SNN / GPU TUNING
# ══════════════════════════════════════════════════════════════════════════════
def enable_gpu_fast_paths(torch_threads: int | None = None) -> None:
    torch.backends.cudnn.benchmark = True
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    # A launch-bound SNN gets no benefit from intra-op parallelism, but its
    # worker threads DO fight the ROS executor for cores. Fewer is usually
    # faster here.
    if torch_threads is not None and torch_threads > 0:
        try:
            torch.set_num_threads(int(torch_threads))
            torch.set_num_interop_threads(1)
        except Exception:
            pass


def detect_snn_lib(model) -> str:
    mods = " ".join({type(m).__module__ for m in model.modules()})
    if "norse" in mods:
        return "norse"
    if "spikingjelly" in mods:
        return "spikingjelly"
    return "unknown"


def apply_snn_speedups(policy, step_mode=None, backend=None, quiet=False) -> str:
    m = getattr(policy, "model", policy)
    lib = detect_snn_lib(m)
    if not quiet:
        print(f"  [snn] framework: {lib}")

    if lib == "norse":
        if (step_mode or backend) and not quiet:
            print("  [snn] --sj_* flags do nothing on norse — ignoring.")
        try:
            torch._C._jit_set_texpr_fuser_enabled(True)
            torch._C._jit_override_can_fuse_on_gpu(True)
        except Exception:
            pass
        return lib

    if lib == "spikingjelly" and (step_mode or backend):
        try:
            from spikingjelly.activation_based import functional as sj_f, neuron as sj_n
        except Exception as e:
            print(f"  [snn] spikingjelly unavailable ({e})")
            return lib
        if step_mode:
            try:
                sj_f.set_step_mode(m, step_mode)
                if not quiet:
                    print(f"  [snn] step_mode -> '{step_mode}'")
            except Exception as e:
                print(f"  [snn] set_step_mode failed: {e}")
        if backend:
            try:
                sj_f.set_backend(m, backend, instance=sj_n.BaseNode)
                if not quiet:
                    print(f"  [snn] backend -> '{backend}'")
            except Exception as e:
                print(f"  [snn] set_backend failed: {e}")
    return lib


_TSTEP_ATTRS = ("T", "n_steps", "num_steps", "timesteps", "time_steps",
                "n_timesteps", "snn_T", "steps")


def set_snn_timesteps(policy, new_T: int, old_T: int | None = None,
                      quiet=False) -> int:
    """Best-effort inference-time reduction of the SNN unroll length.
    CHANGES POLICY OUTPUT — validate offline against full T first."""
    m = getattr(policy, "model", policy)
    changed = 0
    for name, mod in m.named_modules():
        for attr in _TSTEP_ATTRS:
            v = getattr(mod, attr, None)
            if isinstance(v, int) and not isinstance(v, bool):
                if old_T is not None and v != old_T:
                    continue
                if old_T is None and not (1 <= v <= 512):
                    continue
                try:
                    setattr(mod, attr, int(new_T))
                    if not quiet:
                        print(f"  [snn] {name or '<root>'}.{attr}: {v} -> {new_T}")
                    changed += 1
                except Exception:
                    pass
    if changed == 0 and not quiet:
        print(f"  [snn] --snn_T: nothing matched (old_T={old_T}). "
              f"T is likely a local in forward(); edit detr_svae.py.")
    return changed


def _maybe_reset_snn(policy) -> None:
    m = getattr(policy, "model", policy)
    reset = getattr(m, "reset", None) or getattr(m, "reset_states", None)
    if callable(reset):
        try:
            reset(); return
        except Exception:
            pass
    try:
        from spikingjelly.activation_based import functional as _sj
        _sj.reset_net(m)
    except Exception:
        pass


_AMP_DTYPE = {"off": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def prep_inputs(qpos_np: np.ndarray, img_np: np.ndarray,
                size=(224, 224)) -> tuple:
    """qpos (D,) float, img (N,H,W,C) uint8  ->  CUDA tensors for the policy."""
    qpos_t = torch.from_numpy(qpos_np).float().cuda().unsqueeze(0)
    t = torch.from_numpy(img_np).cuda()
    t = t.permute(0, 3, 1, 2).float().div_(255.0)
    if t.shape[-2:] != torch.Size(size):
        t = F.interpolate(t, size=size, mode="bilinear", align_corners=False)
    return qpos_t, t.unsqueeze(0)


class ForwardRunner:
    """Reset + optional autocast + forward, returning numpy chunks."""

    def __init__(self, policy, amp: str = "off", action_dim: int = 14):
        self.policy = policy
        self.amp_dtype = _AMP_DTYPE.get(amp, None)
        self.action_dim = action_dim

    def __call__(self, qpos_t, image_t):
        with torch.inference_mode():
            _maybe_reset_snn(self.policy)
            if self.amp_dtype is not None:
                with torch.autocast("cuda", dtype=self.amp_dtype):
                    out = self.policy(qpos_t, image_t)
            else:
                out = self.policy(qpos_t, image_t)

        if isinstance(out, dict):
            act, pad = out["action"], out.get("is_pad", None)
        else:
            act, pad = out, None

        chunk = act[0, :, :self.action_dim].float().cpu().numpy()
        pad_np = None
        if pad is not None:
            p = pad[0].float().cpu().numpy()
            pad_np = p.reshape(p.shape[0])
        return chunk, pad_np


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE ENGINES
# ══════════════════════════════════════════════════════════════════════════════
class SyncEngine:
    """Blocking inference in the control loop. Predictable periodic hitch."""

    def __init__(self, runner: ForwardRunner):
        self.runner = runner
        self._pending = None
        self.last_latency = 0.0

    def can_submit(self) -> bool:
        return True

    def submit(self, qpos_np, img_np, t) -> bool:
        t0 = time.time()
        qpos_t, img_t = prep_inputs(qpos_np, img_np)
        chunk, pad = self.runner(qpos_t, img_t)
        self.last_latency = time.time() - t0
        self._pending = (chunk, pad, t)
        return True

    def poll(self):
        r, self._pending = self._pending, None
        return r

    def drain(self):
        self._pending = None

    def close(self):
        pass


class ThreadEngine:
    """Background thread. WARNING: the GIL makes this useless for launch-bound
    models. Kept only for fast (<50 ms) policies."""

    def __init__(self, runner: ForwardRunner):
        self.runner = runner
        self._lock = threading.Lock()
        self._pending = None
        self._busy = False
        self.last_latency = 0.0

    def can_submit(self) -> bool:
        with self._lock:
            return not self._busy

    def submit(self, qpos_np, img_np, t) -> bool:
        with self._lock:
            if self._busy:
                return False
            self._busy = True

        def _work():
            try:
                t0 = time.time()
                qpos_t, img_t = prep_inputs(qpos_np, img_np)
                chunk, pad = self.runner(qpos_t, img_t)
                self.last_latency = time.time() - t0
                with self._lock:
                    self._pending = (chunk, pad, t)
            except Exception as e:
                print(f"\n  [thread] inference failed: {type(e).__name__}: {e}")
            finally:
                with self._lock:
                    self._busy = False

        threading.Thread(target=_work, daemon=True).start()
        return True

    def poll(self):
        with self._lock:
            r, self._pending = self._pending, None
        return r

    def drain(self):
        with self._lock:
            self._pending = None

    def close(self):
        pass


def _proc_worker(ckpt_dir, eval_ckpt, policy_class, amp, action_dim,
                 snn_T, snn_T_from, sj_step_mode, sj_backend,
                 state_dim, n_cams, torch_threads, req_q, res_q, ready_q):
    """
    Child process. Owns its own CUDA context and its own GIL, so the parent's
    control loop is genuinely unaffected by inference cost.

    Deliberately imports nothing ROS-related.
    """
    try:
        enable_gpu_fast_paths(torch_threads)
        run_config, policy, _ = load_policy(ckpt_dir, eval_ckpt, policy_class)
        policy.cuda().eval()
        apply_snn_speedups(policy, sj_step_mode, sj_backend, quiet=True)
        if snn_T is not None:
            set_snn_timesteps(policy, snn_T, snn_T_from, quiet=True)
        runner = ForwardRunner(policy, amp=amp, action_dim=action_dim)

        # warm up so the first real query isn't a cold-start outlier
        qz = torch.zeros(1, state_dim, device="cuda")
        iz = torch.zeros(1, n_cams, 3, 224, 224, device="cuda")
        for _ in range(6):
            runner(qz, iz)
        torch.cuda.synchronize()
        ready_q.put(("ok", None))
    except Exception as e:
        import traceback
        ready_q.put(("err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
        return

    while True:
        item = req_q.get()
        if item is None:
            break
        try:
            qpos_np, img_np, t = item
            t0 = time.time()
            qpos_t, img_t = prep_inputs(qpos_np, img_np)
            chunk, pad = runner(qpos_t, img_t)
            res_q.put((chunk, pad, t, time.time() - t0))
        except Exception as e:
            print(f"  [proc] inference failed: {type(e).__name__}: {e}")


class ProcEngine:
    """True parallel inference in a separate process. No GIL contention."""

    def __init__(self, cfg: dict, state_dim: int, n_cams: int, timeout: float = 180.0):
        import torch.multiprocessing as mp
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        self.mp = mp
        self.req_q = mp.Queue(maxsize=2)
        self.res_q = mp.Queue()
        ready_q = mp.Queue()

        self.proc = mp.Process(
            target=_proc_worker,
            args=(cfg["ckpt_dir"], cfg["eval_ckpt"], cfg["policy_class"],
                  cfg["amp"], cfg["action_dim"], cfg["snn_T"], cfg["snn_T_from"],
                  cfg["sj_step_mode"], cfg["sj_backend"],
                  state_dim, n_cams, cfg["torch_threads"],
                  self.req_q, self.res_q, ready_q),
            daemon=True,
        )
        print("  [proc] starting inference process (loads the model again; "
              "expect ~20-60 s and a second CUDA context)...")
        self.proc.start()

        status, err = ready_q.get(timeout=timeout)
        if status != "ok":
            self.proc.terminate()
            raise RuntimeError(f"inference process failed to start:\n{err}")
        print("  [proc] inference process ready.")

        self._inflight = 0
        self.last_latency = 0.0

    def can_submit(self) -> bool:
        return self._inflight == 0

    def submit(self, qpos_np, img_np, t) -> bool:
        if self._inflight > 0:
            return False
        try:
            self.req_q.put_nowait((qpos_np, img_np, t))
            self._inflight += 1
            return True
        except Exception:
            return False

    def poll(self):
        try:
            chunk, pad, t, lat = self.res_q.get_nowait()
        except Exception:
            return None
        self._inflight = max(0, self._inflight - 1)
        self.last_latency = lat
        return (chunk, pad, t)

    def drain(self):
        while True:
            try:
                self.res_q.get_nowait()
            except Exception:
                break
        self._inflight = 0

    def close(self):
        try:
            self.req_q.put(None)
            self.proc.join(timeout=5)
        except Exception:
            pass
        if self.proc.is_alive():
            self.proc.terminate()


# ══════════════════════════════════════════════════════════════════════════════
#  DEFERRED ROBOT IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
def _import_robot_stack():
    global DT, FPS, TASK_CONFIGS, START_ARM_POSE
    global PUPPET_GRIPPER_JOINT_CLOSE, PUPPET_GRIPPER_JOINT_OPEN
    global move_arms, move_grippers, torque_on, make_real_env
    global create_interbotix_global_node, get_interbotix_global_node, robot_startup

    from constants import DT, FPS, TASK_CONFIGS, START_ARM_POSE
    from constants import PUPPET_GRIPPER_JOINT_CLOSE, PUPPET_GRIPPER_JOINT_OPEN
    from robot_utils import move_arms, move_grippers, torque_on
    from real_env import make_real_env
    from interbotix_common_modules.common_robot.robot import (
        create_interbotix_global_node,
        get_interbotix_global_node,
        robot_startup,
    )


def _import_task_configs_only():
    global DT, FPS, TASK_CONFIGS
    from constants import DT, FPS, TASK_CONFIGS


# ══════════════════════════════════════════════════════════════════════════════
#  OPENING CEREMONY
# ══════════════════════════════════════════════════════════════════════════════
def opening_ceremony(env):
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

    move_arms([env.puppet_bot_left, env.puppet_bot_right],
              [pose_with_flip, pose_with_flip], move_time=2.0)
    move_grippers([env.puppet_bot_left, env.puppet_bot_right],
                  [PUPPET_GRIPPER_JOINT_CLOSE, PUPPET_GRIPPER_JOINT_CLOSE],
                  move_time=0.5)

    print("\nRobot at start pose. Press ENTER to begin (Ctrl+C to quit)...", flush=True)
    input()



# ══════════════════════════════════════════════════════════════════════════════
#  ACTION SMOOTHING
# ══════════════════════════════════════════════════════════════════════════════
def smooth_chunk_time(chunk: np.ndarray, window: int) -> np.ndarray:
    """
    Low-pass a predicted chunk along its TIME axis.

    Zero-lag: the whole future trajectory is already known, so this is an
    offline (non-causal) filter. Unlike an output EMA it removes jitter without
    delaying the motion. Savitzky-Golay preserves peaks and slopes better than
    a boxcar, so use it when scipy is available.
    """
    if window is None or window < 3 or chunk.shape[0] < window:
        return chunk
    if window % 2 == 0:
        window += 1
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(chunk, window_length=window, polyorder=2,
                             axis=0, mode="interp").astype(chunk.dtype)
    except Exception:
        pad = window // 2
        padded = np.pad(chunk, ((pad, pad), (0, 0)), mode="edge")
        kern = np.ones(window, dtype=np.float64) / window
        out = np.empty_like(chunk, dtype=np.float64)
        for d in range(chunk.shape[1]):
            out[:, d] = np.convolve(padded[:, d], kern, mode="valid")
        return out.astype(chunk.dtype)


def ensemble_step(chunk_buf, t: int, k: float):
    """
    Average every buffered chunk that predicts global step `t`, weighting newer
    chunks higher — ACT's temporal ensembling, applied across chunks instead of
    across per-step queries.

    chunk_buf holds newest-last dicts of {chunk, pad, start}.
    Returns (action, pad_logit, n_used).
    """
    vals, pads, ws = [], [], []
    n = len(chunk_buf)
    for age, c in enumerate(reversed(chunk_buf)):     # age 0 == newest
        idx = t - c["start"]
        if 0 <= idx < c["chunk"].shape[0]:
            vals.append(c["chunk"][idx])
            pads.append(c["pad"][idx] if c["pad"] is not None else -999.0)
            ws.append(math.exp(-k * age))
    if not vals:
        c = chunk_buf[-1]
        idx = min(max(t - c["start"], 0), c["chunk"].shape[0] - 1)
        pl = c["pad"][idx] if c["pad"] is not None else -999.0
        return c["chunk"][idx], float(pl), 0
    w = np.asarray(ws, dtype=np.float64)
    w /= w.sum()
    action = (np.stack(vals, 0) * w[:, None]).sum(0)
    return action, float(np.dot(np.asarray(pads, dtype=np.float64), w)), len(vals)


# ══════════════════════════════════════════════════════════════════════════════
#  CHUNK BLENDING (global-timeline indexed)
# ══════════════════════════════════════════════════════════════════════════════
def blend_on_timeline(old_chunk, old_start, new_chunk, new_start, t_now,
                      ramp, blend_type="linear", exp_k=5.0) -> np.ndarray:
    """
    Crossfade old_chunk into new_chunk over `ramp` steps starting at global
    step t_now. Chunks are indexed on the global timeline: chunk[i] executes at
    step start + i. Correct for both sync (new_start == t_now) and async
    (new_start < t_now, head of the new chunk already in the past).
    """
    out = new_chunk.copy()
    if old_chunk is None:
        return out
    n_old, n_new = old_chunk.shape[0], new_chunk.shape[0]
    ramp = max(int(ramp), 1)
    for k in range(ramp):
        g = t_now + k
        i_new, i_old = g - new_start, g - old_start
        if not (0 <= i_new < n_new) or not (0 <= i_old < n_old):
            continue
        if blend_type == "exp":
            w = 1.0 - math.exp(-exp_k * k / max(ramp - 1, 1))
        else:
            w = k / max(ramp - 1, 1)
        out[i_new] = (1.0 - min(max(w, 0.0), 1.0)) * old_chunk[i_old] \
                     + min(max(w, 0.0), 1.0) * new_chunk[i_new]
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  EPISODE LOOP
# ══════════════════════════════════════════════════════════════════════════════
_break_episode = False


def _on_press(key):
    global _break_episode
    try:
        if key.char == 'q':
            print("\n[INFO] 'q' pressed — ending episode early.")
            _break_episode = True
    except AttributeError:
        pass


def run_one_episode(env, engine, stats, camera_names, episode_len, num_queries,
                    query_frequency, blend_chunks, blend_type, blend_len,
                    save_video, done_threshold=0.9, done_patience=5,
                    done_min_steps=50, action_dim=14,
                    action_mode="ensemble", ensemble_size=4, ensemble_k=0.1,
                    smooth_chunk=0, slew_limit=0.0, ema=0.0):
    global _break_episode
    _break_episode = False

    from pynput import keyboard

    qpos_mean, qpos_std = stats["qpos_mean"], stats["qpos_std"]
    action_mean, action_std = stats["action_mean"], stats["action_std"]

    engine.drain()

    chunk = chunk_pad = None
    chunk_start = 0
    last_submit = -10 ** 9
    chunk_buf = deque(maxlen=max(1, ensemble_size))   # newest last
    use_ens = (action_mode == "ensemble")
    if use_ens:
        blend_chunks = False        # ensembling already smooths the handoff
    prev_cmd = None
    delta_all, delta_handoff = [], []
    accept_steps = set()

    video_frames, work_ms, period_s, infer_ms = [], [], [], []
    prev_loop_t = None
    done_streak = 0
    n_accept = n_reject = n_hold = 0
    warned_stale = False
    ramp = max(1, min(blend_len, query_frequency)) if blend_chunks else 1

    listener = keyboard.Listener(on_press=_on_press)
    listener.start()
    print(f">> qf={query_frequency}  blend={blend_chunks} ({blend_type}, ramp={ramp})"
          f"  num_queries={num_queries}   Press 'q' to end early. <<")

    ts = env.reset(fake=True)

    def _accept(res, t):
        """Take a finished chunk if it is still usable. Returns True if taken."""
        nonlocal chunk, chunk_pad, chunk_start, n_accept, n_reject, warned_stale
        new_chunk, new_pad, issued_t = res
        age = t - issued_t
        if age >= new_chunk.shape[0] - 1:
            n_reject += 1
            if not warned_stale:
                warned_stale = True
                print(f"\n  [!] Chunk issued at t={issued_t} arrived at t={t} "
                      f"(age {age} >= {new_chunk.shape[0]} actions). REJECTED.")
                print( "      Inference is slower than one full chunk. Raise "
                       "--query_frequency, use --async_mode process, or cut "
                       "the forward cost.\n")
            return False
        if smooth_chunk >= 3:
            new_chunk = smooth_chunk_time(new_chunk, smooth_chunk)
        chunk_buf.append({"chunk": new_chunk, "pad": new_pad, "start": issued_t})
        if blend_chunks and chunk is not None:
            new_chunk = blend_on_timeline(chunk, chunk_start, new_chunk,
                                          issued_t, t_now=t, ramp=ramp,
                                          blend_type=blend_type)
        chunk, chunk_pad, chunk_start = new_chunk, new_pad, issued_t
        n_accept += 1
        accept_steps.add(t)
        return True

    for t in range(episode_len):
        if _break_episode:
            break
        t0 = time.time()
        if prev_loop_t is not None:
            period_s.append(t0 - prev_loop_t)      # true loop period
        prev_loop_t = t0

        obs = ts.observation
        qpos_raw = obs['qpos'].copy()
        qpos_norm = ((qpos_raw - qpos_mean) / (qpos_std + 1e-8)).astype(np.float32)
        img_np = np.stack([obs['images'][c] for c in camera_names], 0)

        if save_video:
            cams = [obs['images'][c][:, :, ::-1].copy() for c in camera_names]
            if len(cams) == 4:
                video_frames.append(np.concatenate(
                    [np.concatenate(cams[:2], 1), np.concatenate(cams[2:], 1)], 0))
            else:
                video_frames.append(np.concatenate(cams, 1))

        # ── Inference scheduling ──────────────────────────────────────────
        if chunk is None:
            ti = time.time()
            engine.submit(qpos_norm, img_np, t)
            res = engine.poll()
            while res is None:                    # block only for the very first chunk
                time.sleep(0.002)
                res = engine.poll()
            chunk, chunk_pad, chunk_start = res[0], res[1], res[2]
            if smooth_chunk >= 3:
                chunk = smooth_chunk_time(chunk, smooth_chunk)
            chunk_buf.append({"chunk": chunk, "pad": chunk_pad,
                              "start": chunk_start})
            n_accept += 1
            infer_ms.append((time.time() - ti) * 1e3)
            last_submit = t
        else:
            res = engine.poll()
            if res is not None:
                infer_ms.append(engine.last_latency * 1e3)
                _accept(res, t)
            if (t - last_submit) >= query_frequency and engine.can_submit():
                if engine.submit(qpos_norm, img_np, t):
                    last_submit = t
                    res = engine.poll()           # SyncEngine returns immediately
                    if res is not None:
                        infer_ms.append(engine.last_latency * 1e3)
                        _accept(res, t)

        # ── Pick this step's action ───────────────────────────────────────
        if use_ens:
            raw_action, pad_logit, n_used = ensemble_step(chunk_buf, t, ensemble_k)
            if n_used == 0:
                n_hold += 1
            idx = t - chunk_start
        else:
            idx = t - chunk_start
            if idx >= chunk.shape[0]:
                n_hold += 1
                idx = chunk.shape[0] - 1
            idx = max(idx, 0)
            raw_action = chunk[idx]
            pad_logit = chunk_pad[idx] if chunk_pad is not None else -999.0
            n_used = 1

        # ── Episode-done detection ────────────────────────────────────────
        pad_prob = 1.0 / (1.0 + math.exp(-float(pad_logit)))
        if t >= done_min_steps and pad_prob > done_threshold:
            done_streak += 1
            if done_streak >= done_patience:
                print(f"\n  [t={t}] Episode-done ({done_streak} consecutive, "
                      f"pad_prob={pad_prob:.2f}) — stopping.")
                break
        else:
            done_streak = 0

        action = raw_action * action_std[:action_dim] + action_mean[:action_dim]

        # Slew limiting caps how far any joint may move in one control step.
        # Hard safety net against single-step spikes; does not add lag while
        # the commanded motion stays under the cap.
        if slew_limit > 0.0 and prev_cmd is not None:
            action = np.clip(action, prev_cmd - slew_limit, prev_cmd + slew_limit)

        # EMA is causal, so it DOES add lag. Last resort only.
        if ema > 0.0 and prev_cmd is not None:
            action = ema * prev_cmd + (1.0 - ema) * action

        if prev_cmd is not None:
            d = float(np.abs(action - prev_cmd).max())
            delta_all.append(d)
            if t in accept_steps:
                delta_handoff.append(d)
        prev_cmd = action.copy()

        if t % 30 == 0:
            print(f"  t={t:>4}  idx={idx:>3}  n_ens={n_used}  "
                  f"grip_L={action[6]:.4f}  grip_R={action[13]:.4f}  "
                  f"pad_prob={pad_prob:.2f}")

        ts = env.step(action)

        elapsed = time.time() - t0                 # work only, before sleeping
        work_ms.append(elapsed * 1e3)
        if elapsed < DT:
            time.sleep(DT - elapsed)

    listener.stop()

    # ── Diagnostics ───────────────────────────────────────────────────────
    n_steps = len(work_ms)
    print("\n  ---- episode diagnostics ----")
    if period_s:
        # ACTUAL loop rate, measured start-to-start. This is what the robot saw.
        print(f"  control rate : {1.0 / np.mean(period_s):5.1f} Hz actual "
              f"(target {1.0 / DT:.0f} Hz) over {n_steps} steps")
        jitter = np.percentile(period_s, 99) * 1e3
        print(f"                 p99 period {jitter:.0f} ms "
              f"(nominal {DT * 1e3:.0f} ms)")
    if work_ms:
        over = sum(1 for w in work_ms if w > DT * 1e3)
        print(f"  loop work    : mean {np.mean(work_ms):5.1f} ms, "
              f"max {np.max(work_ms):6.1f} ms, {over} steps over budget")
    if infer_ms:
        mx = np.max(infer_ms)
        budget = query_frequency * DT * 1e3
        print(f"  inference    : {len(infer_ms)} queries, mean "
              f"{np.mean(infer_ms):.0f} ms, max {mx:.0f} ms "
              f"(budget {budget:.0f} ms)")
        if mx > budget:
            print(f"  [!] Peak latency exceeds the re-query budget. Raise "
                  f"--query_frequency to >= {int(math.ceil(mx * 1.25 / (DT * 1e3)))}.")
    if delta_all:
        print(f"  jitter       : max per-step joint delta, mean "
              f"{np.mean(delta_all) * 1e3:.1f} mrad, "
              f"p99 {np.percentile(delta_all, 99) * 1e3:.1f} mrad")
        if delta_handoff:
            print(f"                 at chunk handoffs: "
                  f"{np.mean(delta_handoff) * 1e3:.1f} mrad "
                  f"({len(delta_handoff)} handoffs)")
            if np.mean(delta_handoff) > 2.5 * np.mean(delta_all):
                print("  [!] Handoff steps jump far more than typical steps ->")
                print("      the shake is at chunk boundaries. Raise --blend_len")
                print("      or use --action_mode ensemble.")
            else:
                print("      Handoffs look like ordinary steps -> the shake is")
                print("      inside the chunk (policy noise), not at the seams.")
                print("      Try --smooth_chunk 9 and a larger --ensemble_size.")
    print(f"  chunks       : {n_accept} accepted, {n_reject} rejected (expired)")
    print(f"  stale steps  : {n_hold} steps held a run-out chunk")
    if n_hold > 0.1 * max(n_steps, 1):
        print("  [!] Over 10% of steps ran on an exhausted chunk. The robot was")
        print("      replaying a stale action. Fix the latency or raise --query_frequency.")
    print("  -----------------------------")

    return video_frames


# ══════════════════════════════════════════════════════════════════════════════
#  WARMUP / BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════
def benchmark(runner, state_dim, n_cams, iters=30, warmup=8) -> float:
    # Random, not zeros: an all-zero image can leave LIF neurons silent and
    # under-report the real cost on some kernels.
    g = torch.Generator(device="cuda").manual_seed(0)
    qpos = torch.randn(1, state_dim, device="cuda", generator=g)
    img = torch.rand(1, n_cams, 3, 224, 224, device="cuda", generator=g)
    for _ in range(warmup):
        runner(qpos, img)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        runner(qpos, img)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / iters * 1e3
    print(f"\n  Forward pass: {ms:.1f} ms  ({1000.0 / ms:.1f} queries/s)")
    return ms


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main(args):
    _ensure_on_path(SCRIPT_DIR)
    enable_gpu_fast_paths(args.torch_threads)

    if args.benchmark:
        _import_task_configs_only()
    else:
        _import_robot_stack()

    run_config, policy, stats = load_policy(args.ckpt_dir, args.eval_ckpt,
                                            args.policy_class)
    pcfg = run_config["policy_config"]
    task_name = args.task_name or run_config["task"]
    num_queries = pcfg.get("num_queries", 100)
    state_dim = pcfg.get("state_dim", 14)
    action_dim = min(pcfg.get("action_dim", 14), 14)

    if task_name not in TASK_CONFIGS:
        raise KeyError(f"Task '{task_name}' not in TASK_CONFIGS. "
                       f"Available: {list(TASK_CONFIGS.keys())}")
    task_cfg = TASK_CONFIGS[task_name]
    camera_names = task_cfg["camera_names"]
    episode_len = args.episode_len or task_cfg["episode_len"]
    n_cams = len(camera_names)

    policy.cuda().eval()
    apply_snn_speedups(policy, args.sj_step_mode, args.sj_backend)

    snn_T_from = args.snn_T_from
    if args.snn_T is not None:
        if snn_T_from is None:
            for k in ("T", "n_steps", "num_steps", "timesteps", "snn_T"):
                if isinstance(pcfg.get(k), int):
                    snn_T_from = pcfg[k]
                    break
        print(f"  [snn] reducing timesteps -> {args.snn_T} (from {snn_T_from})")
        set_snn_timesteps(policy, args.snn_T, snn_T_from)

    if args.compile:
        try:
            policy.model = torch.compile(policy.model, mode="reduce-overhead")
            print("  [compile] enabled (reduce-overhead)")
        except Exception as e:
            print(f"  [compile] failed: {e}")

    runner = ForwardRunner(policy, amp=args.amp, action_dim=action_dim)

    print("\nMeasuring forward latency...")
    ms = benchmark(runner, state_dim, n_cams, iters=args.benchmark_iters)

    # ── Query frequency ─────────────────────────────────────────────────────
    need_qf = int(math.ceil(ms * 1.30 / 1e3 / DT))
    if args.query_frequency is not None:
        query_frequency = args.query_frequency
    else:
        query_frequency = min(max(need_qf, 10), num_queries - 5)
        print(f"  [auto] query_frequency -> {query_frequency} "
              f"(from measured {ms:.0f} ms)")
    if args.auto_qf and query_frequency < need_qf:
        query_frequency = min(need_qf, num_queries - 5)
        print(f"  [auto_qf] raised query_frequency -> {query_frequency}")
    query_frequency = max(1, min(query_frequency, num_queries - 1))

    blend_chunks = args.blend_chunks
    if blend_chunks is None:
        blend_chunks = query_frequency < num_queries

    chunk_span_ms = num_queries * DT * 1e3
    print(f"\nTask:            {task_name}")
    print(f"Episode len:     {episode_len}")
    print(f"Num queries:     {num_queries}  (chunk covers {chunk_span_ms:.0f} ms)")
    print(f"Query frequency: {query_frequency}  (budget {query_frequency * DT * 1e3:.0f} ms)")
    print(f"Async mode:      {args.async_mode}")
    print(f"Blend chunks:    {blend_chunks}  (type={args.blend_type}, len={args.blend_len})")
    print(f"AMP:             {args.amp}")
    print(f"Cameras:         {camera_names}\n")

    if ms >= chunk_span_ms:
        print("  " + "!" * 68)
        print(f"  !! One forward ({ms:.0f} ms) is longer than an entire action")
        print(f"  !! chunk ({chunk_span_ms:.0f} ms). No scheduling can fix this —")
        print(f"  !! every chunk expires before it arrives. Cut the forward cost")
        print(f"  !! (--snn_T, --amp bf16) or the robot will just hold position.")
        print("  " + "!" * 68 + "\n")

    if args.async_mode == "thread" and ms > 50:
        print("  " + "!" * 68)
        print(f"  !! --async_mode thread with a {ms:.0f} ms forward.")
        print(f"  !! The GIL means this thread will starve the control loop.")
        print(f"  !! Use --async_mode process instead.")
        print("  " + "!" * 68 + "\n")

    if args.benchmark:
        print("Benchmark only — exiting.")
        return

    # ── Build the engine ────────────────────────────────────────────────────
    if args.async_mode == "process":
        cfg = dict(ckpt_dir=args.ckpt_dir, eval_ckpt=args.eval_ckpt,
                   policy_class=args.policy_class, amp=args.amp,
                   action_dim=action_dim, snn_T=args.snn_T,
                   snn_T_from=snn_T_from, sj_step_mode=args.sj_step_mode,
                   sj_backend=args.sj_backend, torch_threads=args.torch_threads)
        # free the parent's copy first — the child loads its own
        del runner, policy
        torch.cuda.empty_cache()
        engine = ProcEngine(cfg, state_dim, n_cams)
    elif args.async_mode == "thread":
        engine = ThreadEngine(runner)
    else:
        engine = SyncEngine(runner)

    # ── Robot init ──────────────────────────────────────────────────────────
    try:
        global_node = get_interbotix_global_node()
    except Exception:
        global_node = None
    if global_node is None:
        global_node = create_interbotix_global_node()

    env = make_real_env(init_node=False, setup_robots=True)
    robot_startup(global_node)

    from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
    from rclpy.node import Node
    if args.ros_threads <= 1:
        executor = SingleThreadedExecutor()
        print(f"  [ros] SingleThreadedExecutor")
    else:
        executor = MultiThreadedExecutor(num_threads=args.ros_threads)
        print(f"  [ros] MultiThreadedExecutor(num_threads={args.ros_threads})")
    for rec in [env.recorder_left, env.recorder_right, env.image_recorder]:
        if isinstance(rec, Node):
            executor.add_node(rec)
    threading.Thread(target=executor.spin, daemon=True).start()

    # ── Rollouts ────────────────────────────────────────────────────────────
    n_success = n_done = 0
    try:
        for rollout_id in range(args.num_rollouts):
            print(f"\n{'=' * 60}\nRollout {rollout_id + 1} / {args.num_rollouts}\n{'=' * 60}")
            try:
                opening_ceremony(env)
            except KeyboardInterrupt:
                print("\nAborted.")
                break

            try:
                video_frames = run_one_episode(
                    env, engine, stats, camera_names, episode_len, num_queries,
                    query_frequency=query_frequency, blend_chunks=blend_chunks,
                    blend_type=args.blend_type, blend_len=args.blend_len,
                    save_video=args.save_video, done_threshold=args.done_threshold,
                    done_patience=args.done_patience,
                    done_min_steps=args.done_min_steps, action_dim=action_dim,
                    action_mode=args.action_mode,
                    ensemble_size=args.ensemble_size,
                    ensemble_k=args.ensemble_k,
                    smooth_chunk=args.smooth_chunk,
                    slew_limit=args.slew_limit, ema=args.ema)
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
            n_done += 1
            if result == 'y':
                n_success += 1
                print("  Marked as SUCCESS")
            elif result == 'q':
                print("Exiting early.")
                break
            else:
                print("  Marked as FAILURE")
    finally:
        engine.close()

    if n_done > 0:
        print(f"\nFinal: {n_success}/{n_done} = {100.0 * n_success / n_done:.1f}% success")


def parse_args():
    p = argparse.ArgumentParser(
        description="Run a trained ACT or SACT policy on the real ALOHA robot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--policy_class", default=None, choices=["ACT", "SACT"])
    p.add_argument("--task_name", default=None)
    p.add_argument("--num_rollouts", type=int, default=5)
    p.add_argument("--eval_ckpt", default=None)
    p.add_argument("--episode_len", type=int, default=None)
    p.add_argument("--save_video", action="store_true")
    p.add_argument("--video_dir",
                   default="/workspace/aloha_mujoco_project/snn_aloha/eval_videos")

    p.add_argument("--async_mode", default="off",
                   choices=["off", "thread", "process"],
                   help="off = blocking (predictable hitch). process = true "
                        "parallel inference. thread = GIL-bound, avoid.")
    p.add_argument("--query_frequency", type=int, default=None,
                   help="Re-query every N steps. Auto-sized from measured "
                        "latency if omitted.")
    p.add_argument("--auto_qf", action="store_true",
                   help="Raise query_frequency to fit measured latency +30%%.")
    p.add_argument("--blend_chunks", type=lambda v: v.lower() in ('true', '1', 'yes'),
                   default=None)
    p.add_argument("--blend_type", default="linear", choices=["linear", "exp"])
    p.add_argument("--blend_len", type=int, default=10)

    p.add_argument("--action_mode", default="ensemble",
                   choices=["ensemble", "chunk"],
                   help="ensemble = average all buffered chunks covering the "
                        "current step (temporal ensembling, much smoother). "
                        "chunk = execute one chunk, crossfade at handoffs.")
    p.add_argument("--ensemble_size", type=int, default=4,
                   help="How many recent chunks to keep for averaging.")
    p.add_argument("--ensemble_k", type=float, default=0.1,
                   help="Recency weight. 0.0 = uniform (smoothest), higher "
                        "favours the newest chunk (more responsive, shakier). "
                        "ACT uses 0.01.")
    p.add_argument("--smooth_chunk", type=int, default=0,
                   help="Savitzky-Golay window over each chunk's time axis "
                        "(odd, try 7-11). Zero-lag. 0 disables.")
    p.add_argument("--slew_limit", type=float, default=0.0,
                   help="Max per-joint change per control step, radians. "
                        "Try 0.02-0.05. 0 disables.")
    p.add_argument("--ema", type=float, default=0.0,
                   help="Causal EMA on the command, 0-1. ADDS LAG; prefer "
                        "--smooth_chunk. 0 disables.")

    p.add_argument("--amp", default="off", choices=["off", "bf16", "fp16"])
    p.add_argument("--snn_T", type=int, default=None,
                   help="Reduce SNN unroll length at inference. CHANGES OUTPUT.")
    p.add_argument("--snn_T_from", type=int, default=None)
    p.add_argument("--sj_step_mode", default=None, choices=["s", "m"],
                   help="SpikingJelly only; inert on norse.")
    p.add_argument("--sj_backend", default=None, choices=["torch", "cupy"],
                   help="SpikingJelly only; inert on norse.")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--benchmark_iters", type=int, default=30)

    p.add_argument("--ros_threads", type=int, default=2,
                   help="Threads for the rclpy executor. The default of "
                        "os.cpu_count() floods the GIL and can slow a "
                        "launch-bound SNN forward by several times. "
                        "Use 1 for a SingleThreadedExecutor.")
    p.add_argument("--torch_threads", type=int, default=1,
                   help="torch.set_num_threads. Intra-op parallelism does not "
                        "help a launch-bound SNN but does fight ROS for cores.")

    p.add_argument("--done_threshold", type=float, default=0.9)
    p.add_argument("--done_patience", type=int, default=5)
    p.add_argument("--done_min_steps", type=int, default=50)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
    sys.exit(0)