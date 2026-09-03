"""Thin recording wrapper around LeRobotDataset for Panda pick-and-place teleop.

Maps the teleop session lifecycle onto the lerobot API:
    SUCCESS  -> save_episode()         (encodes video + parquet = a checkpoint)
    CANCEL   -> clear_episode_buffer() (discard a bad take, redo)
    END      -> finalize()             (writes footer; dataset invalid without it)

Crash safety: metadata_buffer_size is forced to 1 on BOTH create and resume
(lerobot's resume() otherwise defaults to 10 and would silently buffer/lose
episode metadata if interrupted), so every completed episode is flushed to disk
immediately. A SIGINT/SIGTERM/atexit guard also runs finalize() so the parquet
footer is always written. Re-running with the same root resumes and appends.

The dataset is NEVER pushed to the Hub -- push_to_hub is simply never called.

A sidecar `episodes_meta.jsonl` logs the randomization params + outcome per
episode for reproducibility.
"""
import atexit
import json
import os
import shutil
import signal
import time
from pathlib import Path

# Force fully offline: this dataset is local-only and must NEVER reach the Hub.
# Set before importing lerobot/huggingface_hub so the offline flags take effect.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = "panda_pick_place"
TASK = "pick up the object and place it in the bin"
CAMS = ("front", "diag", "wrist")
IMG_H, IMG_W = 480, 640
STATE_DIM = 8          # 7 arm joints + gripper
ACTION_DIM = 8         # 7 arm targets + gripper target

# Depth quantization range (metres). Everything outside [DEPTH_NEAR, DEPTH_FAR]
# gets clipped before encoding. Chosen to cover our workspace:
#   wrist cam:  ~0.06-1.20 m
#   diag cam:   ~0.76-4.58 m (background hits the ceiling, that is fine --
#               objects of interest sit within ~0.7-1.5 m)
#   front cam:  ~0.89-20 m   (same story: background clipped, objects preserved)
# With uint8 across a 2.0 m span each level is ~7.8 mm -- adequate for a
# 15x20 backbone feature grid; if we ever need mm precision we can switch to
# uint16 PNG per frame (dtype: "image") instead of the video pipeline.
DEPTH_NEAR = 0.05
DEPTH_FAR = 2.05


def depth_to_uint8(depth_m: np.ndarray, near: float = DEPTH_NEAR,
                   far: float = DEPTH_FAR) -> np.ndarray:
    """Quantize a metric depth map (h,w) float32 metres to uint8 [0,255].

    Encoding: near = 255 (bright), far = 0 (black). This matches the visualization
    in control.Viewer._draw_depth so eyeballing recorded frames looks right.
    Video-encode friendly (3-channel repeat)."""
    z = np.clip(depth_m, near, far)
    g = ((1.0 - (z - near) / (far - near)) * 255.0).astype(np.uint8)  # (h,w)
    return np.repeat(g[:, :, None], 3, axis=2)                       # (h,w,3)


def build_features():
    feats = {
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,),
                              "names": [f"arm_{i}" for i in range(7)] + ["gripper"]},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,),
                  "names": [f"arm_{i}" for i in range(7)] + ["gripper"]},
    }
    for cam in CAMS:
        feats[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channels"],
        }
        # Depth is stored as a uint8 video (single channel repeated 3x) so it
        # reuses lerobot`s H.264 pipeline. Consumers dequantize back to metres
        # via (255 - v)/255 * (FAR - NEAR) + NEAR (see depth_from_uint8).
        feats[f"observation.depths.{cam}"] = {
            "dtype": "video", "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channels"],
        }
    return feats


class Recorder:
    """Owns a write-mode LeRobotDataset and the per-episode buffer state."""

    def __init__(self, dataset, root, fps):
        self.ds = dataset
        self.root = Path(root)
        self.fps = fps
        self.recording = False
        self._meta_path = self.root / "episodes_meta.jsonl"
        self._ep_info = None
        self._finalized = False
        self._install_interrupt_guard()

    # --- crash / interrupt safety ------------------------------------------
    def _install_interrupt_guard(self):
        """Ensure a clean finalize() runs on Ctrl-C, SIGTERM, or normal exit so
        the parquet footer is written and the metadata buffer is flushed. With
        metadata_buffer_size=1 every completed episode is already on disk, so the
        worst case (a SIGKILL we can't catch) loses only the in-flight take."""
        atexit.register(self._safe_finalize)

        def _handler(signum, frame):
            print(f"\n[recorder] signal {signum} received -- finalizing dataset...")
            self._safe_finalize()
            # Re-raise as KeyboardInterrupt so the caller's normal shutdown
            # (viewer close, etc.) still runs. finalize() is idempotent.
            raise KeyboardInterrupt

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                pass  # not main thread: atexit still covers normal exit

    def _safe_finalize(self):
        """Flush + finalize exactly once. Discards any in-flight partial take
        (an incomplete episode is never a clean success). Safe to call from a
        signal handler, atexit, or explicitly; lerobot's finalize() is itself
        idempotent."""
        if self._finalized:
            return
        self._finalized = True
        try:
            if self.recording:
                self.cancel()  # discard partial take (logs + clears buffer)
        except Exception:
            pass
        try:
            self.ds.finalize()
            print(f"[recorder] dataset finalized cleanly ({self.num_episodes} episodes) at {self.root}")
        except Exception as e:
            print(f"[recorder] WARNING: finalize failed on exit: {e}")

    # --- construction -------------------------------------------------------
    @classmethod
    def open(cls, root, repo_id=REPO_ID, fps=30, resume=True):
        """Create a new dataset, or resume an existing one at `root`."""
        root = Path(root)
        # A *resumable* dataset has both info.json and tasks.parquet -- the
        # latter only exists once >=1 episode was saved. An info.json without
        # tasks.parquet is a 0-episode partial from a crashed create; resuming
        # it makes lerobot try to repair metadata from the Hub. Wipe it so we
        # create cleanly (offline) instead.
        has_info = (root / "meta" / "info.json").exists()
        has_tasks = (root / "meta" / "tasks.parquet").exists()
        if has_info and not has_tasks:
            print(f"[recorder] found 0-episode partial at {root}; wiping for clean create")
            shutil.rmtree(root)
            has_info = False
        common = dict(image_writer_threads=4, batch_encoding_size=1)
        if has_info and resume:
            ds = LeRobotDataset.resume(repo_id, root=root, **common)
            print(f"[recorder] resuming {repo_id} at {root} "
                  f"({ds.num_episodes} existing episodes)")
        else:
            ds = LeRobotDataset.create(
                repo_id, fps=fps, features=build_features(), root=root,
                robot_type="panda", use_videos=True, metadata_buffer_size=1,
                **common)
            print(f"[recorder] created new dataset {repo_id} at {root}")
        # Episode metadata is buffered in memory and flushed only every
        # `metadata_buffer_size` episodes (lerobot default 10) or at finalize().
        # LeRobotDataset.resume() does NOT expose that kwarg, so a resumed
        # session silently buffers 10 -- and if it ends without a clean
        # finalize(), up to 9 episodes' metadata is lost while their frames +
        # video (written per-episode) survive, leaving orphan frames and a
        # non-contiguous episode_index that lerobot's v3.0 reader cannot load.
        # Force per-episode flush on BOTH create and resume so an interrupted
        # session can lose at most the in-flight partial take.
        ds.meta._metadata_buffer_size = 1
        return cls(ds, root, fps)

    @property
    def num_episodes(self):
        return self.ds.num_episodes

    # --- episode lifecycle --------------------------------------------------
    def start_episode(self, info=None):
        """Begin buffering a new take. `info` = randomization params (optional)."""
        self.recording = True
        self._ep_info = info or {}
        self._t0 = time.time()

    def add(self, images: dict, state, action, depths: dict | None = None):
        """Buffer one timestep.

        Args:
            images: {cam_name: (H,W,3) uint8 RGB}
            state : (STATE_DIM,) float32 -- current qpos[:8]
            action: (ACTION_DIM,) float32 -- control target this step
            depths: {cam_name: (H,W) float32 metric depth in metres}, optional.
                    If provided, quantized to uint8 [near=255, far=0] and stored
                    as video under `observation.depths.{cam}`. If omitted the
                    depth features are filled with zeros so the schema stays
                    consistent across frames (all-black = "far / unknown").
        """
        if not self.recording:
            return
        frame = {
            "observation.state": np.asarray(state, np.float32),
            "action": np.asarray(action, np.float32),
            "task": TASK,
        }
        for cam in CAMS:
            frame[f"observation.images.{cam}"] = images[cam]
            if depths is not None and cam in depths:
                frame[f"observation.depths.{cam}"] = depth_to_uint8(depths[cam])
            else:
                # Placeholder so lerobot never sees a missing key mid-episode.
                # All-zero uint8 decodes to "far" under our convention.
                frame[f"observation.depths.{cam}"] = np.zeros((IMG_H, IMG_W, 3), np.uint8)
        self.ds.add_frame(frame)

    def success(self):
        """Persist the current episode (checkpoint) and advance."""
        if not self.recording or not self.ds.has_pending_frames():
            return False
        self.ds.save_episode(parallel_encoding=False)
        self._log_meta("success")
        self.recording = False
        print(f"[recorder] saved episode -> total {self.num_episodes}")
        return True

    def cancel(self):
        """Discard the current take without saving."""
        if not self.recording:
            return
        self.ds.clear_episode_buffer(delete_images=True)
        self._log_meta("cancelled")
        self.recording = False
        print("[recorder] episode cancelled (buffer cleared)")

    def end(self):
        """Finalize the dataset (writes a valid footer). Idempotent -- the same
        path runs on Ctrl-C / SIGTERM / interpreter exit via the guard."""
        self._safe_finalize()

    # --- sidecar reproducibility log ---------------------------------------
    def _log_meta(self, outcome):
        rec = {"episode": self.num_episodes, "outcome": outcome,
               "duration_s": round(time.time() - self._t0, 2)}
        info = dict(self._ep_info)
        for k, v in info.items():                 # make json-safe
            if isinstance(v, np.ndarray):
                info[k] = v.tolist()
        rec.update(info)
        with open(self._meta_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
