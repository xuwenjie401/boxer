#! /usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
"""Incremental OAK RTAB-Map demo viewer.

This viewer replays precomputed Boxer OBB CSVs as an online SLAM-like demo:
the camera moves through the OAK RTAB trajectory, current-frame 3D boxes appear,
and TrackerViewer incrementally fuses them into persistent global boxes.
"""

import argparse
import json
import os
import time as time_module
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import cv2
import moderngl
import numpy as np
import torch
import utils.imgui_compat as imgui

from loaders.oak_rtab_loader import OakRtabLoader, _matrix_from_row
from utils.demo_utils import EVAL_PATH
from utils.file_io import read_obb_csv
from utils.tw.obb import ObbTW
from utils.tw.pose import PoseTW
from utils.tw.tensor_utils import find_nearest2
from utils.viewer_3d import (
    TrackerViewer,
    launch_viewer,
    load_view_file,
    scale_factor,
    subsample_timed_obbs,
)


DEFAULT_RTAB_EXPORT = (
    "/home/wjxu22/Datasets/outputs/rtab/oak_stereo_imu_gravity_lossless_export"
)


def _is_oak_rtab_export(path: str) -> bool:
    root = Path(path).expanduser()
    return (
        root.is_dir()
        and (root / "metadata.json").is_file()
        and (root / "poses" / "rgb_poses.csv").is_file()
    )


def _pose_from_matrix_np(matrix: np.ndarray) -> PoseTW:
    matrix = np.asarray(matrix, dtype=np.float32)
    return PoseTW.from_Rt(
        torch.from_numpy(matrix[:3, :3]),
        torch.from_numpy(matrix[:3, 3]),
    ).float()


def _quat_xyzw_to_matrix_np(quat_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = q / norm
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _load_odom_camera_trajectory(
    odom_path: str | Path,
    loader: OakRtabLoader,
) -> tuple[np.ndarray, list[PoseTW]] | None:
    odom_path = Path(odom_path).expanduser()
    if not odom_path.is_file():
        return None

    with np.load(odom_path) as data:
        timestamps = data["timestamps_ns"].astype(np.int64)
        positions = data["positions_xyz"].astype(np.float64)
        quats = data["quaternions_xyzw"].astype(np.float64)
    if len(timestamps) < 2:
        return None

    mats = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(timestamps), axis=0)
    for i in range(len(timestamps)):
        mats[i, :3, :3] = _quat_xyzw_to_matrix_np(quats[i])
        mats[i, :3, 3] = positions[i]

    rgb_timestamps = np.array(
        [int(row["rgb_stamp_ns"]) for row in loader.rows], dtype=np.int64
    )
    ref_rgb_idx = 0
    ref_odom_idx = int(find_nearest2(timestamps, int(rgb_timestamps[ref_rgb_idx])))
    T_world_rgb_ref = _matrix_from_row(
        loader.rows[ref_rgb_idx], "map_rgb_optical"
    ).astype(np.float64)
    T_world_odom_ref = mats[ref_odom_idx]
    T_odom_rgb = np.linalg.inv(T_world_odom_ref) @ T_world_rgb_ref
    mats = mats @ T_odom_rgb

    traj = [_pose_from_matrix_np(mat) for mat in mats]
    return timestamps, traj


def _build_oak_seq_ctx(loader: OakRtabLoader, odom_path: str | Path = "") -> dict:
    rgb_timestamps = np.array(
        [int(row["rgb_stamp_ns"]) for row in loader.rows], dtype=np.int64
    )
    cam_template, _, _, _, _, _, _ = loader._camera_for_resize()

    odom_loaded = _load_odom_camera_trajectory(odom_path, loader) if odom_path else None
    if odom_loaded is not None:
        pose_timestamps, traj = odom_loaded
        calibs = [cam_template.clone().float() for _ in range(len(pose_timestamps))]
        print(
            f"Loaded 30Hz odom camera trajectory: {len(pose_timestamps)} poses "
            f"from {odom_path}"
        )
    else:
        pose_timestamps = rgb_timestamps
        traj = []
        calibs = []
        for row in loader.rows:
            T_world_cam_np = _matrix_from_row(row, "map_rgb_optical").astype(np.float32)
            traj.append(_pose_from_matrix_np(T_world_cam_np))
            calibs.append(cam_template.clone().float())

    return {
        "source": "oak_rtab",
        "loader": loader,
        "rgb_num_frames": len(rgb_timestamps),
        "rgb_timestamps": rgb_timestamps,
        "rgb_images": None,
        "is_nebula": True,
        "traj": traj,
        "pose_ts": pose_timestamps,
        "calibs": calibs,
        "calib_ts": pose_timestamps,
        "time_to_uids_slaml": None,
        "time_to_uids_slamr": None,
        "uid_to_p3": None,
        "sdp_global": loader.voxel_centers,
    }


def _empty_obbs() -> ObbTW:
    return ObbTW(torch.zeros(0, 165))


def _merge_obbs(a: ObbTW, b: ObbTW) -> ObbTW:
    if len(a) == 0:
        return b
    if len(b) == 0:
        return a
    return torch.cat([a, b], dim=0)


def _expand_timed_obbs_to_nav_timeline(
    timed_obbs: dict[int, ObbTW],
    nav_timestamps: np.ndarray,
    max_match_dt_ns: int = 50_000_000,
) -> dict[int, ObbTW]:
    """Use the 30Hz pose timestamps as playback frames while keeping detections sparse."""
    if not timed_obbs or len(nav_timestamps) < 2:
        return timed_obbs

    det_ts = np.array(sorted(timed_obbs.keys()), dtype=np.int64)
    nav_timestamps = np.asarray(nav_timestamps, dtype=np.int64)
    mask = (nav_timestamps >= det_ts[0]) & (nav_timestamps <= det_ts[-1])
    selected_nav_ts = nav_timestamps[mask]
    if len(selected_nav_ts) < 2:
        return timed_obbs

    expanded = {int(ts): _empty_obbs() for ts in selected_nav_ts}
    for ts in det_ts:
        idx = int(find_nearest2(selected_nav_ts, int(ts)))
        nav_ts = int(selected_nav_ts[idx])
        if abs(nav_ts - int(ts)) <= max_match_dt_ns:
            expanded[nav_ts] = _merge_obbs(expanded[nav_ts], timed_obbs[int(ts)])
        else:
            expanded[int(ts)] = timed_obbs[int(ts)]

    return dict(sorted(expanded.items()))


def _stack_obbs(timed_obbs: dict[int, ObbTW]) -> ObbTW:
    parts = [obbs for _, obbs in sorted(timed_obbs.items()) if len(obbs) > 0]
    if not parts:
        return ObbTW(torch.zeros(0, 165))
    return torch.cat(parts, dim=0)


def _count_obbs(timed_obbs: dict[int, ObbTW]) -> int:
    return sum(len(obbs) for obbs in timed_obbs.values())


def _downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if max_points <= 0 or len(points) <= max_points:
        return points
    rng = np.random.default_rng(0)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[np.sort(idx)]


class OakRtabIncrementalViewer(TrackerViewer):
    title = "OAK RTAB Incremental Boxer Demo"

    def __init__(
        self,
        *,
        rtab_loader: OakRtabLoader,
        max_points: int,
        point_size: float,
        point_alpha: float,
        raw_conf: float,
        fused_conf: float,
        fusion_min_hits: int,
        fusion_iou: float,
        fusion_max_missed: int,
        playback_fps: float,
        playback_speed: float,
        label_hold_sec: float,
        normal_stride: int,
        raw_hold_ms: float,
        media_panel_frac: float,
        stereo_manifest_path: str = "",
        **kwargs,
    ) -> None:
        self._rtab_loader = rtab_loader
        self._rtab_max_points = int(max_points)
        self._rtab_initial_point_size = float(point_size)
        self._rtab_initial_point_alpha = float(point_alpha)
        self._oak_raw_conf = float(raw_conf)
        self._oak_fused_conf = float(fused_conf)
        self._oak_fusion_min_hits = int(fusion_min_hits)
        self._oak_fusion_iou = float(fusion_iou)
        self._oak_fusion_max_missed = int(fusion_max_missed)
        self._oak_playback_fps = float(playback_fps)
        self._oak_playback_speed = max(0.1, float(playback_speed))
        self._oak_label_hold_sec = max(0.0, float(label_hold_sec))
        self._oak_normal_stride = max(1, int(normal_stride))
        self._oak_raw_hold_ns = int(max(0.0, float(raw_hold_ms)) * 1e6)
        self._oak_label_frame_indices: set[int] = set()
        self._oak_detection_frame_indices: set[int] = set()
        self._oak_nonempty_timestamps = np.array([], dtype=np.int64)
        self._oak_rgb_hold_frame_idx: Optional[int] = None
        self._oak_rgb_hold_timestamp: Optional[int] = None
        self._oak_rgb_hold_until_time = 0.0
        self._oak_pending_rgb_label_frame_idx: Optional[int] = None
        self._oak_last_rgb_update_frame_idx = -1
        self._oak_media_bb2d_current_boxes: list[
            tuple[float, float, float, float, str, int]
        ] = []
        self._oak_media_bb2d_img_wh: tuple[int, int] = (0, 0)
        self._oak_root_path = Path(kwargs.get("root_path", ".")).expanduser().resolve()
        if stereo_manifest_path:
            self._stereo_manifest_path = Path(stereo_manifest_path).expanduser().resolve()
            self._stereo_cache_dir = self._stereo_manifest_path.parent
        else:
            self._stereo_cache_dir = self._oak_root_path / "stereo_cache"
            self._stereo_manifest_path = self._stereo_cache_dir / "stereo_manifest.json"
        self._stereo_pairs: list[dict] = []
        self._stereo_pair_timestamps = np.array([], dtype=np.int64)
        self._stereo_lru_cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self._stereo_lru_max_items = 12
        self._stereo_textures = {"left": None, "right": None}
        self._stereo_tex_sizes = {"left": (0, 0), "right": (0, 0)}
        self._load_stereo_manifest()
        super().__init__(**kwargs)

        self.raw_conf_threshold = self._oak_raw_conf
        self.prob_threshold = self._oak_raw_conf
        self.tracker_conf_threshold = self._oak_fused_conf
        self.tracker_min_hits = self._oak_fusion_min_hits
        self.tracker_iou_threshold = self._oak_fusion_iou
        self.tracker_merge_iou = max(self._oak_fusion_iou, 0.05)
        self.tracker_max_missed = self._oak_fusion_max_missed
        self.tracker_min_conf_mass = max(
            self._oak_fused_conf * max(1, self._oak_fusion_min_hits), 0.8
        )
        self.playback_fps = self._oak_playback_fps

        self.point_size = self._rtab_initial_point_size
        self.point_alpha = self._rtab_initial_point_alpha
        self.obs_point_size = max(self.point_size * 2.0, 4.0)
        self.obs_point_alpha = 0.85
        self.obs_trail_secs = 0.20
        self.visibility_obs_trail_frames = 2
        self.frustum_scale = 0.35
        self.show_global_points = self.point_count > 0
        self.show_obs_points = True
        self.show_raw_set = True
        self.show_tracked_all_set = True
        self.show_tracked_visible_set = True
        self.show_rgb_raw = True
        self.show_rgb_tracked_visible = True
        self.show_rgb_tracked_all = False
        self.rgb_panel_max_frac = min(0.60, max(0.18, float(media_panel_frac)))
        self.tracked_all_line_width = 5
        self.visible_line_width = 7
        self.raw_line_width = 2
        self.alpha = 0.65
        self._force_free_view()
        self._refresh_playback_markers()

        self._reset_tracker()
        self._prime_first_frame()

    def _load_stereo_manifest(self) -> None:
        if not self._stereo_manifest_path.is_file():
            return
        with self._stereo_manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        pairs = manifest.get("pairs", [])
        clean_pairs = []
        pair_ts = []
        for pair in pairs:
            try:
                left_ts = int(pair["left_timestamp_ns"])
                right_ts = int(pair["right_timestamp_ns"])
            except Exception:
                continue
            clean_pairs.append(pair)
            pair_ts.append((left_ts + right_ts) // 2)
        self._stereo_pairs = clean_pairs
        self._stereo_pair_timestamps = np.asarray(pair_ts, dtype=np.int64)
        if len(self._stereo_pairs) > 0:
            print(
                f"Loaded stereo cache manifest: {len(self._stereo_pairs)} pairs "
                f"from {self._stereo_manifest_path}"
            )

    def _force_free_view(self) -> None:
        self.follow_view = False
        self._smooth_eye = None
        self._smooth_target = None
        self._smooth_up = None

    def _refresh_playback_markers(self) -> None:
        self._oak_detection_frame_indices = set()
        self._oak_label_frame_indices = set()
        nonempty_ts = []
        for idx, ts in enumerate(self.sorted_timestamps):
            obbs = self.timed_obbs.get(ts)
            if obbs is not None and len(obbs) > 0:
                self._oak_detection_frame_indices.add(idx)
                self._oak_label_frame_indices.add(idx)
                nonempty_ts.append(int(ts))

        # Include exact-ish 2D-label frames too, but keep the tolerance tight so
        # neighboring 30Hz pose frames do not all become "hold" frames.
        if self._bb2d_data is not None and self._bb2d_timestamps is not None:
            for idx, ts in enumerate(self.sorted_timestamps):
                bb2_idx = int(find_nearest2(self._bb2d_timestamps, int(ts)))
                bb2_ts = int(self._bb2d_timestamps[bb2_idx])
                if abs(bb2_ts - int(ts)) > 20_000_000:
                    continue
                entry = self._bb2d_data.get(bb2_ts)
                if entry is not None and len(entry.get("bb2d", [])) > 0:
                    self._oak_label_frame_indices.add(idx)

        self._oak_nonempty_timestamps = np.asarray(nonempty_ts, dtype=np.int64)

    def _frame_has_label(self, frame_idx: int) -> bool:
        return int(frame_idx) in self._oak_label_frame_indices

    def _display_obbs_for_timestamp(self, ts_ns: int) -> Optional[ObbTW]:
        if self._oak_raw_hold_ns <= 0 or len(self._oak_nonempty_timestamps) == 0:
            return None
        idx = (
            int(
                np.searchsorted(
                    self._oak_nonempty_timestamps, int(ts_ns), side="right"
                )
            )
            - 1
        )
        if idx < 0:
            return None
        det_ts = int(self._oak_nonempty_timestamps[idx])
        if int(ts_ns) - det_ts > self._oak_raw_hold_ns:
            return None
        return self.timed_obbs.get(det_ts)

    def _clear_rgb_hold(self) -> None:
        self._oak_rgb_hold_frame_idx = None
        self._oak_rgb_hold_timestamp = None
        self._oak_rgb_hold_until_time = 0.0
        self._oak_pending_rgb_label_frame_idx = None

    def _media_bb2d_for_timestamp(
        self, ts_ns: int
    ) -> tuple[list[tuple[float, float, float, float, str, int]], tuple[int, int]]:
        bb2d_data = getattr(self, "_bb2d_data", None)
        bb2d_timestamps = getattr(self, "_bb2d_timestamps", None)
        if bb2d_data is None or bb2d_timestamps is None:
            return [], (0, 0)
        idx = int(find_nearest2(bb2d_timestamps, int(ts_ns)))
        nearest_ts = int(bb2d_timestamps[idx])
        if abs(nearest_ts - int(ts_ns)) >= 50_000_000:
            return [], (0, 0)

        entry = bb2d_data.get(nearest_ts)
        if entry is None:
            return [], (0, 0)

        bb2d = entry["bb2d"]
        labels = entry["labels"]
        sem_ids = entry.get("sem_ids", [-1] * len(labels))
        img_wh = (int(entry["img_width"]), int(entry["img_height"]))
        orig_h = img_wh[1]
        display_wh = img_wh
        boxes = []
        for j in range(len(bb2d)):
            x1 = float(bb2d[j, 0])
            y1 = float(bb2d[j, 1])
            x2 = float(bb2d[j, 2])
            y2 = float(bb2d[j, 3])
            if not self._vrs_is_nebula:
                rx1 = orig_h - 1 - y2
                ry1 = x1
                rx2 = orig_h - 1 - y1
                ry2 = x2
                x1, y1, x2, y2 = rx1, ry1, rx2, ry2
                display_wh = (int(entry["img_height"]), int(entry["img_width"]))
            boxes.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                    self._remap_label(labels[j])[:12],
                    int(sem_ids[j]),
                )
            )
        return boxes, display_wh

    def _set_rgb_panel_timestamp(
        self,
        ts_ns: int,
        frame_idx: int,
        *,
        force: bool = False,
        hold_label_frame_idx: Optional[int] = None,
    ) -> None:
        now = time_module.time()
        if (
            self._oak_rgb_hold_timestamp is not None
            and now >= self._oak_rgb_hold_until_time
        ):
            self._clear_rgb_hold()

        label_idx = hold_label_frame_idx
        if label_idx is None and self._frame_has_label(frame_idx):
            label_idx = frame_idx
        if label_idx is not None and self._oak_label_hold_sec > 0.0:
            label_idx = max(0, min(int(label_idx), self.total_frames - 1))
            if self._oak_rgb_hold_frame_idx != label_idx:
                self._oak_rgb_hold_frame_idx = label_idx
                self._oak_rgb_hold_timestamp = int(self.sorted_timestamps[label_idx])
                self._oak_rgb_hold_until_time = now + self._oak_label_hold_sec

        if self._oak_rgb_hold_timestamp is not None:
            display_ts = int(self._oak_rgb_hold_timestamp)
        else:
            if (
                not force
                and self._oak_normal_stride > 1
                and not self._frame_has_label(frame_idx)
                and self._oak_last_rgb_update_frame_idx >= 0
                and frame_idx - self._oak_last_rgb_update_frame_idx
                < self._oak_normal_stride
            ):
                return
            display_ts = int(ts_ns)
            self._oak_last_rgb_update_frame_idx = int(frame_idx)

        rgb = self._load_rgb_for_timestamp(display_ts)
        if rgb is not None:
            self._upload_rgb_texture(rgb)
        (
            self._oak_media_bb2d_current_boxes,
            self._oak_media_bb2d_img_wh,
        ) = self._media_bb2d_for_timestamp(display_ts)

    def _process_detection_frame(self, frame_idx: int) -> None:
        ts = self.sorted_timestamps[frame_idx]
        detections = self._filter_frame_obbs(
            self.timed_obbs.get(ts, ObbTW(torch.zeros(0, 165)))
        )
        if len(detections) == 0:
            return
        cam, T_wr = self._get_cam_and_pose(ts)
        obs_pts = self._get_observed_points_trail(
            ts, trail_duration_ns=int(self.obs_trail_secs * 1e9)
        )
        self.tracker.update(
            detections,
            frame_idx,
            cam=cam,
            T_world_rig=T_wr,
            observed_points=obs_pts,
        )

    def _process_detections_between(self, start_idx: int, end_idx: int) -> None:
        if end_idx < start_idx:
            return
        start_idx = max(0, int(start_idx))
        end_idx = min(self.total_frames - 1, int(end_idx))
        for idx in range(start_idx, end_idx + 1):
            if idx in self._oak_detection_frame_indices:
                self._process_detection_frame(idx)

    def on_key_event(self, key, action, modifiers):
        """Playback keys advance time, but never take over the 3D orbit camera."""
        if key == self.wnd.keys.ESCAPE:
            if action == self.wnd.keys.ACTION_PRESS:
                self.is_playing = False
                self._force_free_view()
            return

        if action == self.wnd.keys.ACTION_PRESS and key in (
            self.wnd.keys.SPACE,
            self.wnd.keys.RIGHT,
            self.wnd.keys.LEFT,
        ):
            if self.total_frames == 0:
                return

            if key == self.wnd.keys.SPACE:
                if self.current_frame_idx >= self.total_frames - 1:
                    self._reset_tracker()
                    self._step_to_frame(0)
                    self.is_playing = True
                else:
                    self.is_playing = not self.is_playing
                self._last_step_time = time_module.time()
            elif key == self.wnd.keys.RIGHT:
                self.is_playing = False
                self._step_forward()
            elif key == self.wnd.keys.LEFT:
                self.is_playing = False
                if self.current_frame_idx > 0:
                    self._step_to_frame(self.current_frame_idx - 1)

            self._force_free_view()
            return

        super().on_key_event(key, action, modifiers)
        self._force_free_view()

    def _load_stereo_for_timestamp(
        self, side: str, ts_ns: int
    ) -> Optional[np.ndarray]:
        if len(self._stereo_pair_timestamps) == 0 or side not in ("left", "right"):
            return None
        idx = int(find_nearest2(self._stereo_pair_timestamps, int(ts_ns)))
        idx = max(0, min(idx, len(self._stereo_pairs) - 1))
        pair = self._stereo_pairs[idx]
        side_ts = int(pair[f"{side}_timestamp_ns"])
        key = (side, side_ts)
        if key in self._stereo_lru_cache:
            img = self._stereo_lru_cache[key]
            self._stereo_lru_cache.move_to_end(key)
            return img

        img_path = self._stereo_cache_dir / pair[f"{side}_path"]
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self._stereo_lru_cache[key] = img
        if len(self._stereo_lru_cache) > self._stereo_lru_max_items:
            self._stereo_lru_cache.popitem(last=False)
        return img

    def _upload_stereo_texture(self, side: str, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        tex = self._stereo_textures.get(side)
        if tex is None or self._stereo_tex_sizes.get(side) != (w, h):
            if tex is not None:
                self.imgui.remove_texture(tex)
                tex.release()
            tex = self.ctx.texture((w, h), 3, img.tobytes())
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.imgui.register_texture(tex)
            self._stereo_textures[side] = tex
            self._stereo_tex_sizes[side] = (w, h)
        else:
            tex.write(img.tobytes())

    def _update_stereo_textures(self, ts_ns: int) -> None:
        for side in ("left", "right"):
            img = self._load_stereo_for_timestamp(side, ts_ns)
            if img is not None:
                self._upload_stereo_texture(side, img)

    def _step_to_frame(self, target_idx: int) -> None:
        if self.total_frames == 0:
            self.current_frame_idx = 0
            self.is_playing = False
            return

        target_idx = max(0, min(int(target_idx), self.total_frames - 1))
        manual_step = not self.is_playing
        if manual_step:
            self._clear_rgb_hold()
        if target_idx < self.current_frame_idx:
            self._reset_tracker()
            self._process_detections_between(0, target_idx)
        elif target_idx == self.current_frame_idx:
            if not getattr(self.tracker, "tracks", []):
                self._process_detection_frame(target_idx)
        else:
            start_idx = self.current_frame_idx + 1
            if self.current_frame_idx == 0 and not getattr(self.tracker, "tracks", []):
                start_idx = 0
            self._process_detections_between(start_idx, target_idx)

        self.current_frame_idx = target_idx
        self._rebuild_current_view()

        ts = self.sorted_timestamps[self.current_frame_idx]
        hold_label_idx = self._oak_pending_rgb_label_frame_idx
        self._oak_pending_rgb_label_frame_idx = None
        self._set_rgb_panel_timestamp(
            ts,
            self.current_frame_idx,
            force=manual_step,
            hold_label_frame_idx=hold_label_idx,
        )
        self._update_stereo_textures(ts)

    def _rebuild_current_view(self):
        ts = self.sorted_timestamps[self.current_frame_idx]
        original = self.timed_obbs.get(ts)
        display_obbs = None
        if original is None or len(original) == 0:
            display_obbs = self._display_obbs_for_timestamp(ts)

        replaced = display_obbs is not None and len(display_obbs) > 0
        if replaced:
            self.timed_obbs[ts] = display_obbs
        try:
            super()._rebuild_current_view()
        finally:
            if replaced:
                if original is None:
                    self.timed_obbs.pop(ts, None)
                else:
                    self.timed_obbs[ts] = original

    def _next_auto_frame_index(self, frame_idx: int) -> int:
        if frame_idx >= self.total_frames - 1:
            return frame_idx
        return min(self.total_frames - 1, frame_idx + 1)

    def _advance_playback(self) -> None:
        if not self.is_playing or self.total_frames == 0:
            return
        if self.current_frame_idx >= self.total_frames - 1:
            self.is_playing = False
            return

        now = time_module.time()
        if self._last_step_time <= 0.0:
            self._last_step_time = now

        effective_fps = max(0.1, float(self.playback_fps) * self._oak_playback_speed)
        interval = 1.0 / effective_fps
        elapsed = now - self._last_step_time
        if elapsed < interval:
            return

        frames_due = max(1, min(30, int(elapsed / interval)))
        target_idx = self.current_frame_idx
        label_idx = None
        for _ in range(frames_due):
            next_idx = self._next_auto_frame_index(target_idx)
            if next_idx == target_idx:
                break
            target_idx = next_idx
            if self._frame_has_label(target_idx):
                label_idx = target_idx

        self._last_step_time = now
        if target_idx != self.current_frame_idx:
            self._oak_pending_rgb_label_frame_idx = label_idx
            self._step_to_frame(target_idx)

    def render_3d(self, time: float, frame_time: float) -> None:
        self._advance_playback()
        was_playing = self.is_playing
        self.is_playing = False
        try:
            super().render_3d(time, frame_time)
        finally:
            self.is_playing = was_playing and self.current_frame_idx < self.total_frames - 1

    def _compute_rgb_panel_width(self, win_w: int, panel_h: int) -> int:
        has_media = self._rgb_texture is not None or any(
            tex is not None for tex in self._stereo_textures.values()
        )
        if not self.show_rgb or not has_media:
            return 0
        return int(win_w * self.rgb_panel_max_frac)

    def render_ui(self) -> None:
        rgb_texture = self._rgb_texture
        rgb_tex_size = self._rgb_tex_size
        if rgb_texture is not None or any(
            tex is not None for tex in self._stereo_textures.values()
        ):
            self._rgb_texture = None
            self._rgb_tex_size = (0, 0)
            try:
                super().render_ui()
            finally:
                self._rgb_texture = rgb_texture
                self._rgb_tex_size = rgb_tex_size
            self._force_free_view()
            if self.show_rgb:
                self._render_oak_media_panel()
        else:
            super().render_ui()
            self._force_free_view()

    def _render_oak_media_panel(self) -> None:
        win_w, win_h = self.wnd.size
        panel_w = self._compute_rgb_panel_width(win_w, win_h)
        if panel_w <= 0:
            self._rgb_panel_rect = None
            return

        panel_x = self.ui_panel_width
        imgui.set_next_window_position(panel_x, 0, imgui.ALWAYS)
        imgui.set_next_window_size(panel_w, win_h, imgui.ALWAYS)
        expanded, _ = imgui.begin(
            "RGB + Stereo",
            flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE,
        )
        if expanded:
            avail_w, avail_h = imgui.get_content_region_available()
            label_h = imgui.calc_text_size("RGB").y + 5.0
            gap_h = 8.0
            image_h = max(1.0, (avail_h - 3.0 * label_h - 2.0 * gap_h) / 3.0)
            self._draw_media_image(
                "RGB",
                self._rgb_texture,
                self._rgb_tex_size,
                image_h,
                overlay_2d=True,
            )
            imgui.dummy(imgui.ImVec2(1.0, gap_h))
            self._draw_media_image(
                "Stereo Left",
                self._stereo_textures["left"],
                self._stereo_tex_sizes["left"],
                image_h,
                overlay_2d=False,
            )
            imgui.dummy(imgui.ImVec2(1.0, gap_h))
            self._draw_media_image(
                "Stereo Right",
                self._stereo_textures["right"],
                self._stereo_tex_sizes["right"],
                image_h,
                overlay_2d=False,
            )

        px, py = imgui.get_window_position()
        pw, ph = imgui.get_window_size()
        self._rgb_panel_rect = (px, py, pw, ph)
        imgui.end()

    def _draw_media_image(
        self,
        label: str,
        texture,
        tex_size: tuple[int, int],
        image_h: float,
        *,
        overlay_2d: bool,
    ) -> None:
        imgui.text(label)
        avail_w, _ = imgui.get_content_region_available()
        tex_w, tex_h = tex_size
        if texture is None or tex_w <= 0 or tex_h <= 0:
            imgui.dummy(imgui.ImVec2(max(1.0, avail_w), max(1.0, image_h)))
            return

        img_scale = min(avail_w / tex_w, image_h / tex_h)
        draw_w = tex_w * img_scale
        draw_h = tex_h * img_scale
        imgui.image(texture.glo, draw_w, draw_h)
        img_min = imgui.get_item_rect_min()
        if overlay_2d:
            self._draw_rgb_2d_overlay(img_min, draw_w, draw_h)
        leftover_h = image_h - draw_h
        if leftover_h > 1.0:
            imgui.dummy(imgui.ImVec2(1.0, leftover_h))

    def _draw_rgb_2d_overlay(self, img_min, draw_w: float, draw_h: float) -> None:
        if not self.show_bb2_csv or not self._oak_media_bb2d_current_boxes:
            return
        csv_w, csv_h = self._oak_media_bb2d_img_wh
        if csv_w <= 0 or csv_h <= 0:
            return
        csv_sx = draw_w / csv_w
        csv_sy = draw_h / csv_h
        draw_list = imgui.get_window_draw_list()
        box_col = imgui.get_color_u32_rgba(0.0, 1.0, 0.0, 1.0)
        text_col = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0)
        bg_col = imgui.get_color_u32_rgba(0.0, 0.35, 0.0, 0.65)
        label_scale = max(0.5, min(1.0, float(self.rgb_text_scale)))
        imgui.set_window_font_scale(label_scale)
        for x1, y1, x2, y2, label, _sem_id in self._oak_media_bb2d_current_boxes:
            rx0 = img_min.x + x1 * csv_sx
            ry0 = img_min.y + y1 * csv_sy
            rx1 = img_min.x + x2 * csv_sx
            ry1 = img_min.y + y2 * csv_sy
            draw_list.add_rect(
                rx0,
                ry0,
                rx1,
                ry1,
                box_col,
                0.0,
                0,
                self.rgb_bb2_thickness,
            )
            tw, th = imgui.calc_text_size(label)
            draw_list.add_rect_filled(rx0 - 1, ry0 - th - 2, rx0 + tw + 2, ry0, bg_col)
            draw_list.add_text(rx0, ry0 - th - 1, text_col, label)
        imgui.set_window_font_scale(1.0)

    def _load_rgb_for_timestamp(self, ts_ns: int) -> Optional[np.ndarray]:
        if len(self._rgb_timestamps) == 0:
            return None

        idx = int(find_nearest2(self._rgb_timestamps, ts_ns))
        idx = max(0, min(idx, len(self._rtab_loader.rows) - 1))
        ts_key = int(self._rgb_timestamps[idx])
        if ts_key in self._rgb_lru_cache:
            img = self._rgb_lru_cache[ts_key]
            self._rgb_lru_cache.move_to_end(ts_key)
        else:
            row = self._rtab_loader.rows[idx]
            img_path = self._rtab_loader.export_dir / row["image_path"]
            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                return None
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            self._rgb_lru_cache[ts_key] = img
            if len(self._rgb_lru_cache) > self._rgb_lru_max_items:
                self._rgb_lru_cache.popitem(last=False)

        self._rgb_vrs_h, self._rgb_vrs_w = img.shape[:2]
        h, w = img.shape[:2]
        target_h = 1200
        scale = target_h / h
        self._rgb_img_scale = scale
        target_w = int(w * scale)
        return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    def _load_semidense_points(self) -> None:
        positions = _downsample_points(
            self._rtab_loader.voxel_centers, self._rtab_max_points
        )
        if len(positions) == 0:
            print("No OAK RTAB voxel points found")
            self.point_count = 0
            self._point_positions = None
            self.show_global_points = False
            return

        self._point_positions = positions
        self.point_count = len(positions)
        self.show_global_points = True

        colors = np.full((self.point_count, 3), 0.25, dtype=np.float32)
        vertex_data = np.hstack([positions, colors]).astype(np.float32)
        self.point_vbo = self.ctx.buffer(vertex_data.tobytes())
        self.point_vao = self.ctx.vertex_array(
            self.point_prog,
            [(self.point_vbo, "3f 3f", "in_position", "in_color")],
        )
        self._all_obs_timestamps = np.asarray(self._rgb_timestamps, dtype=np.int64)
        print(f"Loaded {self.point_count:,} OAK RTAB voxel points")

    def _get_observed_points(self, ts_ns: int) -> Optional[torch.Tensor]:
        if len(self._rgb_timestamps) == 0:
            return None
        cam, T_wr = self._get_cam_and_pose(ts_ns)
        if cam is not None and T_wr is not None:
            try:
                T_world_cam_pose = T_wr @ cam.T_camera_rig.inverse()
            except Exception:
                T_world_cam_pose = T_wr
            T_world_cam = T_world_cam_pose.matrix.cpu().numpy().astype(np.float32)
        else:
            idx = int(find_nearest2(self._rgb_timestamps, ts_ns))
            idx = max(0, min(idx, len(self._rtab_loader.rows) - 1))
            row = self._rtab_loader.rows[idx]
            T_world_cam = _matrix_from_row(row, "map_rgb_optical").astype(np.float32)
        _, width, height, fx, fy, cx, cy = self._rtab_loader._camera_for_resize()
        pts = self._rtab_loader._visible_sdp(T_world_cam, width, height, fx, fy, cx, cy)
        return pts if pts is not None and len(pts) > 0 else None

    def _get_observed_points_trail(
        self, ts_ns: int, trail_duration_ns: int = 200_000_000
    ) -> Optional[torch.Tensor]:
        if trail_duration_ns <= 0 or len(self._rgb_timestamps) == 0:
            return self._get_observed_points(ts_ns)

        start_ts = int(ts_ns) - int(trail_duration_ns)
        timestamps = np.asarray(self._rgb_timestamps, dtype=np.int64)
        idx_hi = int(np.searchsorted(timestamps, int(ts_ns), side="right"))
        idx_lo = int(np.searchsorted(timestamps, start_ts, side="left"))
        idx_lo = max(0, min(idx_lo, len(timestamps)))
        idx_hi = max(idx_lo, min(idx_hi, len(timestamps)))
        if idx_hi <= idx_lo:
            return self._get_observed_points(ts_ns)

        parts = []
        for obs_ts in timestamps[idx_lo:idx_hi]:
            pts = self._get_observed_points(int(obs_ts))
            if pts is not None and len(pts) > 0:
                parts.append(pts)
        if not parts:
            return None
        pts_cat = torch.cat(parts, dim=0)
        if len(pts_cat) > self._rtab_loader.max_sdp_points:
            idx = torch.linspace(
                0, len(pts_cat) - 1, self._rtab_loader.max_sdp_points
            ).long()
            pts_cat = pts_cat[idx]
        return pts_cat

    def _recolor_global_points_by_tracks(self) -> None:
        # Keep the RTAB map as a stable context layer; current observed voxels are
        # emphasized separately, so per-frame recoloring the full map is unnecessary.
        return

    def _prime_first_frame(self) -> None:
        if self.total_frames == 0:
            return
        self.current_frame_idx = 0
        ts = self.sorted_timestamps[0]
        detections = self._filter_frame_obbs(
            self.timed_obbs.get(ts, ObbTW(torch.zeros(0, 165)))
        )
        cam, T_wr = self._get_cam_and_pose(ts)
        obs_pts = self._get_observed_points_trail(
            ts, trail_duration_ns=int(self.obs_trail_secs * 1e9)
        )
        self.tracker.update(
            detections,
            0,
            cam=cam,
            T_world_rig=T_wr,
            observed_points=obs_pts,
        )
        self._rebuild_current_view()
        self._set_rgb_panel_timestamp(ts, 0, force=True)
        self._update_stereo_textures(ts)


def _load_timed_obbs(csv_path: str, args) -> dict[int, ObbTW]:
    if not os.path.exists(csv_path):
        raise IOError(f"3D BB CSV not found: {csv_path}")
    timed_obbs = read_obb_csv(csv_path)
    timed_obbs = subsample_timed_obbs(
        timed_obbs, skip_n=args.skip_n, start_n=args.start_n, max_n=args.max_n
    )
    if not timed_obbs:
        raise ValueError(f"No OBB frames selected from {csv_path}")
    return timed_obbs


def _make_loader(args, input_path: str, log_dir: str) -> OakRtabLoader:
    return OakRtabLoader(
        input_path,
        start_frame=args.loader_start,
        skip_frames=args.loader_skip,
        max_frames=args.loader_max if args.loader_max > 0 else None,
        resize=None,
        cache_dir=log_dir,
        voxel_size=args.oak_voxel_size,
        hash_cell_size=args.oak_hash_cell_size,
        visibility_near=args.oak_visibility_near,
        visibility_far=args.oak_visibility_far,
        zbuffer_tolerance=args.oak_zbuffer_tolerance,
        max_sdp_points=args.oak_max_sdp_points,
        zbuffer_grid=args.oak_zbuffer_grid,
    )


def _dry_run(args, input_path: str, log_dir: str, csv_path: str, bb2d_csv_path: str):
    loader = _make_loader(args, input_path, log_dir)
    timed_obbs = _load_timed_obbs(csv_path, args)
    all_obbs = _stack_obbs(timed_obbs)
    stereo_manifest = Path(args.stereo_manifest).expanduser()
    odom_path = Path(args.odom_trajectory).expanduser()
    odom_frames = 0
    if odom_path.is_file():
        with np.load(odom_path) as data:
            odom_frames = int(len(data["timestamps_ns"]))

    print("\nOAK RTAB incremental viewer dry run")
    print(f"  input:        {input_path}")
    print(f"  output dir:   {log_dir}")
    print(f"  3D CSV:       {csv_path}")
    print(f"  2D CSV:       {bb2d_csv_path if os.path.exists(bb2d_csv_path) else '(missing)'}")
    print(
        f"  stereo:       {stereo_manifest if stereo_manifest.is_file() else '(missing)'}"
    )
    print(f"  odom 30Hz:    {odom_path if odom_path.is_file() else '(missing)'}")
    print(f"  OAK frames:   {len(loader.rows)}")
    print(f"  OBB frames:   {len(timed_obbs)}")
    print(f"  OBB boxes:    {len(all_obbs)}")
    if odom_frames > 0:
        print(f"  odom frames:  {odom_frames}")
    print(
        f"  playback:     {args.playback_fps:.1f}Hz x{args.playback_speed:.2f}, "
        f"rgb_hold={args.label_hold_sec:.2f}s, rgb_stride={args.normal_stride}"
    )
    print(f"  media panel:  {args.media_panel_frac:.2f} window fraction")
    print(f"  voxel points: {len(loader.voxel_centers):,}")
    print(f"  voxel cache:  {loader.voxel_cache_path}")
    if timed_obbs:
        ts = np.array(sorted(timed_obbs.keys()), dtype=np.int64)
        print(f"  first OBB ts: {int(ts[0])}")
        print(f"  last OBB ts:  {int(ts[-1])}")
    if len(loader.rows) > 0:
        loader_ts = np.array(
            [int(row["rgb_stamp_ns"]) for row in loader.rows], dtype=np.int64
        )
        if timed_obbs and (ts[0] < loader_ts[0] or ts[-1] > loader_ts[-1]):
            print("  warning: selected OBB timestamps extend beyond loaded OAK frames")


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Incremental OAK RTAB Boxer demo viewer")
    parser.add_argument("--input", type=str, default=DEFAULT_RTAB_EXPORT, help="OAK RTAB export directory")
    parser.add_argument("--output_dir", type=str, default=EVAL_PATH, help="Where Boxer CSVs live")
    parser.add_argument("--write_name", default="boxer", type=str, help="CSV prefix")
    parser.add_argument("--skip_n", type=int, default=1, help="subsample loaded OBB timestamps")
    parser.add_argument("--start_n", type=int, default=0, help="start from n-th OBB timestamp")
    parser.add_argument("--max_n", type=int, default=0, help="max OBB timestamps to replay (0 = all)")
    parser.add_argument("--loader_skip", type=int, default=1, help="subsample OAK RGB frames for trajectory/RGB lookup")
    parser.add_argument("--loader_start", type=int, default=1, help="1-based OAK RGB start frame")
    parser.add_argument("--loader_max", type=int, default=0, help="max OAK RGB frames to load (0 = all)")
    parser.add_argument("--window_w", type=int, default=0, help="Initial window width (0 = default)")
    parser.add_argument("--window_h", type=int, default=0, help="Initial window height (0 = default)")
    parser.add_argument("--load_view", type=str, nargs="?", const="DEFAULT", default=None)
    parser.add_argument("--init_follow", action="store_true", help="Accepted for compatibility; this viewer keeps the 3D point-cloud view free")
    parser.add_argument("--autoplay", action="store_true", help="Automatically start playback")
    parser.add_argument("--playback_fps", type=float, default=30.0, help="Base playback FPS for 30Hz trajectory")
    parser.add_argument("--playback_speed", type=float, default=1.0, help="Playback speed multiplier, e.g. 4.0 for 4x")
    parser.add_argument("--label_hold_sec", type=float, default=0.25, help="RGB-only hold duration on frames with 2D labels / Boxer detections")
    parser.add_argument("--normal_stride", type=int, default=1, help="RGB-only update stride for normal no-label pose frames")
    parser.add_argument("--raw_hold_ms", type=float, default=160.0, help="Keep per-frame 3D BBXs visible after detection frames")
    parser.add_argument("--media_panel_frac", type=float, default=0.30, help="Right RGB+stereo media panel width fraction")
    parser.add_argument("--max_points", type=int, default=250000, help="max RTAB voxel points rendered in 3D (0 = all)")
    parser.add_argument("--point_size", type=float, default=2.0, help="initial RTAB voxel point size")
    parser.add_argument("--point_alpha", type=float, default=0.25, help="initial RTAB voxel point alpha")
    parser.add_argument("--raw_conf", type=float, default=0.45, help="per-frame 3D detection confidence threshold")
    parser.add_argument("--fused_conf", type=float, default=0.40, help="online fused/tracked 3D box confidence threshold")
    parser.add_argument("--fusion_min_hits", type=int, default=2, help="matches needed before a fused track is shown")
    parser.add_argument("--fusion_iou", type=float, default=0.20, help="online fusion/tracker 3D IoU match threshold")
    parser.add_argument("--fusion_max_missed", type=int, default=90, help="visible unmatched frames before a fused track can be removed")
    parser.add_argument("--bb2d_csv", type=str, default="", help="2D BB CSV path or filename relative to output sequence dir")
    parser.add_argument("--stereo_manifest", type=str, default="", help="stereo manifest path (default: output sequence stereo_cache/stereo_manifest.json)")
    parser.add_argument("--odom_trajectory", type=str, default="", help="30Hz odom npz path (default: output sequence odom_trajectory_30hz.npz)")
    parser.add_argument("--no_smooth_trajectory", action="store_true", help="disable 30Hz odom playback timeline")
    parser.add_argument("--dry_run", action="store_true", help="validate inputs and print counts without opening the GUI")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose tracker logging")
    parser.add_argument("--oak_voxel_size", type=float, default=0.05, help="OAK RTAB voxel size in meters")
    parser.add_argument("--oak_hash_cell_size", type=float, default=1.0, help="OAK RTAB spatial hash cell size in meters")
    parser.add_argument("--oak_visibility_near", type=float, default=0.15, help="OAK RTAB near visibility cutoff in meters")
    parser.add_argument("--oak_visibility_far", type=float, default=6.0, help="OAK RTAB far visibility cutoff in meters")
    parser.add_argument("--oak_zbuffer_tolerance", type=float, default=None, help="OAK RTAB z-buffer visibility tolerance in meters")
    parser.add_argument("--oak_max_sdp_points", type=int, default=100000, help="OAK RTAB max visible SDP points per frame")
    parser.add_argument("--oak_zbuffer_grid", type=int, default=2, help="OAK RTAB z-buffer grid size in pixels")
    # fmt: on
    args = parser.parse_args()

    input_path = os.path.abspath(os.path.expanduser(args.input))
    if not _is_oak_rtab_export(input_path):
        raise IOError(
            "Expected an OAK RTAB export directory with metadata.json and "
            f"poses/rgb_poses.csv: {input_path}"
        )

    seq_name = os.path.basename(input_path.rstrip("/"))
    output_dir = os.path.expanduser(args.output_dir)
    log_dir = os.path.join(output_dir, seq_name)
    os.makedirs(log_dir, exist_ok=True)

    csv_path = os.path.join(log_dir, f"{args.write_name}_3dbbs.csv")
    if args.bb2d_csv:
        bb2d_csv_path = (
            args.bb2d_csv
            if os.path.isabs(args.bb2d_csv)
            else os.path.join(log_dir, args.bb2d_csv)
        )
    else:
        bb2d_csv_path = os.path.join(log_dir, "owl_2dbbs.csv")
    if args.stereo_manifest:
        args.stereo_manifest = (
            args.stereo_manifest
            if os.path.isabs(args.stereo_manifest)
            else os.path.join(log_dir, args.stereo_manifest)
        )
    else:
        args.stereo_manifest = os.path.join(
            log_dir, "stereo_cache", "stereo_manifest.json"
        )
    if args.odom_trajectory:
        args.odom_trajectory = (
            args.odom_trajectory
            if os.path.isabs(args.odom_trajectory)
            else os.path.join(log_dir, args.odom_trajectory)
        )
    else:
        args.odom_trajectory = os.path.join(log_dir, "odom_trajectory_30hz.npz")

    if args.dry_run:
        _dry_run(args, input_path, log_dir, csv_path, bb2d_csv_path)
        return

    loader = _make_loader(args, input_path, log_dir)
    use_odom = (not args.no_smooth_trajectory) and os.path.exists(args.odom_trajectory)
    seq_ctx = _build_oak_seq_ctx(loader, args.odom_trajectory if use_odom else "")
    timed_obbs = _load_timed_obbs(csv_path, args)
    if use_odom:
        before_frames = len(timed_obbs)
        timed_obbs = _expand_timed_obbs_to_nav_timeline(
            timed_obbs,
            np.asarray(seq_ctx["pose_ts"], dtype=np.int64),
        )
        print(
            f"==> 30Hz playback timeline: {before_frames} detection frames "
            f"expanded to {len(timed_obbs)} pose frames"
        )

    total_dets = _count_obbs(timed_obbs)
    print(
        f"==> OAK incremental replay: {len(timed_obbs)} OBB frames, "
        f"{total_dets} boxes, {len(loader.rows)} OAK RGB frames"
    )
    if not os.path.exists(bb2d_csv_path):
        bb2d_csv_path = ""

    view_path, load_view_data = load_view_file(log_dir, args.load_view)

    default_w, default_h = 2250 * scale_factor, 1100 * scale_factor
    init_w = args.window_w if args.window_w > 0 else default_w
    init_h = args.window_h if args.window_h > 0 else default_h

    class Viewer(OakRtabIncrementalViewer):
        window_size = (init_w, init_h)

        def __init__(self, **kw):
            super().__init__(
                rtab_loader=loader,
                max_points=args.max_points,
                point_size=args.point_size,
                point_alpha=args.point_alpha,
                raw_conf=args.raw_conf,
                fused_conf=args.fused_conf,
                fusion_min_hits=args.fusion_min_hits,
                fusion_iou=args.fusion_iou,
                fusion_max_missed=args.fusion_max_missed,
                playback_fps=args.playback_fps,
                playback_speed=args.playback_speed,
                label_hold_sec=args.label_hold_sec,
                normal_stride=args.normal_stride,
                raw_hold_ms=args.raw_hold_ms,
                media_panel_frac=args.media_panel_frac,
                stereo_manifest_path=args.stereo_manifest,
                timed_obbs=timed_obbs,
                root_path=log_dir,
                seq_ctx=seq_ctx,
                bb2d_csv_path=bb2d_csv_path,
                init_follow=False,
                init_show_obs=True,
                init_image_panel_width=args.media_panel_frac,
                verbose=args.verbose,
                load_view_data=load_view_data,
                view_save_path=view_path,
                **kw,
            )
            if args.autoplay:
                self.is_playing = True
                self._force_free_view()

    launch_viewer(Viewer)


if __name__ == "__main__":
    main()
