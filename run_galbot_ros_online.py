#! /usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Run Boxer online from ROS2 subscriptions for Galbot/Isaac bags.

This runner is intentionally separate from run_boxer.py. It expects the user to
play a ROS2 bag manually, subscribes to the head camera topics, synchronizes
RGB/depth/mask messages, and feeds a bounded queue consumed by BoxerNet.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import queue
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import cv2
import numpy as np
import torch

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.qos import qos_profile_sensor_data
except ModuleNotFoundError:
    rclpy = None

    class Node:  # type: ignore[no-redef]
        pass

    QoSProfile = None
    ReliabilityPolicy = None
    DurabilityPolicy = None
    qos_profile_sensor_data = None

from loaders.base_loader import BaseLoader
from loaders.galbot_loader import (
    DEFAULT_INTRINSICS,
    INTRINSIC_NAMES,
    GalbotLoader,
    normalize_galbot_camera,
)
from utils.demo_utils import CKPT_PATH, DEFAULT_BOXERNET_CKPT, EVAL_PATH, CudaTimer
from utils.file_io import ObbCsvWriter2, save_bb2d_csv
from utils.taxonomy import load_text_labels
from utils.tw.obb import ObbTW
from utils.tw.pose import PoseTW
from utils.tw.tensor_utils import pad_string, string2tensor


DEFAULT_DATA_DIR = (
    "/home/agxi/Datasets/output/galbot_home/"
    "galbot_slam_home_b_20260519_202249"
)
ROBOT_BBOX_MASK_OVERLAP = 0.25
ROBOT_BBOX_CENTER_OVERLAP = 0.50
ROBOT_MASK_DILATE_PX = 3


def comma_separated_list(value):
    if not value:
        return []
    if isinstance(value, list):
        return value
    return [x.strip() for x in value.split(",") if x.strip()]


@dataclass
class FramePacket:
    frame_idx: int
    time_ns: int
    datum: dict[str, Any]
    img_rgb: np.ndarray
    depth_valid: int
    depth_total: int
    intrinsics_source: str
    tf_ready: bool
    pose_mode: str
    camera_pose_convention: str
    sync_depth_delta_ms: float
    sync_mask_delta_ms: Optional[float]
    robot_mask: Optional[np.ndarray]


@dataclass
class OnlineResult:
    frame_idx: int
    time_ns: int
    raw_obbs: ObbTW
    active_obbs: ObbTW
    active_tracks: list[SimpleNamespace]
    rgb_image: np.ndarray
    bb2d_xyxy: np.ndarray
    bb2d_labels: list[str]
    bb2d_scores: np.ndarray
    camera_position: np.ndarray
    camera_rpy_deg: np.ndarray
    tf_ready: bool
    pose_mode: str
    camera_pose_convention: str
    num_2d: int
    num_2d_raw: int
    num_2d_filtered_robot: int
    num_3d: int
    num_tracks: int
    labels: list[str]
    scores: list[float]
    timing_ms: dict[str, float]


class LiveState:
    """Thread-safe state shared by ROS callbacks, inference, and viewer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.latest_packet: Optional[FramePacket] = None
        self.latest_result: Optional[OnlineResult] = None
        self.packet_version = 0
        self.result_version = 0
        self.errors: list[str] = []
        self.stats = {
            "rgb": 0,
            "depth": 0,
            "mask": 0,
            "synced": 0,
            "queued": 0,
            "processed": 0,
            "dropped_queue": 0,
            "dropped_sync": 0,
            "dropped_throttle": 0,
            "dropped_intrinsics": 0,
            "dropped_depth": 0,
            "tf_missing": 0,
            "filtered_robot_bbox": 0,
        }

    def bump(self, key: str, inc: int = 1) -> None:
        with self._lock:
            self.stats[key] = self.stats.get(key, 0) + inc

    def set_packet(self, packet: FramePacket) -> None:
        with self._lock:
            self.latest_packet = packet
            self.packet_version += 1

    def set_result(self, result: OnlineResult) -> None:
        with self._lock:
            self.latest_result = result
            self.result_version += 1
            self.stats["processed"] = self.stats.get("processed", 0) + 1

    def add_error(self, message: str) -> None:
        with self._lock:
            self.errors.append(message)
            self.errors = self.errors[-8:]

    def snapshot(self) -> tuple[dict[str, int], int, Optional[OnlineResult], list[str]]:
        with self._lock:
            return (
                dict(self.stats),
                self.result_version,
                self.latest_result,
                list(self.errors),
            )


def _stamp_to_ns(header) -> int:
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return 0
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _as_array_buffer(data, dtype):
    try:
        return np.frombuffer(data, dtype=dtype)
    except TypeError:
        return np.asarray(data, dtype=dtype)


def decode_rgb_image(msg) -> np.ndarray:
    encoding = str(msg.encoding).lower()
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    if encoding in ("rgb8", "bgr8"):
        arr = _as_array_buffer(msg.data, np.uint8).reshape(h, step)
        img = arr[:, : w * 3].reshape(h, w, 3)
        if encoding == "bgr8":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.copy()
    if encoding in ("rgba8", "bgra8"):
        arr = _as_array_buffer(msg.data, np.uint8).reshape(h, step)
        img = arr[:, : w * 4].reshape(h, w, 4)
        code = cv2.COLOR_RGBA2RGB if encoding == "rgba8" else cv2.COLOR_BGRA2RGB
        return cv2.cvtColor(img, code)
    if encoding in ("mono8", "8uc1"):
        arr = _as_array_buffer(msg.data, np.uint8).reshape(h, step)
        gray = arr[:, :w]
        return np.repeat(gray[:, :, None], 3, axis=2).copy()
    raise ValueError(f"Unsupported RGB image encoding: {msg.encoding}")


def decode_depth_image(msg) -> Optional[np.ndarray]:
    encoding = str(msg.encoding).lower()
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    if encoding in ("16uc1", "mono16"):
        dtype = ">u2" if int(msg.is_bigendian) else "<u2"
        row_elems = step // np.dtype(dtype).itemsize
        arr = _as_array_buffer(msg.data, dtype).reshape(h, row_elems)
        return (arr[:, :w].astype(np.float32) / 1000.0).copy()
    if encoding == "32fc1":
        dtype = ">f4" if int(msg.is_bigendian) else "<f4"
        row_elems = step // np.dtype(dtype).itemsize
        arr = _as_array_buffer(msg.data, dtype).reshape(h, row_elems)
        depth = arr[:, :w].astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        return depth.copy()
    return None


def decode_mask_image(msg) -> Optional[np.ndarray]:
    encoding = str(msg.encoding).lower()
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    if encoding in ("mono8", "8uc1"):
        arr = _as_array_buffer(msg.data, np.uint8).reshape(h, step)
        return arr[:, :w].copy()
    if encoding in ("16uc1", "mono16"):
        dtype = ">u2" if int(msg.is_bigendian) else "<u2"
        row_elems = step // np.dtype(dtype).itemsize
        arr = _as_array_buffer(msg.data, dtype).reshape(h, row_elems)
        return arr[:, :w].copy()
    if encoding in ("rgb8", "bgr8", "rgba8", "bgra8"):
        rgb = decode_rgb_image(msg)
        return np.max(rgb, axis=2).astype(np.uint8)
    return None


def _frame_key(frame_id: str) -> str:
    return str(frame_id).strip("/")


def _frame_key_candidates(frame_id: str) -> list[str]:
    key = _frame_key(frame_id)
    candidates = [key]
    if "/" in key:
        candidates.append(key.split("/")[-1])
    if key.startswith("galbot_one_golf/"):
        candidates.append(key[len("galbot_one_golf/") :])
    out = []
    for candidate in candidates:
        if candidate and candidate not in out:
            out.append(candidate)
    return out


def _quat_xyzw_to_R(x, y, z, w):
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _invert_transform(T: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float32)
    R = T[:3, :3].astype(np.float32)
    t = T[:3, 3].astype(np.float32)
    out[:3, :3] = R.T
    out[:3, 3] = -(R.T @ t)
    return out


def _nearest_key(buf: dict[int, Any], ts: int) -> Optional[int]:
    if not buf:
        return None
    return min(buf.keys(), key=lambda key: abs(key - ts))


def _empty_obbs() -> ObbTW:
    return ObbTW(torch.zeros(0, 165))


def _robot_bbox_keep_mask(
    bb2d: torch.Tensor,
    robot_mask: Optional[np.ndarray],
    img_h: int,
    img_w: int,
) -> torch.Tensor:
    if len(bb2d) == 0:
        return torch.zeros(0, dtype=torch.bool)
    if robot_mask is None:
        return torch.ones(len(bb2d), dtype=torch.bool)

    mask = np.asarray(robot_mask)
    if mask.ndim == 3:
        mask = np.max(mask, axis=2)
    if mask.shape[:2] != (img_h, img_w):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (img_w, img_h),
            interpolation=cv2.INTER_NEAREST,
        )
    robot = mask > 0
    if ROBOT_MASK_DILATE_PX > 0:
        k = ROBOT_MASK_DILATE_PX * 2 + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        robot = cv2.dilate(robot.astype(np.uint8), kernel, iterations=1).astype(bool)

    keep = []
    boxes = bb2d.detach().cpu().numpy()
    for box in boxes:
        x0_f, x1_f, y0_f, y1_f = [float(v) for v in box]
        x0 = max(0, min(img_w, int(np.floor(min(x0_f, x1_f)))))
        x1 = max(0, min(img_w, int(np.ceil(max(x0_f, x1_f)))))
        y0 = max(0, min(img_h, int(np.floor(min(y0_f, y1_f)))))
        y1 = max(0, min(img_h, int(np.ceil(max(y0_f, y1_f)))))
        if x1 <= x0 or y1 <= y0:
            keep.append(False)
            continue

        crop = robot[y0:y1, x0:x1]
        overlap = float(np.count_nonzero(crop)) / float(crop.size)

        cx0 = x0 + (x1 - x0) // 4
        cx1 = x1 - (x1 - x0) // 4
        cy0 = y0 + (y1 - y0) // 4
        cy1 = y1 - (y1 - y0) // 4
        center = robot[cy0:cy1, cx0:cx1]
        center_overlap = (
            float(np.count_nonzero(center)) / float(center.size)
            if center.size > 0
            else overlap
        )
        keep.append(
            overlap < ROBOT_BBOX_MASK_OVERLAP
            and center_overlap < ROBOT_BBOX_CENTER_OVERLAP
        )
    return torch.tensor(keep, dtype=torch.bool)


def _pose_data_to_position_rpy_deg(T_world_rig: PoseTW) -> tuple[np.ndarray, np.ndarray]:
    data = T_world_rig._data.detach().cpu().numpy().reshape(-1)
    R = data[:9].reshape(3, 3)
    t = data[9:12].astype(np.float32)
    sy = float(np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]))
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    rpy = np.degrees(np.array([roll, pitch, yaw], dtype=np.float32))
    return t, rpy


def _stack_obbs_or_empty(items) -> ObbTW:
    items = list(items)
    if not items:
        return _empty_obbs()
    return torch.stack(items)


def _load_json_intrinsics(
    data_dir: str,
    camera: str,
    rgb_topic: str,
    camera_info_topic: str,
    camera_frame: str,
) -> tuple[Optional[tuple[float, float, float, float]], str]:
    intrinsic_name = INTRINSIC_NAMES[camera]
    candidates = [
        os.path.join(data_dir, "parameters", "sensor", intrinsic_name),
        os.path.join(data_dir, intrinsic_name),
    ]

    for path in candidates:
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            intrinsics = GalbotLoader._intrinsics_from_data(data)
            if intrinsics is not None:
                return intrinsics, path

    info_path = os.path.join(data_dir, "recording_info.json")
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info = json.load(f)
        cam_info = info.get("camera_info", {}).get(camera, {})
        intrinsics = GalbotLoader._intrinsics_from_data(cam_info.get("intrinsic"))
        if intrinsics is not None:
            return intrinsics, info_path

    combined_path = os.path.join(data_dir, "camera_intrinsics.json")
    if os.path.exists(combined_path):
        with open(combined_path, "r") as f:
            combined = json.load(f)
        for name, camera_data in combined.get("cameras", {}).items():
            topic = camera_data.get("rgb_topic") or camera_data.get("topic")
            info_topic = camera_data.get("camera_info_topic")
            prim = camera_data.get("camera_prim") or camera_data.get("camera_frame")
            cam_name = str(camera_data.get("camera_name", "")).lower()
            matched = (
                topic == rgb_topic
                or info_topic == camera_info_topic
                or (prim and _frame_key(prim) == _frame_key(camera_frame))
                or (camera == "head" and "head" in cam_name)
                or (camera == "hand_left" and "left" in cam_name)
                or (camera == "hand_right" and "right" in cam_name)
            )
            if not matched:
                continue
            intrinsics = GalbotLoader._intrinsics_from_data(camera_data)
            if intrinsics is not None:
                return intrinsics, f"{combined_path}:{name}"

    return None, ""


def build_boxer_datum(
    *,
    img_rgb: np.ndarray,
    depth_np: Optional[np.ndarray],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    time_ns: int,
    T_world_cam: Optional[np.ndarray],
    resize_hw: int,
) -> dict[str, Any]:
    src_h, src_w = img_rgb.shape[:2]
    if resize_hw > 0:
        resize_h = resize_w = int(resize_hw)
        scale_x = resize_w / src_w
        scale_y = resize_h / src_h
        img_rgb = cv2.resize(
            img_rgb, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR
        )
        if depth_np is not None and depth_np.shape[:2] != (resize_h, resize_w):
            depth_np = cv2.resize(
                depth_np, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST
            )
    else:
        resize_h, resize_w = src_h, src_w
        scale_x = scale_y = 1.0

    fx_s = float(fx) * scale_x
    fy_s = float(fy) * scale_y
    cx_s = float(cx) * scale_x
    cy_s = float(cy) * scale_y

    if T_world_cam is None:
        T_wr_data = torch.tensor(
            [1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0], dtype=torch.float32
        )
        R_wc = T_wr_data[:9].reshape(3, 3).numpy().astype(np.float32)
        t_wc = T_wr_data[9:].numpy().astype(np.float32)
    else:
        R_wc = T_world_cam[:3, :3].astype(np.float32)
        t_wc = T_world_cam[:3, 3].astype(np.float32)
        T_wr_data = torch.tensor([*R_wc.reshape(-1), *t_wc], dtype=torch.float32)

    if depth_np is not None:
        sdp_w = BaseLoader.sdp_from_depth(depth_np, fx_s, fy_s, cx_s, cy_s, R_wc, t_wc)
    else:
        sdp_w = torch.zeros(0, 3, dtype=torch.float32)

    return {
        "img0": BaseLoader.img_to_tensor(img_rgb).float(),
        "cam0": BaseLoader.pinhole_from_K(
            resize_w,
            resize_h,
            fx_s,
            fy_s,
            cx_s,
            cy_s,
            valid_radius=(resize_w, resize_h),
        ).float(),
        "T_world_rig0": PoseTW(T_wr_data),
        "sdp_w": sdp_w,
        "time_ns0": int(time_ns),
        "rotated0": torch.tensor(False).reshape(1),
        "bb2d0": torch.zeros(0, 4, dtype=torch.float32),
        "gt_labels": [],
        "obbs": _empty_obbs(),
    }


class GalbotRosOnlineNode(Node):
    """ROS2 subscriber node that emits synchronized Boxer frame packets."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        frame_queue: "queue.Queue[FramePacket]",
        state: LiveState,
        resize_hw: int,
    ) -> None:
        super().__init__("galbot_boxer_online")
        self.args = args
        self.frame_queue = frame_queue
        self.state = state
        self.resize_hw = int(resize_hw)
        self.data_dir = os.path.expanduser(args.data_dir)
        self.camera = normalize_galbot_camera(args.camera)
        self.sync_slop_ns = int(float(args.sync_slop_ms) * 1e6)
        self.buffer_size = max(4, int(args.buffer_size))
        self.start_index = max(0, int(args.start_n) - 1)
        self.skip_n = max(1, int(args.skip_n))
        self.max_frames = max(0, int(args.max_frames))
        self.disable_robot_mask = bool(args.disable_robot_mask)
        self.mask_threshold = float(args.mask_threshold)
        self.max_fps = float(args.max_fps)
        self._last_emit_wall = 0.0
        self.pose_mode = str(args.pose_mode)

        self.camera_frame, self.rgb_topic, self.depth_topic, self.camera_info_topic = (
            self._load_topics()
        )
        self.camera_pose_convention = self._resolve_camera_pose_convention(
            args.camera_pose_convention
        )
        self.mask_topic = self._load_mask_topic()
        self.tf_topic = "/tf"
        self.tf_static_topic = "/tf_static"

        self.fx = self.fy = self.cx = self.cy = None
        self.intrinsics_source = "uninitialized"
        self._warned_zero_camera_info = False
        self._load_intrinsics_fallback()

        self._rgb_buf: dict[int, np.ndarray] = {}
        self._depth_buf: dict[int, np.ndarray] = {}
        self._mask_buf: dict[int, np.ndarray] = {}
        self._buf_lock = threading.Lock()
        self._tf_by_child: dict[str, tuple[str, np.ndarray]] = {}
        self._synced_seen = 0
        self._emitted = 0

        from sensor_msgs.msg import CameraInfo, Image
        from tf2_msgs.msg import TFMessage

        sensor_qos = qos_profile_sensor_data
        tf_static_qos = QoSProfile(depth=10)
        tf_static_qos.reliability = ReliabilityPolicy.RELIABLE
        tf_static_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.create_subscription(Image, self.rgb_topic, self._on_rgb, sensor_qos)
        if self.depth_topic:
            self.create_subscription(Image, self.depth_topic, self._on_depth, sensor_qos)
        if self.mask_topic:
            self.create_subscription(Image, self.mask_topic, self._on_mask, sensor_qos)
        if self.camera_info_topic:
            self.create_subscription(
                CameraInfo, self.camera_info_topic, self._on_camera_info, sensor_qos
            )
        self.create_subscription(TFMessage, self.tf_topic, self._on_tf, sensor_qos)
        self.create_subscription(
            TFMessage, self.tf_static_topic, self._on_tf, tf_static_qos
        )

        print("==> ROS2 online Boxer node ready")
        print(f"    camera: {self.camera}")
        print(f"    rgb:    {self.rgb_topic}")
        print(f"    depth:  {self.depth_topic or '<none>'}")
        print(f"    mask:   {self.mask_topic or '<none>'}")
        print(f"    info:   {self.camera_info_topic or '<none>'}")
        print(f"    frame:  {self.camera_frame}")
        print(f"    pose:   {self.pose_mode}, {self.camera_pose_convention}")
        print(f"    resize: {self.resize_hw if self.resize_hw > 0 else 'native'}")
        print(f"    max_fps:{self.max_fps if self.max_fps > 0 else 'unlimited'}")
        if self.fx is not None:
            print(
                "    intrinsics fallback: "
                f"fx={self.fx:.3f}, fy={self.fy:.3f}, "
                f"cx={self.cx:.3f}, cy={self.cy:.3f} ({self.intrinsics_source})"
            )

    def _load_topics(self) -> tuple[str, str, str, str]:
        info_path = os.path.join(self.data_dir, "recording_info.json")
        info = {}
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                info = json.load(f)
        camera_topics = info.get("camera_topics", {})

        selected = None
        for frame_name, topics in camera_topics.items():
            name_l = frame_name.lower()
            if self.camera == "head" and "head" in name_l:
                selected = (frame_name, topics)
                break
            if self.camera == "hand_left" and (
                "left_arm" in name_l or "hand_left" in name_l or "left" in name_l
            ):
                selected = (frame_name, topics)
                break
            if self.camera == "hand_right" and (
                "right_arm" in name_l or "hand_right" in name_l or "right" in name_l
            ):
                selected = (frame_name, topics)
                break

        if selected is not None:
            frame_name, topics = selected
            return (
                frame_name,
                topics.get("rgb") or self._default_rgb_topic(),
                topics.get("depth") or self._default_depth_topic(),
                topics.get("camera_info") or self._default_camera_info_topic(),
            )
        return (
            self._default_camera_frame(),
            self._default_rgb_topic(),
            self._default_depth_topic(),
            self._default_camera_info_topic(),
        )

    def _load_mask_topic(self) -> str:
        if self.args.mask_topic:
            return self.args.mask_topic
        info_path = os.path.join(self.data_dir, "recording_info.json")
        if os.path.exists(info_path):
            try:
                with open(info_path, "r") as f:
                    info = json.load(f)
                camera_topics = info.get("camera_topics", {})
                topics = camera_topics.get(self.camera_frame, {})
                for key in ("robot_mask", "mask", "segmentation"):
                    if topics.get(key):
                        return topics[key]
            except (OSError, json.JSONDecodeError):
                pass
        if self.camera == "head":
            return "/head_front_left_color_robot_mask"
        if self.camera == "hand_left":
            return "/left_arm_color_robot_mask"
        return "/right_arm_color_robot_mask"

    def _resolve_camera_pose_convention(self, value: str) -> str:
        value = str(value or "auto").lower()
        if value != "auto":
            return value
        frame_l = self.camera_frame.lower()
        if "optical" in frame_l or frame_l.endswith("_color") or "color" in frame_l:
            return "ros_optical"
        return "isaac_usd"

    def _default_camera_frame(self) -> str:
        if self.camera == "head":
            return "/galbot_one_golf/head_link2/head_front_left_color"
        if self.camera == "hand_left":
            return "/galbot_one_golf/left_arm_link7/left_arm_color"
        return "/galbot_one_golf/right_arm_link7/right_arm_color"

    def _default_rgb_topic(self) -> str:
        if self.camera == "head":
            return "/head_front_left_color_rgb"
        if self.camera == "hand_left":
            return "/left_arm_color_rgb"
        return "/right_arm_color_rgb"

    def _default_depth_topic(self) -> str:
        if self.camera == "head":
            return "/head_front_left_color_depth"
        if self.camera == "hand_left":
            return "/left_arm_color_depth"
        return "/right_arm_color_depth"

    def _default_camera_info_topic(self) -> str:
        if self.camera == "head":
            return "/head_front_left_color_camera_info"
        if self.camera == "hand_left":
            return "/left_arm_color_camera_info"
        return "/right_arm_color_camera_info"

    def _load_intrinsics_fallback(self) -> None:
        intrinsics, source = _load_json_intrinsics(
            self.data_dir,
            self.camera,
            self.rgb_topic,
            self.camera_info_topic,
            self.camera_frame,
        )
        if intrinsics is None and self.camera in DEFAULT_INTRINSICS:
            intrinsics = DEFAULT_INTRINSICS[self.camera]
            source = "built-in default"
        if intrinsics is None:
            return
        self.fx, self.fy, self.cx, self.cy = intrinsics
        self.intrinsics_source = f"JSON fallback {source}"

    def _update_intrinsics_from_camera_info(self, msg) -> bool:
        K = getattr(msg, "k", None)
        if K is None:
            K = getattr(msg, "K", [])
        K = list(np.asarray(K).reshape(-1))
        if len(K) >= 6 and K[0] > 0 and K[4] > 0:
            self.fx = float(K[0])
            self.fy = float(K[4])
            self.cx = float(K[2])
            self.cy = float(K[5])
            self.intrinsics_source = "bag CameraInfo"
            return True
        if not self._warned_zero_camera_info:
            print(
                "Galbot ROS online: CameraInfo intrinsics are empty/zero; "
                "keeping JSON fallback"
            )
            self._warned_zero_camera_info = True
        return False

    def _on_camera_info(self, msg) -> None:
        self._update_intrinsics_from_camera_info(msg)

    def _on_rgb(self, msg) -> None:
        try:
            ts = _stamp_to_ns(msg.header)
            img = decode_rgb_image(msg)
            with self._buf_lock:
                self._rgb_buf[ts] = img
                self._trim_buffer(self._rgb_buf)
                self.state.bump("rgb")
                self._try_emit_locked()
        except Exception as exc:
            self._record_callback_error("rgb", exc)

    def _on_depth(self, msg) -> None:
        try:
            depth = decode_depth_image(msg)
            if depth is None:
                self.state.bump("dropped_depth")
                return
            ts = _stamp_to_ns(msg.header)
            with self._buf_lock:
                self._depth_buf[ts] = depth
                self._trim_buffer(self._depth_buf)
                self.state.bump("depth")
                self._try_emit_locked()
        except Exception as exc:
            self._record_callback_error("depth", exc)

    def _on_mask(self, msg) -> None:
        try:
            mask = decode_mask_image(msg)
            if mask is None:
                return
            ts = _stamp_to_ns(msg.header)
            with self._buf_lock:
                self._mask_buf[ts] = mask
                self._trim_buffer(self._mask_buf)
                self.state.bump("mask")
                self._try_emit_locked()
        except Exception as exc:
            self._record_callback_error("mask", exc)

    def _on_tf(self, msg) -> None:
        try:
            for tf in getattr(msg, "transforms", []):
                parent = _frame_key(tf.header.frame_id)
                child = _frame_key(tf.child_frame_id)
                t = tf.transform.translation
                q = tf.transform.rotation
                T = np.eye(4, dtype=np.float32)
                T[:3, :3] = _quat_xyzw_to_R(q.x, q.y, q.z, q.w)
                T[:3, 3] = [float(t.x), float(t.y), float(t.z)]
                self._tf_by_child[child] = (parent, T)
        except Exception as exc:
            self._record_callback_error("tf", exc)

    def _record_callback_error(self, name: str, exc: Exception) -> None:
        msg = f"{name} callback error: {exc}"
        print(msg)
        self.state.add_error(msg)

    def _trim_buffer(self, buf: dict[int, Any]) -> None:
        while len(buf) > self.buffer_size:
            del buf[min(buf.keys())]

    def _try_emit_locked(self) -> None:
        if not self._rgb_buf or not self._depth_buf:
            return
        while self._rgb_buf and self._depth_buf:
            rgb_ts = min(self._rgb_buf.keys())
            depth_ts = _nearest_key(self._depth_buf, rgb_ts)
            if depth_ts is None:
                return
            depth_delta_ns = int(depth_ts - rgb_ts)
            if abs(depth_delta_ns) > self.sync_slop_ns:
                if depth_ts < rgb_ts:
                    del self._depth_buf[depth_ts]
                else:
                    del self._rgb_buf[rgb_ts]
                self.state.bump("dropped_sync")
                continue

            mask_ts = None
            mask = None
            if self._mask_buf and not self.disable_robot_mask:
                nearest_mask_ts = _nearest_key(self._mask_buf, rgb_ts)
                if (
                    nearest_mask_ts is not None
                    and abs(nearest_mask_ts - rgb_ts) <= self.sync_slop_ns
                ):
                    mask_ts = nearest_mask_ts
                    mask = self._mask_buf.pop(mask_ts)

            img_rgb = self._rgb_buf.pop(rgb_ts)
            depth_np = self._depth_buf.pop(depth_ts)
            self._emit_frame(
                img_rgb=img_rgb,
                depth_np=depth_np,
                mask_np=mask,
                time_ns=rgb_ts,
                depth_delta_ns=depth_delta_ns,
                mask_delta_ns=(None if mask_ts is None else int(mask_ts - rgb_ts)),
            )

    def _emit_frame(
        self,
        *,
        img_rgb: np.ndarray,
        depth_np: np.ndarray,
        mask_np: Optional[np.ndarray],
        time_ns: int,
        depth_delta_ns: int,
        mask_delta_ns: Optional[int],
    ) -> None:
        raw_idx = self._synced_seen
        self._synced_seen += 1
        self.state.bump("synced")

        if raw_idx < self.start_index:
            return
        if (raw_idx - self.start_index) % self.skip_n != 0:
            return
        if self.max_frames > 0 and self._emitted >= self.max_frames:
            return
        if self.max_fps > 0:
            now = time.perf_counter()
            min_dt = 1.0 / self.max_fps
            if now - self._last_emit_wall < min_dt:
                self.state.bump("dropped_throttle")
                return
            self._last_emit_wall = now

        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            self.state.bump("dropped_intrinsics")
            return

        robot_mask = None
        if mask_np is not None:
            robot_mask = (mask_np > self.mask_threshold).astype(np.uint8)

        if robot_mask is not None and robot_mask.shape[:2] == depth_np.shape[:2]:
            depth_np = depth_np.copy()
            depth_np[robot_mask > 0] = 0.0

        T_world_cam_tf = self._current_T_world_cam()
        tf_ready = T_world_cam_tf is not None
        if not tf_ready:
            self.state.bump("tf_missing")
        T_world_cam = self._effective_T_world_cam(T_world_cam_tf)

        datum = build_boxer_datum(
            img_rgb=img_rgb,
            depth_np=depth_np,
            fx=float(self.fx),
            fy=float(self.fy),
            cx=float(self.cx),
            cy=float(self.cy),
            time_ns=time_ns,
            T_world_cam=T_world_cam,
            resize_hw=self.resize_hw,
        )
        valid_depth = int(np.count_nonzero(depth_np > 0))
        packet = FramePacket(
            frame_idx=self._emitted,
            time_ns=int(time_ns),
            datum=datum,
            img_rgb=img_rgb,
            depth_valid=valid_depth,
            depth_total=int(depth_np.size),
            intrinsics_source=self.intrinsics_source,
            tf_ready=tf_ready,
            pose_mode=self.pose_mode,
            camera_pose_convention=self.camera_pose_convention,
            sync_depth_delta_ms=depth_delta_ns / 1e6,
            sync_mask_delta_ms=None if mask_delta_ns is None else mask_delta_ns / 1e6,
            robot_mask=robot_mask,
        )

        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
                self.state.bump("dropped_queue")
            except queue.Empty:
                pass
        try:
            self.frame_queue.put_nowait(packet)
            self.state.bump("queued")
            self.state.set_packet(packet)
            self._emitted += 1
            if self.args.dry_run and self._emitted % max(1, self.args.log_every) == 0:
                ratio = valid_depth / max(1, depth_np.size)
                print(
                    f"[dry-run] frame={packet.frame_idx} t={packet.time_ns} "
                    f"depth={ratio:.1%} tf={packet.tf_ready} "
                    f"info={packet.intrinsics_source}"
                )
        except queue.Full:
            self.state.bump("dropped_queue")

    def _current_T_world_cam(self) -> Optional[np.ndarray]:
        frame = self._resolve_tf_child_key(self.camera_frame)
        if frame is None:
            return None
        T_root_frame = np.eye(4, dtype=np.float32)
        seen = set()
        while frame in self._tf_by_child and frame not in seen:
            seen.add(frame)
            parent, T_parent_frame = self._tf_by_child[frame]
            T_root_frame = T_parent_frame @ T_root_frame
            frame = parent
        return T_root_frame

    def _resolve_tf_child_key(self, frame_id: str) -> Optional[str]:
        for candidate in _frame_key_candidates(frame_id):
            if candidate in self._tf_by_child:
                return candidate
        target_suffix = "/" + _frame_key(frame_id).split("/")[-1]
        for child in self._tf_by_child:
            if child.endswith(target_suffix):
                return child
        return None

    def _effective_T_world_cam(self, T_world_frame: Optional[np.ndarray]):
        if self.pose_mode == "identity":
            return np.eye(4, dtype=np.float32)
        if T_world_frame is None:
            return None

        T = T_world_frame.astype(np.float32).copy()
        if self.pose_mode == "tf_inverse":
            T = _invert_transform(T)

        if self.camera_pose_convention == "isaac_usd":
            T_frame_optical = np.eye(4, dtype=np.float32)
            T_frame_optical[:3, :3] = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
            T = T @ T_frame_optical
        return T


@dataclass
class ModelBundle:
    device: str
    owl: Any
    boxernet: Any
    text_labels: list[str]
    sem_name_to_id: dict[str, int]
    sem_id_to_name: dict[int, str]
    method: str


def select_device(force_cpu: bool) -> str:
    if torch.backends.mps.is_available() and not force_cpu:
        return "mps"
    if torch.cuda.is_available() and not force_cpu:
        return "cuda"
    return "cpu"


def load_models(args: argparse.Namespace) -> ModelBundle:
    from boxernet.boxernet import BoxerNet
    from owl.owl_wrapper import OwlWrapper

    device = select_device(args.force_cpu)
    print(f"==> Using device {device}")
    text_labels = load_text_labels(args.labels)
    print(f"==> Text prompts: {args.labels[0] if args.labels else 'custom'}")
    if len(text_labels) > 64:
        print(text_labels[:64])
        print(f"    ... and {len(text_labels) - 64} more")
    else:
        print(text_labels)

    owl = OwlWrapper(
        device,
        text_prompts=text_labels,
        min_confidence=args.thresh2d,
        precision=args.force_precision,
    )
    boxernet = BoxerNet.load_from_checkpoint(args.ckpt, device=device)
    sem_name_to_id = {label: i for i, label in enumerate(text_labels)}
    sem_id_to_name = {v: k for k, v in sem_name_to_id.items()}
    print(f"==> Boxer input resize: {boxernet.hw}x{boxernet.hw}")
    return ModelBundle(
        device=device,
        owl=owl,
        boxernet=boxernet,
        text_labels=text_labels,
        sem_name_to_id=sem_name_to_id,
        sem_id_to_name=sem_id_to_name,
        method="OWLv2",
    )


class InferenceWorker(threading.Thread):
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        frame_queue: "queue.Queue[FramePacket]",
        state: LiveState,
        models: ModelBundle,
        raw_writer: Optional[ObbCsvWriter2],
        fused_writer: Optional[ObbCsvWriter2],
        bb2d_csv_path: str,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.args = args
        self.frame_queue = frame_queue
        self.state = state
        self.models = models
        self.raw_writer = raw_writer
        self.fused_writer = fused_writer
        self.bb2d_csv_path = bb2d_csv_path
        self.stop_event = stop_event
        self.timer = CudaTimer(models.device)
        self.bb2d_written_once = False
        self.raw_history: list[ObbTW] = []
        self._latest_fused_obbs = _empty_obbs()
        self._latest_fused_snapshots: list[SimpleNamespace] = []
        self._last_fusion_frame = -1
        self._last_fusion_ms = 0.0
        self._semantic_embedding_cache: Optional[dict[str, torch.Tensor]] = None
        self._warned_semantic_fallback = False

        self.tracker = None
        if args.fusion_backend == "tracker":
            from utils.track_3d_boxes import BoundingBox3DTracker

            self.tracker = BoundingBox3DTracker(
                iou_threshold=args.tracker_iou_threshold,
                min_hits=args.tracker_min_hits,
                conf_threshold=args.thresh3d,
                samp_per_dim=args.tracker_samp_per_dim,
                max_missed=args.tracker_max_missed,
                force_cpu=args.force_cpu,
                merge_iou_threshold=args.tracker_merge_iou,
                merge_semantic_threshold=args.tracker_merge_sem,
                merge_iou_2d_threshold=args.tracker_merge_iou_2d,
                merge_interval=args.tracker_merge_interval,
                min_confidence_mass=args.tracker_min_conf_mass,
                min_obs_points=args.tracker_min_obs_points,
                verbose=args.verbose_tracker,
            )
            print("==> Fusion backend: tracker")
        else:
            history = (
                "all detections"
                if int(args.fusion_max_detections) <= 0
                else f"last {int(args.fusion_max_detections)} detections"
            )
            print(
                "==> Fusion backend: batch fuser "
                f"(view_fusion-style, semantic={args.fusion_semantics}, {history})"
            )

    def _append_history(self, raw_obbs: ObbTW) -> None:
        if len(raw_obbs) == 0:
            return
        self.raw_history.append(raw_obbs.clone().cpu())
        max_det = int(self.args.fusion_max_detections)
        if max_det <= 0:
            return
        total = sum(len(obbs) for obbs in self.raw_history)
        while self.raw_history and total > max_det:
            total -= len(self.raw_history.pop(0))

    def _history_as_obbs(self) -> ObbTW:
        chunks = [obbs._data for obbs in self.raw_history if len(obbs) > 0]
        if not chunks:
            return _empty_obbs()
        return ObbTW(torch.cat(chunks, dim=0)).float()

    @staticmethod
    def _label_onehot_embeddings(obbs: ObbTW) -> torch.Tensor:
        if len(obbs) == 0:
            return torch.empty(0, 0)
        sem_ids = obbs.sem_id.squeeze(-1).cpu().long()
        unique = sorted(set(int(x) for x in sem_ids.tolist()))
        id_to_col = {sid: idx for idx, sid in enumerate(unique)}
        emb = torch.zeros((len(obbs), max(1, len(unique))), dtype=torch.float32)
        for row, sid in enumerate(sem_ids.tolist()):
            emb[row, id_to_col[int(sid)]] = 1.0
        return emb

    def _owl_semantic_embeddings(self, obbs: ObbTW) -> torch.Tensor:
        if len(obbs) == 0:
            return torch.empty(0, 512)

        if self._semantic_embedding_cache is None:
            self._semantic_embedding_cache = {}
            owl = getattr(self.models, "owl", None)
            prompts = list(getattr(owl, "text_prompts", []) or [])
            embeddings = getattr(owl, "text_embeddings", None)
            if embeddings is not None and prompts:
                embeddings_cpu = embeddings.detach().cpu().float()
                embeddings_cpu = torch.nn.functional.normalize(
                    embeddings_cpu, dim=-1
                )
                for text, embedding in zip(prompts, embeddings_cpu):
                    self._semantic_embedding_cache[str(text)] = embedding

        texts = obbs.text_string()
        rows = []
        missing = []
        for text in texts:
            embedding = self._semantic_embedding_cache.get(text, None)
            if embedding is None:
                missing.append(text)
            else:
                rows.append(embedding)

        if missing:
            if not self._warned_semantic_fallback:
                uniq = sorted(set(missing))
                print(
                    "Semantic embedding cache miss for "
                    f"{len(uniq)} labels; falling back to view_fusion helper: "
                    + ", ".join(uniq[:8])
                )
                self._warned_semantic_fallback = True
            from utils.fuse_3d_boxes import precompute_semantic_embeddings

            return precompute_semantic_embeddings(obbs).cpu().float()

        return torch.stack(rows, dim=0).float()

    def _run_batch_fusion(
        self, frame_idx: int, force: bool = False
    ) -> tuple[ObbTW, list[SimpleNamespace]]:
        every = max(1, int(self.args.fusion_every))
        if (
            not force
            and self._last_fusion_frame >= 0
            and frame_idx % every != 0
        ):
            return self._latest_fused_obbs, self._latest_fused_snapshots

        detections = self._history_as_obbs()
        prob_threshold = float(self.args.fusion_prob_threshold)
        if prob_threshold > 0 and len(detections) > 0:
            keep = (detections.prob >= prob_threshold).reshape(-1)
            detections = detections[keep]
        if len(detections) == 0:
            self._latest_fused_obbs = _empty_obbs()
            self._latest_fused_snapshots = []
            self._last_fusion_frame = frame_idx
            return self._latest_fused_obbs, self._latest_fused_snapshots

        from utils.fuse_3d_boxes import BoundingBox3DFuser

        if self.args.fusion_semantics == "label":
            semantic_embeddings = self._label_onehot_embeddings(detections)
        elif self.args.fusion_semantics == "embedding":
            semantic_embeddings = self._owl_semantic_embeddings(detections)
        else:
            semantic_embeddings = None

        fuser = BoundingBox3DFuser(
            iou_threshold=self.args.fusion_iou_threshold,
            min_detections=self.args.fusion_min_detections,
            confidence_weighting=self.args.fusion_confidence_weighting,
            samp_per_dim=self.args.fusion_samp_per_dim,
            semantic_threshold=self.args.fusion_semantic_threshold,
            enable_nms=self.args.fusion_enable_nms,
            nms_iou_threshold=self.args.fusion_nms_iou_threshold,
            conf_threshold=0.0,
        )

        timer_name = "fuse"
        self.timer.start(timer_name)
        if self.args.verbose_fusion:
            instances = fuser.fuse(detections, semantic_embeddings=semantic_embeddings)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                instances = fuser.fuse(detections, semantic_embeddings=semantic_embeddings)
        self._last_fusion_ms = self.timer.stop(timer_name)

        snapshots = [
            SimpleNamespace(
                obb=inst.obb.clone().cpu(),
                track_id=idx,
                support_count=inst.support_count,
                missed_count=0,
                accumulated_weight=float(inst.obb.prob.item())
                * max(1, inst.support_count),
                cached_text=inst.obb.reshape(1, -1).text_string()[0]
                if hasattr(inst.obb, "text_string")
                else "?",
            )
            for idx, inst in enumerate(instances)
        ]
        if snapshots:
            fused_obbs = _stack_obbs_or_empty([snap.obb for snap in snapshots])
            fused_obbs.set_inst_id(
                torch.tensor([snap.track_id for snap in snapshots], dtype=torch.int32)
            )
        else:
            fused_obbs = _empty_obbs()
        self._latest_fused_obbs = fused_obbs
        self._latest_fused_snapshots = snapshots
        self._last_fusion_frame = frame_idx
        return fused_obbs, snapshots

    def run(self) -> None:
        print("==> Inference worker started")
        while not self.stop_event.is_set():
            try:
                packet = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                result = self._process(packet)
                self.state.set_result(result)
                if result.frame_idx % max(1, self.args.log_every) == 0:
                    print(
                        f"[online] frame={result.frame_idx} "
                        f"2d={result.num_2d}/{result.num_2d_raw} "
                        f"robotFilt={result.num_2d_filtered_robot} "
                        f"3d={result.num_3d} "
                        f"tracks={result.num_tracks} "
                        f"loadless={sum(result.timing_ms.values()):.0f}ms"
                    )
            except Exception:
                tb = traceback.format_exc()
                print(tb)
                self.state.add_error(tb.splitlines()[-1])

    def _process(self, packet: FramePacket) -> OnlineResult:
        datum = packet.datum
        img_torch = datum["img0"]
        HH, WW = img_torch.shape[2], img_torch.shape[3]
        rgb_image = (
            img_torch[0]
            .detach()
            .cpu()
            .permute(1, 2, 0)
            .numpy()
            .clip(0.0, 1.0)
        )
        rgb_image = (rgb_image * 255.0).astype(np.uint8)
        timings: dict[str, float] = {}

        if self.args.no_sdp:
            datum["sdp_w"] = torch.zeros(0, 3)

        self.timer.start("owl")
        img_torch_255 = img_torch.clone() * 255.0
        bb2d, scores2d, label_ints, _ = self.models.owl.forward(
            img_torch_255,
            False,
            resize_to_HW=(self.args.detector_hw, self.args.detector_hw),
        )
        labels2d = [self.models.text_labels[label_int] for label_int in label_ints]
        num_2d_raw = int(bb2d.shape[0])
        if num_2d_raw > 0:
            robot_keep = _robot_bbox_keep_mask(bb2d, packet.robot_mask, HH, WW)
            bb2d = bb2d[robot_keep]
            scores2d = scores2d[robot_keep]
            labels2d = [
                label for label, keep in zip(labels2d, robot_keep.tolist()) if keep
            ]
        num_2d_filtered_robot = num_2d_raw - int(bb2d.shape[0])
        if num_2d_filtered_robot > 0:
            self.state.bump("filtered_robot_bbox", num_2d_filtered_robot)

        if bb2d.shape[0] > 0:
            bb2d_xyxy = bb2d[:, [0, 2, 1, 3]].detach().cpu().numpy().astype(np.float32)
            bb2d_scores = scores2d.detach().cpu().numpy().astype(np.float32)
        else:
            bb2d_xyxy = np.zeros((0, 4), dtype=np.float32)
            bb2d_scores = np.zeros((0,), dtype=np.float32)
        timings["owl"] = self.timer.stop("owl")

        if bb2d.shape[0] == 0:
            raw_obbs = _empty_obbs()
            scores3d = torch.zeros(0)
            labels3d = []
        else:
            self.timer.start("boxer")
            datum["bb2d"] = bb2d
            if self.args.force_precision is not None:
                precision_dtype = (
                    torch.bfloat16
                    if self.args.force_precision == "bfloat16"
                    else torch.float32
                )
            elif self.models.device == "cuda" and torch.cuda.is_bf16_supported():
                precision_dtype = torch.bfloat16
            else:
                precision_dtype = torch.float32

            if self.models.device == "mps":
                outputs = self.models.boxernet.forward(datum)
            else:
                with torch.autocast(
                    device_type=self.models.device, dtype=precision_dtype
                ):
                    outputs = self.models.boxernet.forward(datum)
            obb_pr_w = outputs["obbs_pr_w"].cpu()[0]

            sem_ids = torch.zeros(len(labels2d), dtype=torch.int32)
            for i, label in enumerate(labels2d):
                if label not in self.models.sem_name_to_id:
                    new_id = len(self.models.sem_name_to_id)
                    self.models.sem_name_to_id[label] = new_id
                    self.models.sem_id_to_name[new_id] = label
                sem_ids[i] = self.models.sem_name_to_id[label]
            obb_pr_w.set_sem_id(sem_ids)

            scores3d_all = obb_pr_w.prob.squeeze(-1).clone()
            keepers = scores3d_all >= self.args.thresh3d
            raw_obbs = obb_pr_w[keepers].clone()
            scores3d = scores3d_all[keepers].clone()
            labels3d = [labels2d[i] for i in range(len(labels2d)) if keepers[i]]
            if len(raw_obbs) > 0:
                mean_scores = (scores2d[keepers].cpu() + scores3d) / 2.0
                raw_obbs.set_prob(mean_scores)
                text_data = torch.stack(
                    [
                        string2tensor(pad_string(label, max_len=128))
                        for label in labels3d
                    ]
                )
                raw_obbs.set_text(text_data)
            timings["boxer"] = self.timer.stop("boxer")

        if self.raw_writer is not None:
            self.raw_writer.write(
                raw_obbs, packet.time_ns, sem_id_to_name=self.models.sem_id_to_name
            )

        if bb2d.shape[0] > 0:
            save_bb2d_csv(
                self.bb2d_csv_path,
                frame_id=packet.frame_idx,
                bb2d=bb2d[:, [0, 2, 1, 3]],
                scores=scores2d,
                labels=labels2d,
                sem_name_to_id=self.models.sem_name_to_id,
                append=self.bb2d_written_once,
                time_ns=packet.time_ns,
                img_width=WW,
                img_height=HH,
                sensor="head",
                device="Galbot ROS2",
            )
            self.bb2d_written_once = True

        self._append_history(raw_obbs)
        if self.args.fusion_backend == "tracker":
            self.timer.start("track")
            active_tracks = self.tracker.update(
                raw_obbs,
                packet.frame_idx,
                cam=datum["cam0"].float(),
                T_world_rig=datum["T_world_rig0"].float(),
                observed_points=datum["sdp_w"].float(),
            )
            timings["track"] = self.timer.stop("track")

            if active_tracks:
                active_obbs = _stack_obbs_or_empty(
                    [track.obb for track in active_tracks]
                )
                ids = torch.tensor(
                    [track.track_id for track in active_tracks], dtype=torch.int32
                )
                active_obbs.set_inst_id(ids)
            else:
                active_obbs = _empty_obbs()

            active_snapshots = [
                SimpleNamespace(
                    obb=track.obb.clone().cpu(),
                    track_id=track.track_id,
                    support_count=track.support_count,
                    missed_count=track.missed_count,
                    accumulated_weight=track.accumulated_weight,
                    cached_text=track.cached_text,
                )
                for track in active_tracks
            ]
        else:
            active_obbs, active_snapshots = self._run_batch_fusion(packet.frame_idx)
            if self._last_fusion_frame == packet.frame_idx:
                timings["fuse"] = self._last_fusion_ms

        if self.fused_writer is not None and len(active_obbs) > 0:
            self.fused_writer.write(
                active_obbs,
                packet.time_ns,
                sem_id_to_name=self.models.sem_id_to_name,
            )

        cam_position, cam_rpy_deg = _pose_data_to_position_rpy_deg(
            datum["T_world_rig0"]
        )

        return OnlineResult(
            frame_idx=packet.frame_idx,
            time_ns=packet.time_ns,
            raw_obbs=raw_obbs.clone().cpu(),
            active_obbs=active_obbs.clone().cpu(),
            active_tracks=active_snapshots,
            rgb_image=rgb_image,
            bb2d_xyxy=bb2d_xyxy,
            bb2d_labels=list(labels2d),
            bb2d_scores=bb2d_scores,
            camera_position=cam_position,
            camera_rpy_deg=cam_rpy_deg,
            tf_ready=packet.tf_ready,
            pose_mode=packet.pose_mode,
            camera_pose_convention=packet.camera_pose_convention,
            num_2d=int(bb2d.shape[0]),
            num_2d_raw=num_2d_raw,
            num_2d_filtered_robot=num_2d_filtered_robot,
            num_3d=int(len(raw_obbs)),
            num_tracks=int(len(active_snapshots)),
            labels=labels3d,
            scores=[float(x) for x in scores3d.reshape(-1).tolist()],
            timing_ms=timings,
        )


def launch_live_viewer(
    *,
    state: LiveState,
    log_dir: str,
    window_w: int,
    window_h: int,
    show_rgb: bool,
    rgb_panel_frac: float,
    verbose: bool,
) -> None:
    import moderngl
    from utils.viewer_3d import OBBViewer, launch_viewer, scale_factor
    import utils.imgui_compat as imgui

    default_w, default_h = 1400 * scale_factor, 900 * scale_factor
    init_w = window_w if window_w > 0 else default_w
    init_h = window_h if window_h > 0 else default_h

    class LiveGalbotFusionViewer(OBBViewer):
        title = "Galbot Boxer Online Fusion"
        window_size = (init_w, init_h)

        def __init__(self, **kw):
            self._state = state
            self._last_result_version = -1
            self._focused_once = False
            self._live_stats = {}
            self._live_errors = []
            self._last_result: Optional[OnlineResult] = None
            self._rgb_texture = None
            self._rgb_tex_size = (0, 0)
            self._bb2d_boxes = np.zeros((0, 4), dtype=np.float32)
            self._bb2d_labels: list[str] = []
            self._bb2d_scores = np.zeros((0,), dtype=np.float32)
            self.show_rgb = bool(show_rgb)
            self.show_rgb_2dbb = True
            self.show_rgb_2dbb_labels = True
            self.rgb_panel_max_frac = float(rgb_panel_frac)
            self.rgb_bb2_thickness = 3.0
            self.rgb_text_scale = 1.0
            super().__init__(
                all_obbs=_empty_obbs(),
                timed_obbs={},
                root_path=log_dir,
                skip_precompute=True,
                seq_name="galbot_ros_online",
                **kw,
            )
            self.show_raw_set = True
            self.show_tracked_all_set = True
            self.show_tracked_visible_set = False
            self.show_text_labels = True
            self.prob_threshold = 0.0
            self.alpha = 0.35
            self.raw_line_width = 2
            self.tracked_all_line_width = 5

        def _get_3d_viewport_size(self) -> tuple[int, int]:
            win_w, win_h = self.wnd.size
            return max(1, win_w - self.ui_panel_width), win_h

        def _upload_rgb_texture(self, img_rgb: np.ndarray) -> None:
            if img_rgb is None:
                return
            img = np.ascontiguousarray(img_rgb)
            h, w = img.shape[:2]
            if h <= 0 or w <= 0:
                return
            if self._rgb_texture is None or self._rgb_tex_size != (w, h):
                if self._rgb_texture is not None:
                    self.imgui.remove_texture(self._rgb_texture)
                    self._rgb_texture.release()
                self._rgb_texture = self.ctx.texture((w, h), 3, img.tobytes())
                self._rgb_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
                self.imgui.register_texture(self._rgb_texture)
                self._rgb_tex_size = (w, h)
            else:
                self._rgb_texture.write(img.tobytes())

        def _draw_2dbb_overlay(self, draw_list, img_min, draw_w: float, draw_h: float):
            tex_w, tex_h = self._rgb_tex_size
            if tex_w <= 0 or tex_h <= 0 or len(self._bb2d_boxes) == 0:
                return
            sx = draw_w / tex_w
            sy = draw_h / tex_h
            text_col = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0)
            label_scale = max(0.5, float(self.rgb_text_scale))
            imgui.set_window_font_scale(label_scale)
            for idx, box in enumerate(self._bb2d_boxes):
                x1, y1, x2, y2 = [float(v) for v in box]
                label = self._bb2d_labels[idx] if idx < len(self._bb2d_labels) else "?"
                score = (
                    float(self._bb2d_scores[idx])
                    if idx < len(self._bb2d_scores)
                    else 0.0
                )
                r, g, b = self._label_to_theme_random_color(label.strip().lower())
                col = imgui.get_color_u32_rgba(float(r), float(g), float(b), 1.0)
                bg_col = imgui.get_color_u32_rgba(
                    float(r) * 0.35, float(g) * 0.35, float(b) * 0.35, 0.75
                )
                rx0 = img_min.x + x1 * sx
                ry0 = img_min.y + y1 * sy
                rx1 = img_min.x + x2 * sx
                ry1 = img_min.y + y2 * sy
                draw_list.add_rect(
                    rx0, ry0, rx1, ry1, col, 0.0, 0, self.rgb_bb2_thickness
                )
                if self.show_rgb_2dbb_labels:
                    text = f"{label[:18]} {score:.2f}"
                    tw, th = imgui.calc_text_size(text)
                    ty0 = max(img_min.y, ry0 - th - 3)
                    draw_list.add_rect_filled(
                        rx0 - 1, ty0 - 1, rx0 + tw + 2, ty0 + th + 1, bg_col
                    )
                    draw_list.add_text(rx0, ty0, text_col, text)
            imgui.set_window_font_scale(1.0)

        def _clear_raw_geometry(self):
            if self.cached_instance_vbo is not None:
                self.cached_instance_vbo.release()
            if self.cached_instance_vao is not None:
                self.cached_instance_vao.release()
            self.cached_instance_vbo = None
            self.cached_instance_vao = None
            self.cached_instance_count = 0

        def _clear_tracked_geometry(self):
            if self.tracked_all_instance_vbo is not None:
                self.tracked_all_instance_vbo.release()
            if self.tracked_all_instance_vao is not None:
                self.tracked_all_instance_vao.release()
            self.tracked_all_instance_vbo = None
            self.tracked_all_instance_vao = None
            self.tracked_all_instance_count = 0
            self.tracked_all_text_labels = []
            self.tracked_all_label_positions = []
            self.tracked_all_label_colors = []

        def _pull_live_state(self):
            stats, version, result, errors = self._state.snapshot()
            self._live_stats = stats
            self._live_errors = errors
            if result is None or version == self._last_result_version:
                return
            self._last_result_version = version
            self._last_result = result
            self._upload_rgb_texture(result.rgb_image)
            self._bb2d_boxes = np.asarray(result.bb2d_xyxy, dtype=np.float32)
            self._bb2d_labels = list(result.bb2d_labels)
            self._bb2d_scores = np.asarray(result.bb2d_scores, dtype=np.float32)

            self.all_obbs = result.raw_obbs
            self.total_detections = len(result.raw_obbs)
            self._cached_filtered_obbs = None
            self._cached_prob_threshold = None
            self._clear_raw_geometry()
            self._clear_tracked_geometry()

            with (
                contextlib.redirect_stdout(io.StringIO())
                if not verbose
                else contextlib.nullcontext()
            ):
                if len(result.raw_obbs) > 0:
                    self._build_geometry_cache()
                self.tracked_all_instances = [
                    SimpleNamespace(obb=obb, support_count=meta.support_count)
                    for obb, meta in zip(result.active_obbs, result.active_tracks)
                ]
                if self.tracked_all_instances:
                    self._build_tracked_all_geometry()

            if (
                not self._focused_once
                and (len(result.active_obbs) > 0 or len(result.raw_obbs) > 0)
            ):
                self._focus_on_scene()
                self._focused_once = True

        def render_3d(self, time_s: float, frame_time: float) -> None:
            self._pull_live_state()
            super().render_3d(time_s, frame_time)

        def _render_rgb_inline(self) -> None:
            result = self._last_result
            if result is not None:
                pos = result.camera_position
                rpy = result.camera_rpy_deg
                imgui.text(f"Frame {result.frame_idx}  t={result.time_ns / 1e9:.3f}s")
                imgui.text(
                    "Cam xyz: "
                    f"{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f} m"
                )
                imgui.text(
                    "Cam rpy: "
                    f"{rpy[0]:+.1f}, {rpy[1]:+.1f}, {rpy[2]:+.1f} deg"
                )
                imgui.text(f"Pose: {result.pose_mode}, {result.camera_pose_convention}")
                if not result.tf_ready:
                    imgui.text_colored("TF missing: fallback pose", 1.0, 0.25, 0.25)

            if not self.show_rgb or self._rgb_texture is None:
                return
            tex_w, tex_h = self._rgb_tex_size
            avail_w, _avail_h = imgui.get_content_region_available()
            _, win_h = self.wnd.size
            max_h = win_h * max(0.12, min(0.75, self.rgb_panel_max_frac))
            if tex_w <= 0 or tex_h <= 0 or avail_w <= 0 or max_h <= 0:
                return
            img_scale = min(avail_w / tex_w, max_h / tex_h)
            draw_w = tex_w * img_scale
            draw_h = tex_h * img_scale
            imgui.image(self._rgb_texture.glo, draw_w, draw_h)
            img_min = imgui.get_item_rect_min()
            if self.show_rgb_2dbb:
                self._draw_2dbb_overlay(
                    imgui.get_window_draw_list(), img_min, draw_w, draw_h
                )

        def _render_main_controls(self):
            self._section_header("Online")
            result = self._last_result
            if result is None:
                imgui.text("Waiting for detections...")
            else:
                imgui.text(f"Frame: {result.frame_idx}")
                imgui.text(
                    f"2DBB: {result.num_2d}/{result.num_2d_raw} "
                    f"(robot filtered {result.num_2d_filtered_robot})"
                )
                imgui.text(f"Raw 3DBB: {result.num_3d}")
                imgui.text(f"Fused Boxes: {result.num_tracks}")
                imgui.text(
                    "Timing: "
                    + " ".join(
                        f"{key}:{val:.0f}ms"
                        for key, val in result.timing_ms.items()
                    )
                )
            stats = self._live_stats
            imgui.text(
                "ROS: "
                f"rgb={stats.get('rgb', 0)} "
                f"depth={stats.get('depth', 0)} "
                f"synced={stats.get('synced', 0)} "
                f"queued={stats.get('queued', 0)} "
                f"dropQ={stats.get('dropped_queue', 0)} "
                f"throttle={stats.get('dropped_throttle', 0)} "
                f"arm2d={stats.get('filtered_robot_bbox', 0)}"
            )
            if self._live_errors:
                imgui.text_colored("Last error:", 1.0, 0.25, 0.25)
                imgui.text_wrapped(self._live_errors[-1][:260])

            self._section_header("RGB")
            _changed, self.show_rgb = imgui.checkbox("Show RGB", self.show_rgb)
            _changed, self.show_rgb_2dbb = imgui.checkbox(
                "Show 2DBBs", self.show_rgb_2dbb
            )
            _changed, self.show_rgb_2dbb_labels = imgui.checkbox(
                "Show 2D Labels", self.show_rgb_2dbb_labels
            )
            imgui.push_item_width(200)
            _changed, self.rgb_panel_max_frac = imgui.slider_float(
                "RGB Max Height", self.rgb_panel_max_frac, 0.15, 0.65
            )
            _changed, self.rgb_bb2_thickness = imgui.slider_float(
                "2DBB Line Width", self.rgb_bb2_thickness, 1.0, 8.0
            )
            imgui.pop_item_width()
            self._render_rgb_inline()

            self._section_header("Visualization")
            self._render_common_visual_controls(
                raw_checkbox_label="Current Frame 3DBBs",
                tracked_all_checkbox_label="Fused Boxes",
                tracked_all_line_label="Fused Line Width",
                show_visible_checkbox=False,
                show_sets_header=False,
            )

            self._section_header("Camera")
            if imgui.button("Focus on Latest"):
                self._focus_on_scene()
            imgui.same_line()
            if imgui.button("Screenshot"):
                self._save_screenshot()

    launch_viewer(LiveGalbotFusionViewer)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Online Boxer runner for Galbot ROS2 bag playback"
    )
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--camera",
        type=str,
        default="head",
        choices=["head", "rgb", "hand_left", "hand_right"],
    )
    parser.add_argument("--mask_topic", type=str, default="")
    parser.add_argument("--output_dir", type=str, default=EVAL_PATH)
    parser.add_argument("--write_name", type=str, default="boxer_online")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--visualize",
        dest="visualize",
        action="store_true",
        default=True,
        help="enable live viewer (default)",
    )
    parser.add_argument(
        "--no_visualize",
        dest="visualize",
        action="store_false",
        help="disable live viewer and print status in the terminal",
    )
    parser.add_argument("--no_csv", action="store_true")
    parser.add_argument("--skip_n", type=int, default=1)
    parser.add_argument("--start_n", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument(
        "--max_fps",
        type=float,
        default=10.0,
        help="maximum synchronized frames sent to Boxer per second (0 = unlimited)",
    )
    parser.add_argument("--sync_slop_ms", type=float, default=35.0)
    parser.add_argument("--buffer_size", type=int, default=32)
    parser.add_argument("--queue_size", type=int, default=2)
    parser.add_argument(
        "--pose_mode",
        type=str,
        default="tf",
        choices=["tf", "tf_inverse", "identity"],
        help="how to interpret the TF chain before camera-axis conversion",
    )
    parser.add_argument(
        "--camera_pose_convention",
        type=str,
        default="auto",
        choices=["auto", "ros_optical", "isaac_usd"],
        help="camera frame convention; auto uses isaac_usd unless frame name contains optical",
    )
    parser.add_argument("--disable_robot_mask", action="store_true")
    parser.add_argument("--mask_threshold", type=float, default=0.0)
    parser.add_argument("--resize_hw", type=int, default=0)
    parser.add_argument("--labels", type=comma_separated_list, default=["lvisplus"])
    parser.add_argument("--detector_hw", type=int, default=960)
    parser.add_argument("--thresh2d", type=float, default=0.25)
    parser.add_argument("--thresh3d", type=float, default=0.5)
    parser.add_argument(
        "--ckpt",
        type=str,
        default=os.path.join(CKPT_PATH, DEFAULT_BOXERNET_CKPT),
    )
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument(
        "--force_precision", type=str, default=None, choices=["float32", "bfloat16"]
    )
    parser.add_argument("--no_sdp", action="store_true")
    parser.add_argument(
        "--fusion_backend",
        type=str,
        default="batch",
        choices=["batch", "tracker"],
        help="batch matches view_fusion-style connected-component fusion; tracker is incremental association",
    )
    parser.add_argument(
        "--fusion_every",
        type=int,
        default=1,
        help="run batch fusion every N processed frames",
    )
    parser.add_argument(
        "--fusion_max_detections",
        type=int,
        default=0,
        help="history cap for batch fusion; <=0 keeps all detections like view_fusion.py",
    )
    parser.add_argument("--fusion_iou_threshold", type=float, default=0.3)
    parser.add_argument(
        "--fusion_prob_threshold",
        type=float,
        default=0.55,
        help="3DBB confidence threshold before fusion, matching view_fusion.py default",
    )
    parser.add_argument("--fusion_min_detections", type=int, default=4)
    parser.add_argument("--fusion_samp_per_dim", type=int, default=8)
    parser.add_argument(
        "--fusion_confidence_weighting",
        type=str,
        default="robust",
        choices=["uniform", "linear", "quadratic", "robust"],
    )
    parser.add_argument("--fusion_semantic_threshold", type=float, default=0.7)
    parser.add_argument(
        "--fusion_semantics",
        type=str,
        default="embedding",
        choices=["label", "embedding", "none"],
        help="embedding uses OWL text embeddings like view_fusion.py; label is a cheap same-label gate",
    )
    parser.add_argument("--fusion_disable_nms", action="store_true")
    parser.add_argument("--fusion_nms_iou_threshold", type=float, default=0.6)
    parser.add_argument("--verbose_fusion", action="store_true")
    parser.add_argument("--tracker_iou_threshold", type=float, default=0.25)
    parser.add_argument("--tracker_min_hits", type=int, default=8)
    parser.add_argument("--tracker_samp_per_dim", type=int, default=8)
    parser.add_argument("--tracker_max_missed", type=int, default=90)
    parser.add_argument("--tracker_merge_iou", type=float, default=0.5)
    parser.add_argument("--tracker_merge_sem", type=float, default=0.7)
    parser.add_argument("--tracker_merge_iou_2d", type=float, default=0.7)
    parser.add_argument("--tracker_merge_interval", type=int, default=5)
    parser.add_argument("--tracker_min_conf_mass", type=float, default=4.0)
    parser.add_argument("--tracker_min_obs_points", type=int, default=4)
    parser.add_argument("--verbose_tracker", action="store_true")
    parser.add_argument("--window_w", type=int, default=0)
    parser.add_argument("--window_h", type=int, default=0)
    parser.add_argument("--hide_rgb", action="store_true")
    parser.add_argument("--rgb_panel_frac", type=float, default=0.32)
    parser.add_argument("--log_every", type=int, default=30)
    parser.add_argument("--verbose_viewer", action="store_true")
    return parser


def require_ros2() -> None:
    if rclpy is not None:
        return
    raise SystemExit(
        "Cannot import rclpy. Source ROS2 before running this script, for example:\n"
        "  source /opt/ros/humble/setup.bash\n"
        "  conda activate jarvis\n"
        "  python run_galbot_ros_online.py --dry_run"
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.camera = normalize_galbot_camera(args.camera)
    args.data_dir = os.path.expanduser(args.data_dir)
    args.output_dir = os.path.expanduser(args.output_dir)
    args.ckpt = os.path.expanduser(args.ckpt)
    args.fusion_enable_nms = not args.fusion_disable_nms
    os.environ.setdefault("BOXER_DISABLE_COMPILE", "1")

    require_ros2()

    seq_name = Path(args.data_dir.rstrip("/")).name or "galbot_online"
    log_dir = os.path.join(args.output_dir, seq_name)
    os.makedirs(log_dir, exist_ok=True)
    ros_log_dir = os.path.join(log_dir, "ros_logs")
    os.makedirs(ros_log_dir, exist_ok=True)
    os.environ.setdefault("ROS_LOG_DIR", ros_log_dir)
    raw_csv_path = os.path.join(log_dir, f"{args.write_name}_3dbbs.csv")
    fused_csv_path = os.path.join(log_dir, f"{args.write_name}_3dbbs_fused.csv")
    bb2d_csv_path = os.path.join(log_dir, f"{args.write_name}_2dbbs.csv")

    models = None
    resize_hw = int(args.resize_hw)
    if args.dry_run:
        print("==> Dry-run mode: ROS sync only, no OWL/BoxerNet model loading")
    else:
        models = load_models(args)
        if resize_hw <= 0:
            resize_hw = int(models.boxernet.hw)

    frame_queue: "queue.Queue[FramePacket]" = queue.Queue(
        maxsize=max(1, int(args.queue_size))
    )
    state = LiveState()
    stop_event = threading.Event()

    raw_writer = None
    fused_writer = None
    if not args.no_csv and not args.dry_run:
        raw_writer = ObbCsvWriter2(raw_csv_path)
        fused_writer = ObbCsvWriter2(fused_csv_path)
        print(f"==> Raw CSV:   {raw_csv_path}")
        print(f"==> Fused CSV: {fused_csv_path}")
        print(f"==> 2D CSV:    {bb2d_csv_path}")

    rclpy.init()
    node = GalbotRosOnlineNode(
        args=args, frame_queue=frame_queue, state=state, resize_hw=resize_hw
    )

    def _spin_node():
        try:
            rclpy.spin(node)
        except Exception as exc:
            if exc.__class__.__name__ != "ExternalShutdownException":
                print(f"[ros-spin] {exc}")
                state.add_error(f"ros spin: {exc}")

    spin_thread = threading.Thread(target=_spin_node, daemon=True)
    spin_thread.start()

    worker = None
    if models is not None:
        worker = InferenceWorker(
            args=args,
            frame_queue=frame_queue,
            state=state,
            models=models,
            raw_writer=raw_writer,
            fused_writer=fused_writer,
            bb2d_csv_path=bb2d_csv_path,
            stop_event=stop_event,
        )
        worker.start()

    print("\n==> Now play the bag in another terminal, for example:")
    print(f"    ros2 bag play {args.data_dir} --clock\n")

    try:
        if args.visualize:
            launch_live_viewer(
                state=state,
                log_dir=log_dir,
                window_w=args.window_w,
                window_h=args.window_h,
                show_rgb=not args.hide_rgb,
                rgb_panel_frac=args.rgb_panel_frac,
                verbose=args.verbose_viewer,
            )
        else:
            last_print = 0.0
            while rclpy.ok() and not stop_event.is_set():
                time.sleep(0.2)
                now = time.perf_counter()
                if now - last_print < 5.0:
                    continue
                last_print = now
                stats, _version, result, errors = state.snapshot()
                msg = (
                    "[status] "
                    f"rgb={stats.get('rgb', 0)} "
                    f"depth={stats.get('depth', 0)} "
                    f"synced={stats.get('synced', 0)} "
                    f"queued={stats.get('queued', 0)} "
                    f"processed={stats.get('processed', 0)} "
                    f"dropQ={stats.get('dropped_queue', 0)} "
                    f"throttle={stats.get('dropped_throttle', 0)} "
                    f"arm2d={stats.get('filtered_robot_bbox', 0)}"
                )
                if result is not None:
                    msg += (
                        f" latest_frame={result.frame_idx} "
                        f"2d={result.num_2d}/{result.num_2d_raw} "
                        f"tracks={result.num_tracks}"
                    )
                print(msg)
                if errors:
                    print(f"[last-error] {errors[-1]}")
    except KeyboardInterrupt:
        print("\n==> Interrupted")
    finally:
        stop_event.set()
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=2.0)
        if worker is not None:
            worker.join(timeout=5.0)
        if raw_writer is not None:
            raw_writer.close()
            print(f"==> Saved raw CSV to {raw_csv_path}")
        if fused_writer is not None:
            fused_writer.close()
            print(f"==> Saved fused CSV to {fused_csv_path}")


if __name__ == "__main__":
    main()
