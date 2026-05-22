# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Loader for Galbot/Isaac recorded clips.

The Genie Sim recorder stores synchronized camera streams as mp4 files under
``observations/videos`` plus camera calibration JSON under ``parameters/sensor``.
This loader exposes those clips through the standard Boxer datum contract.
"""

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from loaders.base_loader import BaseLoader
from utils.tw.obb import ObbTW
from utils.tw.pose import PoseTW


CAMERA_ALIASES = {
    "rgb": "head",
    "head": "head",
    "head_color": "head",
    "hand_left": "hand_left",
    "left": "hand_left",
    "hand_left_color": "hand_left",
    "hand_right": "hand_right",
    "right": "hand_right",
    "hand_right_color": "hand_right",
}

VIDEO_NAMES = {
    "head": "head_color.mp4",
    "hand_left": "hand_left_color.mp4",
    "hand_right": "hand_right_color.mp4",
}

INTRINSIC_NAMES = {
    "head": "intrinsic_head_front_rgb.json",
    "hand_left": "intrinsic_hand_left_rgb.json",
    "hand_right": "intrinsic_hand_right_rgb.json",
}

DEFAULT_INTRINSICS = {
    "head": (409.0, 409.0, 613.8798217773438, 486.08056640625),
    "hand_left": (323.20001220703125, 327.8999938964844, 318.0, 232.0),
    "hand_right": (323.20001220703125, 327.8999938964844, 318.0, 232.0),
}


def normalize_galbot_camera(camera: str) -> str:
    key = (camera or "head").strip().lower()
    if key not in CAMERA_ALIASES:
        valid = ", ".join(sorted(CAMERA_ALIASES))
        raise ValueError(f"Unknown Galbot camera '{camera}'. Valid names: {valid}")
    return CAMERA_ALIASES[key]


def is_galbot_sequence(path: str) -> bool:
    """Return True when ``path`` looks like a Genie Sim/Galbot recording."""
    root = os.path.expanduser(path)
    if os.path.isfile(root):
        if not root.lower().endswith((".mp4", ".webm", ".avi", ".mov")):
            return root.lower().endswith(".db3")
        else:
            video_dir = os.path.dirname(root)
            if os.path.basename(video_dir) == "videos":
                root = os.path.dirname(os.path.dirname(video_dir))
            else:
                return False
    if not os.path.isdir(root):
        return False
    video_dir = os.path.join(root, "observations", "videos")
    sensor_dir = os.path.join(root, "parameters", "sensor")
    if os.path.isdir(video_dir) and os.path.isdir(sensor_dir):
        return True
    info_path = os.path.join(root, "recording_info.json")
    has_db3 = any(name.endswith(".db3") for name in os.listdir(root))
    return os.path.exists(info_path) and has_db3


class GalbotLoader(BaseLoader):
    """Stream frames from a Galbot/Isaac recording directory or video file."""

    def __init__(
        self,
        seq_dir: str,
        camera: str = "head",
        skip_frames: int = 1,
        max_frames: Optional[int] = None,
        start_frame: int = 1,
    ):
        input_path = os.path.expanduser(seq_dir)
        self._video_override = None
        self._db3_override = None
        if os.path.isfile(input_path):
            if input_path.lower().endswith(".db3"):
                self._db3_override = input_path
                input_path = os.path.dirname(input_path)
            else:
                self._video_override = input_path
                video_dir = os.path.dirname(input_path)
                if os.path.basename(video_dir) == "videos":
                    input_path = os.path.dirname(os.path.dirname(video_dir))
        self.seq_dir = input_path
        self.camera = normalize_galbot_camera(camera)
        self.device_name = "Galbot Isaac"
        self.resize = None

        self.mode = "rosbag" if self._looks_like_rosbag_dir() else "video"
        self._reader = None
        self._message_iter = None
        self._connections = {}
        self._camera_info = None
        self._latest_depth = None
        self._tf_by_child = {}
        self._seen_rgb = 0

        if self.mode == "video":
            self.video_path = self._resolve_video_path()
            self.source_width, self.source_height, self.fps, total_frames = (
                self._probe_video()
            )
            self.fx, self.fy, self.cx, self.cy = self._load_intrinsics()
        else:
            self.video_path = ""
            self.db3_path = self._resolve_db3_path()
            self._load_rosbag_config()
            self.fps = self._recording_fps
            self.source_width = 0
            self.source_height = 0
            total_frames = self._count_topic_messages(self.rgb_topic)
            self.fx = self.fy = self.cx = self.cy = None
            self._intrinsics_source = "uninitialized"
            self._warned_zero_camera_info = False
        self.time_origin_ns = self._load_time_origin_ns()

        first = max(0, int(start_frame) - 1)
        step = max(1, int(skip_frames))
        frame_indices = list(range(first, total_frames, step))
        if max_frames is not None:
            frame_indices = frame_indices[: int(max_frames)]
        self.frame_indices = frame_indices
        self.length = len(self.frame_indices)
        self.index = 0

        self._cap = None
        self._cap_lock = threading.Lock()

        size_text = (
            f", size={self.source_width}x{self.source_height}"
            if self.mode == "video"
            else ""
        )
        print(
            "GalbotLoader: "
            f"{os.path.basename(self.seq_dir.rstrip('/')) or self.seq_dir}, "
            f"mode={self.mode}, camera={self.camera}, "
            f"frames={self.length}/{total_frames}, fps={self.fps:.2f}{size_text}"
        )

        self._init_prefetch()

    def _looks_like_rosbag_dir(self) -> bool:
        if self._db3_override is not None:
            return True
        if not os.path.isdir(self.seq_dir):
            return False
        return any(name.endswith(".db3") for name in os.listdir(self.seq_dir))

    def _resolve_db3_path(self) -> str:
        if self._db3_override is not None:
            return self._db3_override
        db3s = sorted(
            os.path.join(self.seq_dir, name)
            for name in os.listdir(self.seq_dir)
            if name.endswith(".db3")
        )
        if not db3s:
            raise FileNotFoundError(f"No .db3 file found under {self.seq_dir}")
        return db3s[0]

    def _load_rosbag_config(self):
        info_path = os.path.join(self.seq_dir, "recording_info.json")
        info = {}
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                info = json.load(f)

        self._recording_fps = float(info.get("fps") or 30.0)
        camera_topics = info.get("camera_topics", {})

        selected = None
        for frame_name, topics in camera_topics.items():
            name_l = frame_name.lower()
            if self.camera == "head" and "head" in name_l:
                selected = (frame_name, topics)
                break
            if self.camera == "hand_left" and (
                "left_arm" in name_l or "hand_left" in name_l
            ):
                selected = (frame_name, topics)
                break
            if self.camera == "hand_right" and (
                "right_arm" in name_l or "hand_right" in name_l
            ):
                selected = (frame_name, topics)
                break

        if selected is not None:
            self.camera_frame, topics = selected
            self.rgb_topic = topics.get("rgb") or self._default_rgb_topic()
            self.depth_topic = topics.get("depth") or ""
            self.camera_info_topic = topics.get("camera_info") or ""
        else:
            self.camera_frame = self._default_camera_frame()
            self.rgb_topic = self._default_rgb_topic()
            self.depth_topic = self._default_depth_topic()
            self.camera_info_topic = self._default_camera_info_topic()

        self.tf_topic = "/tf"

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
        return ""

    def _default_camera_info_topic(self) -> str:
        if self.camera == "head":
            return "/head_front_left_color_camera_info"
        if self.camera == "hand_left":
            return "/left_arm_color_camera_info"
        return "/right_arm_color_camera_info"

    def _count_topic_messages(self, topic_name: str) -> int:
        if not topic_name:
            return 0
        with sqlite3.connect(self.db3_path) as con:
            row = con.execute(
                "SELECT id FROM topics WHERE name = ?", (topic_name,)
            ).fetchone()
            if row is None:
                return 0
            count = con.execute(
                "SELECT COUNT(*) FROM messages WHERE topic_id = ?", (row[0],)
            ).fetchone()[0]
        return int(count)

    def _resolve_video_path(self) -> str:
        if self._video_override is not None:
            return self._video_override
        if os.path.isfile(self.seq_dir):
            return self.seq_dir

        video_dir = os.path.join(self.seq_dir, "observations", "videos")
        candidates = [
            os.path.join(video_dir, VIDEO_NAMES[self.camera]),
            os.path.join(video_dir, f"{self.camera}.mp4"),
            os.path.join(video_dir, f"{self.camera}.webm"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"No video found for camera '{self.camera}' under {video_dir}"
        )

    def _probe_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if total_frames <= 0:
            raise IOError(f"Video has no readable frames: {self.video_path}")
        if fps <= 0:
            fps = 30.0
        return width, height, fps, total_frames

    @staticmethod
    def _intrinsics_from_data(data):
        if data is None:
            return None
        for key in ("K", "k", "K_flat"):
            if key in data:
                K = list(np.asarray(data[key], dtype=np.float32).reshape(-1))
                if len(K) >= 6 and K[0] > 0 and K[4] > 0:
                    return float(K[0]), float(K[4]), float(K[2]), float(K[5])
        camera_matrix = data.get("camera_matrix")
        if isinstance(camera_matrix, dict) and "data" in camera_matrix:
            K = list(np.asarray(camera_matrix["data"], dtype=np.float32).reshape(-1))
            if len(K) >= 6 and K[0] > 0 and K[4] > 0:
                return float(K[0]), float(K[4]), float(K[2]), float(K[5])

        def pick(*names):
            for name in names:
                if name in data:
                    return float(data[name])
            raise KeyError(names[0])

        try:
            return (
                pick("Fx", "fx"),
                pick("Fy", "fy"),
                pick("Cx", "cx", "ppx"),
                pick("Cy", "cy", "ppy"),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _load_intrinsics(self):
        intrinsic_name = INTRINSIC_NAMES[self.camera]
        candidates = [
            os.path.join(self.seq_dir, "parameters", "sensor", intrinsic_name),
            os.path.join(self.seq_dir, intrinsic_name),
        ]
        data = None
        for sensor_path in candidates:
            if os.path.exists(sensor_path):
                with open(sensor_path, "r") as f:
                    data = json.load(f)
                break
        if data is None:
            info_path = os.path.join(self.seq_dir, "recording_info.json")
            if os.path.exists(info_path):
                with open(info_path, "r") as f:
                    info = json.load(f)
                cam_info = info.get("camera_info", {}).get(self.camera, {})
                data = cam_info.get("intrinsic")

        if data is None:
            combined_path = os.path.join(self.seq_dir, "camera_intrinsics.json")
            if os.path.exists(combined_path):
                with open(combined_path, "r") as f:
                    combined = json.load(f)
                cameras = combined.get("cameras", {})
                for camera_data in cameras.values():
                    if self._camera_data_matches(camera_data):
                        data = camera_data
                        break

        intrinsics = self._intrinsics_from_data(data)
        if intrinsics is None:
            raise FileNotFoundError(
                f"No intrinsic JSON found for Galbot camera '{self.camera}'"
            )
        return intrinsics

    def _camera_data_matches(self, camera_data) -> bool:
        topic = camera_data.get("rgb_topic") or camera_data.get("topic")
        if topic == self.rgb_topic:
            return True
        info_topic = camera_data.get("camera_info_topic")
        if info_topic == self.camera_info_topic:
            return True
        prim = camera_data.get("camera_prim") or camera_data.get("camera_frame")
        if prim and self._frame_key(prim) == self._frame_key(self.camera_frame):
            return True
        name = str(camera_data.get("camera_name", "")).lower()
        if self.camera == "head" and "head" in name:
            return True
        if self.camera == "hand_left" and ("left" in name or "hand_left" in name):
            return True
        if self.camera == "hand_right" and ("right" in name or "hand_right" in name):
            return True
        return False

    def _load_intrinsics_or_default(self):
        try:
            intrinsics = self._load_intrinsics()
            self._intrinsics_source = "json"
            return intrinsics
        except FileNotFoundError:
            if self.camera in DEFAULT_INTRINSICS:
                print(
                    "GalbotLoader: CameraInfo/JSON intrinsics unavailable; "
                    f"using built-in {self.camera} intrinsics"
                )
                self._intrinsics_source = "built-in default"
                return DEFAULT_INTRINSICS[self.camera]
            raise

    def _load_time_origin_ns(self) -> int:
        meta_path = os.path.join(self.seq_dir, "meta_info.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                clip_start = meta.get("clip_start_time")
                if clip_start is not None:
                    return int(float(clip_start) * 1_000_000_000)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        return 0

    def _get_capture(self):
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.video_path)
            if not self._cap.isOpened():
                raise IOError(f"Cannot open video: {self.video_path}")
        return self._cap

    def _read_frame_rgb(self, frame_idx: int) -> np.ndarray:
        with self._cap_lock:
            cap = self._get_capture()
            current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current != frame_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame_bgr = cap.read()
            if not ok:
                # Re-open once; OpenCV can occasionally fail after random seeks.
                cap.release()
                self._cap = cv2.VideoCapture(self.video_path)
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame_bgr = self._cap.read()
            if not ok:
                raise IOError(f"Cannot read frame {frame_idx} from {self.video_path}")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def _init_prefetch(self):
        if getattr(self, "mode", "video") == "rosbag":
            self._open_rosbag_stream()
        else:
            super()._init_prefetch()

    def _open_rosbag_stream(self):
        self._close_rosbag_stream()
        try:
            from rosbags.highlevel import AnyReader
            from rosbags.typesys import Stores, get_typestore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Reading raw Galbot/Isaac .db3 recordings requires the "
                "`rosbags` package in the jarvis environment. Install it with "
                "`/home/agxi/miniconda3/envs/jarvis/bin/python -m pip install rosbags` "
                "and rerun."
            ) from exc

        store = getattr(Stores, "ROS2_HUMBLE", None) or getattr(Stores, "ROS2_JAZZY")
        typestore = get_typestore(store)
        self._reader = AnyReader([Path(self.seq_dir)], default_typestore=typestore)
        self._reader.open()
        self._connections = {conn.topic: conn for conn in self._reader.connections}
        self._camera_info = None
        self._latest_depth = None
        self._tf_by_child = {}
        self._seen_rgb = 0
        self.index = 0
        self._prime_intrinsics_from_bag_or_json()
        self._message_iter = self._reader.messages()

    def _prime_intrinsics_from_bag_or_json(self):
        loaded_from_bag = False
        if self.camera_info_topic in self._connections:
            for connection, _timestamp, raw in self._reader.messages():
                if connection.topic != self.camera_info_topic:
                    continue
                msg = self._reader.deserialize(raw, connection.msgtype)
                self._camera_info = msg
                loaded_from_bag = self._update_intrinsics_from_camera_info(msg)
                break

        if not loaded_from_bag:
            self.fx, self.fy, self.cx, self.cy = self._load_intrinsics_or_default()

    def _close_rosbag_stream(self):
        reader = getattr(self, "_reader", None)
        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass
        self._reader = None
        self._message_iter = None

    def __iter__(self):
        if self.mode == "rosbag":
            self._open_rosbag_stream()
            return self
        return super().__iter__()

    def __next__(self) -> dict:
        if self.mode != "rosbag":
            return super().__next__()
        if self.index >= self.length:
            self._close_rosbag_stream()
            raise StopIteration
        if self._message_iter is None:
            self._open_rosbag_stream()

        while True:
            try:
                connection, timestamp, raw = next(self._message_iter)
            except StopIteration:
                self._close_rosbag_stream()
                raise

            topic = connection.topic
            if topic == self.camera_info_topic:
                msg = self._reader.deserialize(raw, connection.msgtype)
                self._camera_info = msg
                self._update_intrinsics_from_camera_info(msg)
                continue

            if topic == self.tf_topic:
                msg = self._reader.deserialize(raw, connection.msgtype)
                self._update_tf(msg)
                continue

            if self.depth_topic and topic == self.depth_topic:
                msg = self._reader.deserialize(raw, connection.msgtype)
                self._latest_depth = (self._stamp_to_ns(msg.header), self._decode_depth(msg))
                continue

            if topic != self.rgb_topic:
                continue

            frame_idx = self._seen_rgb
            self._seen_rgb += 1
            if frame_idx not in self.frame_indices:
                continue

            msg = self._reader.deserialize(raw, connection.msgtype)
            img_rgb = self._decode_rgb(msg)
            depth_np = None
            if self._latest_depth is not None:
                depth_np = self._latest_depth[1]
            datum = self._datum_from_arrays(
                img_rgb=img_rgb,
                depth_np=depth_np,
                time_ns=self._stamp_to_ns(msg.header) or int(timestamp),
                frame_idx=frame_idx,
                T_world_cam=self._current_T_world_cam(),
            )
            self.index += 1
            return datum

    @staticmethod
    def _stamp_to_ns(header) -> int:
        stamp = getattr(header, "stamp", None)
        if stamp is None:
            return 0
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _update_intrinsics_from_camera_info(self, msg):
        K = getattr(msg, "k", None)
        if K is None:
            K = getattr(msg, "K", [])
        K = list(np.asarray(K).reshape(-1))
        if len(K) >= 6 and K[0] > 0 and K[4] > 0:
            self.fx = float(K[0])
            self.fy = float(K[4])
            self.cx = float(K[2])
            self.cy = float(K[5])
            self._intrinsics_source = "bag CameraInfo"
            valid = True
        elif not getattr(self, "_warned_zero_camera_info", False):
            action = (
                "will use JSON fallback"
                if getattr(self, "_intrinsics_source", "uninitialized")
                == "uninitialized"
                else "keeping fallback intrinsics"
            )
            print(
                "GalbotLoader: CameraInfo intrinsics are empty/zero; "
                f"{action}"
            )
            self._warned_zero_camera_info = True
            valid = False
        else:
            valid = False
        self.source_width = int(getattr(msg, "width", self.source_width or 0))
        self.source_height = int(getattr(msg, "height", self.source_height or 0))
        return valid

    @staticmethod
    def _as_array_buffer(data, dtype):
        try:
            return np.frombuffer(data, dtype=dtype)
        except TypeError:
            return np.asarray(data, dtype=dtype)

    def _decode_rgb(self, msg) -> np.ndarray:
        encoding = str(msg.encoding).lower()
        h, w, step = int(msg.height), int(msg.width), int(msg.step)
        if encoding in ("rgb8", "bgr8"):
            arr = self._as_array_buffer(msg.data, np.uint8).reshape(h, step)
            img = arr[:, : w * 3].reshape(h, w, 3)
            if encoding == "bgr8":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img.copy()
        if encoding in ("rgba8", "bgra8"):
            arr = self._as_array_buffer(msg.data, np.uint8).reshape(h, step)
            img = arr[:, : w * 4].reshape(h, w, 4)
            code = cv2.COLOR_RGBA2RGB if encoding == "rgba8" else cv2.COLOR_BGRA2RGB
            return cv2.cvtColor(img, code)
        if encoding in ("mono8", "8uc1"):
            arr = self._as_array_buffer(msg.data, np.uint8).reshape(h, step)
            gray = arr[:, :w]
            return np.repeat(gray[:, :, None], 3, axis=2).copy()
        raise ValueError(f"Unsupported RGB image encoding: {msg.encoding}")

    def _decode_depth(self, msg) -> Optional[np.ndarray]:
        encoding = str(msg.encoding).lower()
        h, w, step = int(msg.height), int(msg.width), int(msg.step)
        if encoding in ("16uc1", "mono16"):
            dtype = ">u2" if int(msg.is_bigendian) else "<u2"
            row_elems = step // np.dtype(dtype).itemsize
            arr = self._as_array_buffer(msg.data, dtype).reshape(h, row_elems)
            return (arr[:, :w].astype(np.float32) / 1000.0).copy()
        if encoding == "32fc1":
            dtype = ">f4" if int(msg.is_bigendian) else "<f4"
            row_elems = step // np.dtype(dtype).itemsize
            arr = self._as_array_buffer(msg.data, dtype).reshape(h, row_elems)
            depth = arr[:, :w].astype(np.float32)
            depth[~np.isfinite(depth)] = 0.0
            return depth.copy()
        return None

    @staticmethod
    def _frame_key(frame_id: str) -> str:
        return str(frame_id).strip("/")

    @staticmethod
    def _frame_key_candidates(frame_id: str) -> list[str]:
        key = GalbotLoader._frame_key(frame_id)
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

    @staticmethod
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

    def _update_tf(self, msg):
        for tf in getattr(msg, "transforms", []):
            parent = self._frame_key(tf.header.frame_id)
            child = self._frame_key(tf.child_frame_id)
            t = tf.transform.translation
            q = tf.transform.rotation
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = self._quat_xyzw_to_R(q.x, q.y, q.z, q.w)
            T[:3, 3] = [float(t.x), float(t.y), float(t.z)]
            self._tf_by_child[child] = (parent, T)

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
        for candidate in self._frame_key_candidates(frame_id):
            if candidate in self._tf_by_child:
                return candidate
        target_suffix = "/" + self._frame_key(frame_id).split("/")[-1]
        for child in self._tf_by_child:
            if child.endswith(target_suffix):
                return child
        return None

    def _datum_from_arrays(
        self,
        img_rgb: np.ndarray,
        depth_np: Optional[np.ndarray],
        time_ns: int,
        frame_idx: int,
        T_world_cam: Optional[np.ndarray] = None,
    ) -> dict:
        src_h, src_w = img_rgb.shape[:2]
        if self.resize is not None:
            resize_h = resize_w = int(self.resize)
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

        fx = float(self.fx) * scale_x
        fy = float(self.fy) * scale_y
        cx = float(self.cx) * scale_x
        cy = float(self.cy) * scale_y

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
            sdp_w = self.sdp_from_depth(depth_np, fx, fy, cx, cy, R_wc, t_wc)
        else:
            sdp_w = torch.zeros(0, 3, dtype=torch.float32)

        return {
            "img0": self.img_to_tensor(img_rgb).float(),
            "cam0": self.pinhole_from_K(
                resize_w,
                resize_h,
                fx,
                fy,
                cx,
                cy,
                valid_radius=(resize_w, resize_h),
            ).float(),
            "T_world_rig0": PoseTW(T_wr_data),
            "sdp_w": sdp_w,
            "time_ns0": int(time_ns)
            if time_ns
            else self.time_origin_ns
            + int(round((frame_idx / self.fps) * 1_000_000_000)),
            "rotated0": torch.tensor(False).reshape(1),
            "bb2d0": torch.zeros(0, 4, dtype=torch.float32),
            "gt_labels": [],
            "obbs": ObbTW(torch.zeros(0, 165)),
        }

    def load(self, idx) -> dict:
        if self.mode == "rosbag":
            raise RuntimeError("Galbot rosbag mode is streaming-only; use next(loader)")
        frame_idx = self.frame_indices[idx]
        img_rgb = self._read_frame_rgb(frame_idx)
        return self._datum_from_arrays(
            img_rgb=img_rgb,
            depth_np=None,
            time_ns=self.time_origin_ns
            + int(round((frame_idx / self.fps) * 1_000_000_000)),
            frame_idx=frame_idx,
        )
