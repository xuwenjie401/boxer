#! /usr/bin/env python3

# pyre-unsafe
"""Extract OAK stereo image cache and smooth odometry trajectory from ROS bags."""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import rosbag2_py
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
from sensor_msgs.msg import Image
from tqdm import tqdm


DEFAULT_EXPORT = (
    "/home/wjxu22/Datasets/outputs/rtab/oak_stereo_imu_gravity_lossless_export"
)
DEFAULT_RAW_BAG = (
    "/home/wjxu22/Datasets/rosbags/"
    "oak_capture_stereo_imu_rgb_20260430_024609"
)
DEFAULT_PROCESSED_BAG = (
    "/home/wjxu22/Datasets/rosbags/processed/oak_rtabmap_pose_graph"
)
LEFT_TOPIC = "/oak/left/image_rect"
RIGHT_TOPIC = "/oak/right/image_rect"
ODOM_TOPIC = "/rtabmap/odom"


def _open_reader(bag_dir: str, topics: Iterable[str]) -> SequentialReader:
    reader = SequentialReader()
    storage_options = StorageOptions(uri=str(bag_dir), storage_id="sqlite3")
    converter_options = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)
    reader.set_filter(StorageFilter(topics=list(topics)))
    return reader


def _ensure_uint8_image(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return np.clip(img / 256, 0, 255).astype(np.uint8)
    finite = np.isfinite(img)
    if not finite.any():
        return np.zeros(img.shape, dtype=np.uint8)
    lo = float(np.min(img[finite]))
    hi = float(np.max(img[finite]))
    if hi <= lo:
        return np.zeros(img.shape, dtype=np.uint8)
    return np.clip((img - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def _image_array_from_msg(msg: Image) -> np.ndarray:
    encoding = msg.encoding.lower()
    specs = {
        "mono8": (np.uint8, 1),
        "8uc1": (np.uint8, 1),
        "mono16": (np.uint16, 1),
        "16uc1": (np.uint16, 1),
        "bgr8": (np.uint8, 3),
        "rgb8": (np.uint8, 3),
        "bgra8": (np.uint8, 4),
        "rgba8": (np.uint8, 4),
        "8uc3": (np.uint8, 3),
        "8uc4": (np.uint8, 4),
    }
    if encoding not in specs:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    dtype, channels = specs[encoding]
    dtype = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")
    row_elems = int(msg.step) // dtype.itemsize
    buf = np.frombuffer(msg.data, dtype=dtype)
    if channels == 1:
        img = buf.reshape(int(msg.height), row_elems)[:, : int(msg.width)]
    else:
        row_pixels = row_elems // channels
        img = buf.reshape(int(msg.height), row_pixels, channels)[:, : int(msg.width), :]
    if dtype.itemsize > 1 and bool(msg.is_bigendian) != (sys.byteorder == "big"):
        img = img.byteswap().view(img.dtype.newbyteorder("="))
    return np.ascontiguousarray(img)


def _image_for_imwrite(msg: Image) -> np.ndarray:
    img = _image_array_from_msg(msg)
    img = _ensure_uint8_image(np.asarray(img))
    encoding = msg.encoding.lower()
    if img.ndim == 3:
        if encoding in ("rgb8", "rgba8"):
            code = cv2.COLOR_RGB2BGR if img.shape[2] == 3 else cv2.COLOR_RGBA2BGR
            img = cv2.cvtColor(img, code)
        elif encoding == "bgra8":
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _materialize_uncompressed_bag(bag_dir: str, cache_root: Path) -> str:
    src_dir = Path(bag_dir).expanduser().resolve()
    metadata_path = src_dir / "metadata.yaml"
    if not metadata_path.exists():
        return str(src_dir)
    metadata = metadata_path.read_text(encoding="utf-8", errors="replace")
    compressed_files = sorted(src_dir.glob("*.db3.zstd"))
    if "compression_format: zstd" not in metadata and not compressed_files:
        return str(src_dir)
    if not compressed_files:
        return str(src_dir)

    zstd_bin = shutil.which("zstd")
    if zstd_bin is None:
        raise RuntimeError("zstd executable not found; cannot decompress processed bag")

    cache_dir = cache_root.expanduser().resolve() / src_dir.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    for compressed_path in compressed_files:
        out_name = compressed_path.name[: -len(".zstd")]
        out_path = cache_dir / out_name
        if not out_path.exists() or out_path.stat().st_size == 0:
            print(f"Decompress bag cache: {compressed_path} -> {out_path}")
            subprocess.run(
                [zstd_bin, "-d", "-f", "-o", str(out_path), str(compressed_path)],
                check=True,
            )

    patched_metadata = metadata.replace(".db3.zstd", ".db3")
    patched_metadata = patched_metadata.replace(
        "compression_format: zstd", 'compression_format: ""'
    )
    patched_metadata = patched_metadata.replace(
        "compression_mode: FILE", 'compression_mode: ""'
    )
    (cache_dir / "metadata.yaml").write_text(patched_metadata, encoding="utf-8")
    return str(cache_dir)


def _count_topic_messages(bag_dir: str, topic_names: Iterable[str]) -> dict[str, int]:
    metadata_path = Path(bag_dir) / "metadata.yaml"
    counts = {topic: 0 for topic in topic_names}
    if not metadata_path.exists():
        return counts
    text = metadata_path.read_text(encoding="utf-8", errors="replace")
    current_topic = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("name: "):
            current_topic = stripped.split("name: ", 1)[1].strip()
        elif stripped.startswith("message_count:") and current_topic in counts:
            counts[current_topic] = int(stripped.split(":", 1)[1].strip())
    return counts


def _nearest_pairs(left: list[dict], right: list[dict], max_dt_ns: int) -> list[dict]:
    if not left or not right:
        return []
    right_ts = np.array([item["timestamp_ns"] for item in right], dtype=np.int64)
    pairs = []
    for left_item in left:
        ts = int(left_item["timestamp_ns"])
        idx = int(np.searchsorted(right_ts, ts))
        candidates = []
        if 0 <= idx < len(right):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        if not candidates:
            continue
        best = min(candidates, key=lambda j: abs(int(right_ts[j]) - ts))
        dt_ns = int(right_ts[best]) - ts
        if abs(dt_ns) <= max_dt_ns:
            pairs.append(
                {
                    "left_index": left_item["index"],
                    "right_index": right[best]["index"],
                    "left_timestamp_ns": ts,
                    "right_timestamp_ns": int(right_ts[best]),
                    "dt_ns": dt_ns,
                    "left_path": left_item["path"],
                    "right_path": right[best]["path"],
                }
            )
    return pairs


def extract_stereo(args) -> Path:
    out_dir = Path(args.output_dir).expanduser().resolve() / "stereo_cache"
    left_dir = out_dir / "left"
    right_dir = out_dir / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    counts = _count_topic_messages(args.raw_bag, [LEFT_TOPIC, RIGHT_TOPIC])
    total = counts.get(LEFT_TOPIC, 0) + counts.get(RIGHT_TOPIC, 0)
    reader = _open_reader(args.raw_bag, [LEFT_TOPIC, RIGHT_TOPIC])
    rows = {LEFT_TOPIC: [], RIGHT_TOPIC: []}
    topic_to_dir = {LEFT_TOPIC: left_dir, RIGHT_TOPIC: right_dir}
    topic_to_side = {LEFT_TOPIC: "left", RIGHT_TOPIC: "right"}
    seen = {LEFT_TOPIC: 0, RIGHT_TOPIC: 0}

    pbar = tqdm(total=total if total > 0 else None, desc="Extract stereo")
    while reader.has_next():
        topic, data, recv_ts = reader.read_next()
        if topic not in rows:
            continue
        msg = deserialize_message(data, Image)
        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(
            msg.header.stamp.nanosec
        )
        if stamp_ns <= 0:
            stamp_ns = int(recv_ts)
        side = topic_to_side[topic]
        idx = seen[topic]
        rel_path = f"{side}/{side}_{idx:06d}.jpg"
        out_path = out_dir / rel_path
        if args.overwrite or not out_path.exists():
            img = _image_for_imwrite(msg)
            ok = cv2.imwrite(
                str(out_path),
                img,
                [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)],
            )
            if not ok:
                raise IOError(f"Failed to write {out_path}")
            height, width = img.shape[:2]
        else:
            img = cv2.imread(str(out_path), cv2.IMREAD_UNCHANGED)
            height, width = img.shape[:2] if img is not None else (msg.height, msg.width)

        rows[topic].append(
            {
                "index": idx,
                "timestamp_ns": stamp_ns,
                "timestamp_sec": stamp_ns / 1e9,
                "recv_timestamp_ns": int(recv_ts),
                "frame_id": msg.header.frame_id,
                "encoding": msg.encoding,
                "width": int(width),
                "height": int(height),
                "path": rel_path,
            }
        )
        seen[topic] += 1
        pbar.update(1)
        if args.max_images > 0 and all(v >= args.max_images for v in seen.values()):
            break
    pbar.close()

    left_rows = rows[LEFT_TOPIC]
    right_rows = rows[RIGHT_TOPIC]
    manifest = {
        "raw_bag": str(Path(args.raw_bag).expanduser().resolve()),
        "left_topic": LEFT_TOPIC,
        "right_topic": RIGHT_TOPIC,
        "left_count": len(left_rows),
        "right_count": len(right_rows),
        "max_pair_dt_ns": int(args.max_pair_dt_ms * 1e6),
        "left": left_rows,
        "right": right_rows,
        "pairs": _nearest_pairs(
            left_rows, right_rows, max_dt_ns=int(args.max_pair_dt_ms * 1e6)
        ),
    }
    manifest_path = out_dir / "stereo_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Stereo cache: {out_dir} "
        f"({len(left_rows)} left, {len(right_rows)} right, {len(manifest['pairs'])} pairs)"
    )
    return manifest_path


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / norm


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = _normalize_quat(q0)
    q1 = _normalize_quat(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return _normalize_quat(q0 + alpha * (q1 - q0))
    theta0 = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta0 = math.sin(theta0)
    theta = theta0 * alpha
    s0 = math.sin(theta0 - theta) / sin_theta0
    s1 = math.sin(theta) / sin_theta0
    return _normalize_quat(s0 * q0 + s1 * q1)


def _resample_poses_30hz(
    timestamps_ns: np.ndarray,
    positions: np.ndarray,
    quats: np.ndarray,
    rate_hz: float,
):
    if len(timestamps_ns) < 2:
        return timestamps_ns, positions, quats
    dt_ns = int(round(1e9 / rate_hz))
    out_ts = np.arange(int(timestamps_ns[0]), int(timestamps_ns[-1]) + 1, dt_ns)
    out_pos = np.empty((len(out_ts), 3), dtype=np.float64)
    out_quat = np.empty((len(out_ts), 4), dtype=np.float64)
    src_ts = timestamps_ns.astype(np.int64)
    for i, ts in enumerate(out_ts):
        hi = int(np.searchsorted(src_ts, ts, side="left"))
        if hi <= 0:
            out_pos[i] = positions[0]
            out_quat[i] = quats[0]
        elif hi >= len(src_ts):
            out_pos[i] = positions[-1]
            out_quat[i] = quats[-1]
        else:
            lo = hi - 1
            span = float(src_ts[hi] - src_ts[lo])
            alpha = 0.0 if span <= 0 else float(ts - src_ts[lo]) / span
            out_pos[i] = (1.0 - alpha) * positions[lo] + alpha * positions[hi]
            out_quat[i] = _slerp(quats[lo], quats[hi], alpha)
    return out_ts.astype(np.int64), out_pos.astype(np.float32), out_quat.astype(np.float32)


def extract_odom(args) -> Path:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_bag = _materialize_uncompressed_bag(
        args.processed_bag,
        Path(args.decompressed_bag_cache)
        if args.decompressed_bag_cache
        else out_dir / "_rosbag_cache",
    )
    counts = _count_topic_messages(processed_bag, [ODOM_TOPIC])
    reader = _open_reader(processed_bag, [ODOM_TOPIC])
    timestamps = []
    positions = []
    quats = []
    frame_ids = []
    child_frame_ids = []

    pbar = tqdm(total=counts.get(ODOM_TOPIC, 0) or None, desc="Extract odom")
    while reader.has_next():
        topic, data, recv_ts = reader.read_next()
        if topic != ODOM_TOPIC:
            continue
        msg = deserialize_message(data, Odometry)
        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(
            msg.header.stamp.nanosec
        )
        if stamp_ns <= 0:
            stamp_ns = int(recv_ts)
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        timestamps.append(stamp_ns)
        positions.append([p.x, p.y, p.z])
        quats.append([q.x, q.y, q.z, q.w])
        frame_ids.append(msg.header.frame_id)
        child_frame_ids.append(msg.child_frame_id)
        pbar.update(1)
    pbar.close()

    if not timestamps:
        raise RuntimeError(f"No {ODOM_TOPIC} messages found in {args.processed_bag}")

    timestamps_np = np.asarray(timestamps, dtype=np.int64)
    positions_np = np.asarray(positions, dtype=np.float32)
    quats_np = np.asarray([_normalize_quat(np.asarray(q)) for q in quats], dtype=np.float32)
    order = np.argsort(timestamps_np, kind="mergesort")
    timestamps_np = timestamps_np[order]
    positions_np = positions_np[order]
    quats_np = quats_np[order]

    resampled_ts, resampled_pos, resampled_quat = _resample_poses_30hz(
        timestamps_np, positions_np, quats_np, args.odom_rate_hz
    )

    out_path = out_dir / "odom_trajectory_30hz.npz"
    np.savez_compressed(
        out_path,
        source_bag=str(Path(args.processed_bag).expanduser().resolve()),
        reader_bag=str(Path(processed_bag).expanduser().resolve()),
        source_topic=ODOM_TOPIC,
        raw_timestamps_ns=timestamps_np,
        raw_positions_xyz=positions_np,
        raw_quaternions_xyzw=quats_np,
        timestamps_ns=resampled_ts,
        positions_xyz=resampled_pos,
        quaternions_xyzw=resampled_quat,
        rate_hz=np.array([args.odom_rate_hz], dtype=np.float32),
        frame_id=np.array([frame_ids[0] if frame_ids else ""]),
        child_frame_id=np.array([child_frame_ids[0] if child_frame_ids else ""]),
    )
    print(
        f"Odom trajectory: {out_path} "
        f"({len(timestamps_np)} raw, {len(resampled_ts)} @ {args.odom_rate_hz:.1f}Hz)"
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract OAK stereo images and 30Hz odom trajectory for Boxer viewer"
    )
    parser.add_argument("--export", type=str, default=DEFAULT_EXPORT)
    parser.add_argument("--raw_bag", type=str, default=DEFAULT_RAW_BAG)
    parser.add_argument("--processed_bag", type=str, default=DEFAULT_PROCESSED_BAG)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--jpeg_quality", type=int, default=90)
    parser.add_argument("--max_pair_dt_ms", type=float, default=5.0)
    parser.add_argument("--odom_rate_hz", type=float, default=30.0)
    parser.add_argument("--max_images", type=int, default=0, help="debug limit per stereo side")
    parser.add_argument("--decompressed_bag_cache", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_stereo", action="store_true")
    parser.add_argument("--skip_odom", action="store_true")
    args = parser.parse_args()

    export_dir = Path(args.export).expanduser().resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = (
            Path(__file__).resolve().parents[1] / "output" / export_dir.name
        ).resolve()
    args.output_dir = str(output_dir)

    if not args.skip_stereo:
        extract_stereo(args)
    if not args.skip_odom:
        extract_odom(args)


if __name__ == "__main__":
    main()
