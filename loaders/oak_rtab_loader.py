# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from loaders.base_loader import BaseLoader
from utils.tw.pose import PoseTW


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _matrix_from_row(row: dict[str, str], prefix: str) -> np.ndarray:
    tx = float(row[f"{prefix}_tx"])
    ty = float(row[f"{prefix}_ty"])
    tz = float(row[f"{prefix}_tz"])
    qx = float(row[f"{prefix}_qx"])
    qy = float(row[f"{prefix}_qy"])
    qz = float(row[f"{prefix}_qz"])
    qw = float(row[f"{prefix}_qw"])

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = 1.0 - 2.0 * (yy + zz)
    matrix[0, 1] = 2.0 * (xy - wz)
    matrix[0, 2] = 2.0 * (xz + wy)
    matrix[1, 0] = 2.0 * (xy + wz)
    matrix[1, 1] = 1.0 - 2.0 * (xx + zz)
    matrix[1, 2] = 2.0 * (yz - wx)
    matrix[2, 0] = 2.0 * (xz - wy)
    matrix[2, 1] = 2.0 * (yz + wx)
    matrix[2, 2] = 1.0 - 2.0 * (xx + yy)
    matrix[:3, 3] = [tx, ty, tz]
    return matrix


def _ply_dtype(type_name: str, endian: str) -> np.dtype:
    type_map = {
        "char": "i1",
        "int8": "i1",
        "uchar": "u1",
        "uint8": "u1",
        "short": "i2",
        "int16": "i2",
        "ushort": "u2",
        "uint16": "u2",
        "int": "i4",
        "int32": "i4",
        "uint": "u4",
        "uint32": "u4",
        "float": "f4",
        "float32": "f4",
        "double": "f8",
        "float64": "f8",
    }
    if type_name not in type_map:
        raise ValueError(f"Unsupported PLY property type: {type_name}")
    code = type_map[type_name]
    if code.endswith("1"):
        return np.dtype(code)
    return np.dtype(endian + code)


def _parse_ply_header(path: Path):
    with path.open("rb") as file:
        first_line = file.readline().decode("utf-8", errors="replace").strip()
        if first_line != "ply":
            raise ValueError(f"Not a PLY file: {path}")

        format_name = None
        vertex_count = None
        active_element = None
        vertex_properties: list[tuple[str, str]] = []

        while True:
            line = file.readline()
            if not line:
                raise ValueError(f"PLY header is missing end_header: {path}")
            text = line.decode("utf-8", errors="replace").strip()
            if text == "end_header":
                data_offset = file.tell()
                break
            parts = text.split()
            if not parts:
                continue
            if parts[0] == "format":
                format_name = parts[1]
            elif parts[0] == "element":
                active_element = parts[1]
                if active_element == "vertex":
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and active_element == "vertex":
                if parts[1] == "list":
                    raise ValueError("PLY list properties on vertices are unsupported.")
                vertex_properties.append((parts[2], parts[1]))

    if format_name is None or vertex_count is None:
        raise ValueError(f"PLY header is missing format or vertex count: {path}")
    names = [name for name, _ in vertex_properties]
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"PLY vertex properties must include x/y/z: {path}")

    if format_name == "binary_little_endian":
        endian = "<"
    elif format_name == "binary_big_endian":
        endian = ">"
    else:
        raise ValueError(f"Unsupported PLY format for voxel cache: {format_name}")

    dtype = np.dtype(
        [(name, _ply_dtype(type_name, endian)) for name, type_name in vertex_properties]
    )
    return vertex_count, dtype, data_offset


def _pack_quantized(q: np.ndarray) -> np.ndarray:
    """Pack signed 21-bit xyz voxel coordinates into sortable int64 keys."""
    q64 = q.astype(np.int64, copy=False)
    bias = np.int64(1 << 20)
    packed = q64 + bias
    if np.any((packed < 0) | (packed >= (1 << 21))):
        raise ValueError("Quantized voxel coordinate is outside the supported range.")
    return (
        (packed[:, 0] << np.int64(42))
        | (packed[:, 1] << np.int64(21))
        | packed[:, 2]
    )


def _aggregate_sorted(keys: np.ndarray, values: np.ndarray):
    if keys.size == 0:
        return keys, values, np.zeros(0, dtype=np.int64)
    order = np.argsort(keys, kind="mergesort")
    keys_sorted = keys[order]
    values_sorted = values[order]
    uniq, start = np.unique(keys_sorted, return_index=True)
    sums = np.add.reduceat(values_sorted, start, axis=0)
    counts = np.diff(np.r_[start, keys_sorted.shape[0]]).astype(np.int64)
    return uniq, sums, counts


def _build_voxel_cache(
    cloud_path: Path,
    cache_path: Path,
    voxel_size: float,
    chunk_points: int = 1_000_000,
):
    vertex_count, dtype, data_offset = _parse_ply_header(cloud_path)
    key_chunks = []
    sum_chunks = []
    count_chunks = []

    print(
        f"==> Building OAK voxel cache from {cloud_path} "
        f"({vertex_count:,} points, voxel={voxel_size:.3f}m)"
    )
    t0 = time.perf_counter()
    processed = 0
    with cloud_path.open("rb") as file:
        file.seek(data_offset)
        while processed < vertex_count:
            count = min(chunk_points, vertex_count - processed)
            vertex = np.fromfile(file, dtype=dtype, count=count)
            if vertex.shape[0] == 0:
                break

            xyz = np.empty((vertex.shape[0], 3), dtype=np.float32)
            xyz[:, 0] = vertex["x"]
            xyz[:, 1] = vertex["y"]
            xyz[:, 2] = vertex["z"]
            valid = np.isfinite(xyz).all(axis=1)
            xyz = xyz[valid]
            if xyz.shape[0] > 0:
                q = np.floor(xyz / voxel_size).astype(np.int64)
                keys = _pack_quantized(q)
                uniq, sums, counts = _aggregate_sorted(keys, xyz.astype(np.float64))
                key_chunks.append(uniq)
                sum_chunks.append(sums)
                count_chunks.append(counts)

            processed += vertex.shape[0]
            if processed == vertex.shape[0] or processed % (5 * chunk_points) == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"    voxelized {processed:,}/{vertex_count:,} points "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )

    if not key_chunks:
        raise ValueError(f"No valid xyz points found in {cloud_path}")

    all_keys = np.concatenate(key_chunks)
    all_sums = np.concatenate(sum_chunks, axis=0)
    all_counts = np.concatenate(count_chunks)
    order = np.argsort(all_keys, kind="mergesort")
    keys_sorted = all_keys[order]
    sums_sorted = all_sums[order]
    counts_sorted = all_counts[order]
    _, start = np.unique(keys_sorted, return_index=True)
    sums = np.add.reduceat(sums_sorted, start, axis=0)
    counts = np.add.reduceat(counts_sorted, start)

    centers = (sums / counts[:, None]).astype(np.float32)
    counts = counts.astype(np.int32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("wb") as file:
        np.savez(
            file,
            centers=centers,
            counts=counts,
            voxel_size=np.array([voxel_size], dtype=np.float32),
            source_path=np.array([str(cloud_path)]),
            source_mtime=np.array([cloud_path.stat().st_mtime], dtype=np.float64),
        )
    os.replace(tmp_path, cache_path)
    print(
        f"==> Wrote OAK voxel cache: {cache_path} "
        f"({centers.shape[0]:,} voxels, {time.perf_counter() - t0:.1f}s)"
    )
    return centers, counts


class OakRtabLoader(BaseLoader):
    """Loader for exported OAK RTAB-Map RGB poses and gravity-aligned point clouds."""

    def __init__(
        self,
        export_dir: str | Path,
        start_frame: int = 1,
        skip_frames: int = 1,
        max_frames: int | None = None,
        resize=None,
        cache_dir: str | Path | None = None,
        voxel_size: float = 0.05,
        hash_cell_size: float = 1.0,
        visibility_near: float = 0.15,
        visibility_far: float = 6.0,
        zbuffer_tolerance: float | None = None,
        max_sdp_points: int = 100_000,
        zbuffer_grid: int = 2,
    ):
        self.export_dir = Path(export_dir).expanduser().resolve()
        self.seq_name = self.export_dir.name
        self.resize = resize
        self.camera = "rgb"
        self.device_name = "OAK RTAB-Map"
        self.voxel_size = float(voxel_size)
        self.hash_cell_size = float(hash_cell_size)
        self.visibility_near = float(visibility_near)
        self.visibility_far = float(visibility_far)
        self.zbuffer_tolerance = (
            1.5 * self.voxel_size
            if zbuffer_tolerance is None
            else float(zbuffer_tolerance)
        )
        self.max_sdp_points = int(max_sdp_points)
        self.zbuffer_grid = int(zbuffer_grid)

        if self.voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        if self.hash_cell_size <= 0:
            raise ValueError("hash_cell_size must be positive")
        if self.visibility_near <= 0 or self.visibility_far <= self.visibility_near:
            raise ValueError("visibility_far must be greater than visibility_near")
        if self.max_sdp_points <= 0:
            raise ValueError("max_sdp_points must be positive")
        if self.zbuffer_grid < 1:
            raise ValueError("zbuffer_grid must be >= 1")

        self.metadata = _read_json(self.export_dir / "metadata.json")
        self.camera_info = self._load_camera_info()
        self.fx = float(self.camera_info["fx"])
        self.fy = float(self.camera_info["fy"])
        self.cx = float(self.camera_info["cx"])
        self.cy = float(self.camera_info["cy"])
        self.orig_w = int(self.camera_info["width"])
        self.orig_h = int(self.camera_info["height"])

        all_rows = self._load_pose_rows()
        start = max(0, start_frame - 1)
        rows = all_rows[start::skip_frames]
        if max_frames is not None:
            rows = rows[:max_frames]
        if not rows:
            raise ValueError(f"No valid posed RGB frames found in {self.export_dir}")
        self.rows = rows
        self.length = len(rows)
        self.index = 0

        self.cloud_path = self._resolve_cloud_path()
        if cache_dir is None:
            cache_dir = self.export_dir / ".cache"
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        cache_name = f"oak_voxels_{self.voxel_size:.3f}m.npz"
        self.voxel_cache_path = self.cache_dir / cache_name
        self.voxel_centers, self.voxel_counts = self._load_or_build_voxels()
        self._build_spatial_index()

        print(
            f"OakRtabLoader: {self.seq_name}, {self.length} frames, "
            f"{self.voxel_centers.shape[0]:,} voxels"
        )
        print(
            f"==> OAK visibility: near={self.visibility_near:.2f}m "
            f"far={self.visibility_far:.2f}m ztol={self.zbuffer_tolerance:.3f}m "
            f"max_sdp={self.max_sdp_points:,}"
        )

        self._prefetch_result = None
        self._prefetch_thread = None

    def _load_camera_info(self):
        camera_path = self.export_dir / "camera" / "rgb_camera_info.json"
        if camera_path.is_file():
            return _read_json(camera_path)
        info = self.metadata.get("rgb_camera_info")
        if info is None:
            raise FileNotFoundError(f"No RGB camera info found under {self.export_dir}")
        return info

    def _load_pose_rows(self):
        pose_csv = self.export_dir / "poses" / "rgb_poses.csv"
        rows = []
        with pose_csv.open("r", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row.get("valid") != "1" or not row.get("image_path"):
                    continue
                image_path = self.export_dir / row["image_path"]
                if not image_path.is_file():
                    continue
                rows.append(row)
        return rows

    def _resolve_cloud_path(self):
        default = self.export_dir / "pointcloud" / "rtabmap_cloud_map_latest.ply"
        if default.is_file():
            return default
        for item in self.metadata.get("cloud_exports", []):
            path_str = item.get("ply_path")
            if not path_str:
                continue
            path = Path(path_str)
            if not path.is_absolute():
                path = self.export_dir / path
            if path.is_file():
                return path
        raise FileNotFoundError(f"No map point cloud PLY found under {self.export_dir}")

    def _load_or_build_voxels(self):
        if self.voxel_cache_path.is_file():
            with np.load(self.voxel_cache_path) as data:
                cached_voxel_size = float(data["voxel_size"][0])
                source_mtime = float(data["source_mtime"][0])
                if (
                    abs(cached_voxel_size - self.voxel_size) < 1e-6
                    and abs(source_mtime - self.cloud_path.stat().st_mtime) < 1e-3
                ):
                    print(f"==> Loading OAK voxel cache: {self.voxel_cache_path}")
                    return (
                        data["centers"].astype(np.float32),
                        data["counts"].astype(np.int32),
                    )
                print(f"==> Ignoring stale OAK voxel cache: {self.voxel_cache_path}")

        return _build_voxel_cache(
            self.cloud_path,
            self.voxel_cache_path,
            voxel_size=self.voxel_size,
        )

    def _build_spatial_index(self):
        q = np.floor(self.voxel_centers / self.hash_cell_size).astype(np.int64)
        keys = _pack_quantized(q)
        order = np.argsort(keys, kind="mergesort")
        self.spatial_keys_sorted = keys[order]
        self.voxel_centers_sorted = self.voxel_centers[order]
        uniq, start = np.unique(self.spatial_keys_sorted, return_index=True)
        end = np.r_[start[1:], self.spatial_keys_sorted.shape[0]]
        self.spatial_lookup = {
            int(key): (int(s), int(e)) for key, s, e in zip(uniq, start, end)
        }
        print(
            f"==> Built OAK spatial hash: {len(self.spatial_lookup):,} "
            f"cells @ {self.hash_cell_size:.2f}m"
        )

    def _camera_for_resize(self):
        if self.resize is None:
            width, height = self.orig_w, self.orig_h
            scale_x = scale_y = 1.0
        elif isinstance(self.resize, (tuple, list)):
            height, width = int(self.resize[0]), int(self.resize[1])
            scale_x = width / self.orig_w
            scale_y = height / self.orig_h
        else:
            height = width = int(self.resize)
            scale_x = width / self.orig_w
            scale_y = height / self.orig_h

        fx = self.fx * scale_x
        fy = self.fy * scale_y
        cx = self.cx * scale_x
        cy = self.cy * scale_y
        cam = self.pinhole_from_K(
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            valid_radius=(width, height),
        )
        return cam, width, height, fx, fy, cx, cy

    def _candidate_voxels(self, cam_pos: np.ndarray):
        lo = np.floor((cam_pos - self.visibility_far) / self.hash_cell_size).astype(
            np.int64
        )
        hi = np.floor((cam_pos + self.visibility_far) / self.hash_cell_size).astype(
            np.int64
        )
        chunks = []
        for ix in range(int(lo[0]), int(hi[0]) + 1):
            for iy in range(int(lo[1]), int(hi[1]) + 1):
                qx = np.arange(int(lo[2]), int(hi[2]) + 1, dtype=np.int64)
                q = np.column_stack(
                    [
                        np.full(qx.shape, ix, dtype=np.int64),
                        np.full(qx.shape, iy, dtype=np.int64),
                        qx,
                    ]
                )
                for key in _pack_quantized(q):
                    hit = self.spatial_lookup.get(int(key))
                    if hit is None:
                        continue
                    s, e = hit
                    chunks.append(self.voxel_centers_sorted[s:e])
        if not chunks:
            return np.zeros((0, 3), dtype=np.float32)
        return np.concatenate(chunks, axis=0)

    def _visible_sdp(self, T_world_cam: np.ndarray, width, height, fx, fy, cx, cy):
        R_wc = T_world_cam[:3, :3].astype(np.float32)
        t_wc = T_world_cam[:3, 3].astype(np.float32)
        points_w = self._candidate_voxels(t_wc)
        if points_w.shape[0] == 0:
            return torch.zeros(0, 3, dtype=torch.float32)

        rel = points_w - t_wc[None, :]
        dist2 = np.einsum("ij,ij->i", rel, rel)
        radius_mask = dist2 <= self.visibility_far * self.visibility_far
        points_w = points_w[radius_mask]
        rel = rel[radius_mask]
        if points_w.shape[0] == 0:
            return torch.zeros(0, 3, dtype=torch.float32)

        points_c = rel @ R_wc
        z = points_c[:, 2]
        depth_mask = (z > self.visibility_near) & (z < self.visibility_far)
        points_w = points_w[depth_mask]
        points_c = points_c[depth_mask]
        z = z[depth_mask]
        if points_w.shape[0] == 0:
            return torch.zeros(0, 3, dtype=torch.float32)

        u = fx * (points_c[:, 0] / z) + cx
        v = fy * (points_c[:, 1] / z) + cy
        in_image = (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
        points_w = points_w[in_image]
        z = z[in_image]
        u = u[in_image]
        v = v[in_image]
        if points_w.shape[0] == 0:
            return torch.zeros(0, 3, dtype=torch.float32)

        grid = self.zbuffer_grid
        zw = max(1, int(np.ceil(width / grid)))
        zh = max(1, int(np.ceil(height / grid)))
        uu = np.floor(u / grid).astype(np.int32).clip(0, zw - 1)
        vv = np.floor(v / grid).astype(np.int32).clip(0, zh - 1)
        flat = vv * zw + uu

        zbuf = np.full(zh * zw, np.inf, dtype=np.float32)
        np.minimum.at(zbuf, flat, z.astype(np.float32))
        visible = z <= (zbuf[flat] + self.zbuffer_tolerance)
        points_w = points_w[visible]
        z = z[visible]
        if points_w.shape[0] == 0:
            return torch.zeros(0, 3, dtype=torch.float32)

        if points_w.shape[0] > self.max_sdp_points:
            # Prefer nearer visible surface points while keeping deterministic output.
            keep = np.argpartition(z, self.max_sdp_points - 1)[: self.max_sdp_points]
            points_w = points_w[keep]

        return torch.from_numpy(points_w.astype(np.float32, copy=False))

    def load(self, idx):
        row = self.rows[idx]
        image_path = self.export_dir / row["image_path"]
        img_rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_rgb is None:
            raise ValueError(f"Failed to read image: {image_path}")
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

        cam, width, height, fx, fy, cx, cy = self._camera_for_resize()
        if (img_rgb.shape[1], img_rgb.shape[0]) != (width, height):
            img_rgb = cv2.resize(
                img_rgb, (width, height), interpolation=cv2.INTER_LINEAR
            )

        T_world_cam_np = _matrix_from_row(row, "map_rgb_optical").astype(np.float32)
        T_world_cam = PoseTW.from_Rt(
            torch.from_numpy(T_world_cam_np[:3, :3]),
            torch.from_numpy(T_world_cam_np[:3, 3]),
        )

        return {
            "img0": self.img_to_tensor(img_rgb).float(),
            "cam0": cam.float(),
            "T_world_rig0": T_world_cam.float(),
            "sdp_w": self._visible_sdp(T_world_cam_np, width, height, fx, fy, cx, cy),
            "time_ns0": int(row["rgb_stamp_ns"]),
            "rotated0": torch.tensor(False).reshape(1),
            "bb2d0": torch.zeros(0, 4, dtype=torch.float32),
            "gt_labels": [],
        }
