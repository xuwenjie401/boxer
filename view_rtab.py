#! /usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
"""View OAK RTAB-Map Boxer results with RGB, trajectory, and voxel point cloud."""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
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
    SequenceOBBViewer,
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


def _stack_timed_obbs(timed_obbs: dict[int, ObbTW]) -> ObbTW:
    obbs = []
    for ts in sorted(timed_obbs.keys()):
        frame_obbs = timed_obbs[ts]
        if len(frame_obbs) > 0:
            obbs.append(frame_obbs)
    if not obbs:
        return ObbTW(torch.zeros(0, 165))
    return torch.cat(obbs, dim=0)


def _split_obbs(obbs: ObbTW) -> list[ObbTW]:
    if len(obbs) == 0:
        return []
    return [obbs[i] for i in range(len(obbs))]


def _build_rtab_seq_ctx(loader: OakRtabLoader) -> dict:
    rgb_timestamps = np.array(
        [int(row["rgb_stamp_ns"]) for row in loader.rows], dtype=np.int64
    )
    cam_template, _, _, _, _, _, _ = loader._camera_for_resize()

    traj = []
    calibs = []
    for row in loader.rows:
        T_world_cam_np = _matrix_from_row(row, "map_rgb_optical").astype(np.float32)
        T_world_cam = PoseTW.from_Rt(
            torch.from_numpy(T_world_cam_np[:3, :3]),
            torch.from_numpy(T_world_cam_np[:3, 3]),
        ).float()
        traj.append(T_world_cam)
        calibs.append(cam_template.clone().float())

    return {
        "source": "oak_rtab",
        "loader": loader,
        "rgb_num_frames": len(rgb_timestamps),
        "rgb_timestamps": rgb_timestamps,
        "rgb_images": None,
        "is_nebula": True,
        "traj": traj,
        "pose_ts": rgb_timestamps,
        "calibs": calibs,
        "calib_ts": rgb_timestamps,
        "time_to_uids_slaml": None,
        "time_to_uids_slamr": None,
        "uid_to_p3": None,
        "sdp_global": loader.voxel_centers,
    }


def _timeline_from_loader(
    loader: OakRtabLoader, raw_timed_obbs: dict[int, ObbTW]
) -> dict[int, ObbTW]:
    empty = ObbTW(torch.zeros(0, 165))
    return {
        int(row["rgb_stamp_ns"]): raw_timed_obbs.get(int(row["rgb_stamp_ns"]), empty)
        for row in loader.rows
    }


def _load_timed_csv(path: str, label: str) -> dict[int, ObbTW]:
    if not os.path.exists(path):
        print(f"==> {label} CSV not found: {path}")
        return {}
    print(f"==> Loading {label} CSV: {path}")
    timed_obbs = read_obb_csv(path)
    total = sum(len(obbs) for obbs in timed_obbs.values())
    print(f"==> Loaded {len(timed_obbs)} frames, {total} boxes from {label}")
    return timed_obbs


class RtabViewer(SequenceOBBViewer):
    title = "OAK RTAB Boxer Viewer"

    def __init__(
        self,
        *,
        rtab_loader: OakRtabLoader,
        rtab_points: np.ndarray,
        fused_obbs: ObbTW,
        initial_show_raw: bool,
        initial_show_fused: bool,
        point_size: float,
        point_alpha: float,
        max_points: int,
        **kwargs,
    ):
        self._rtab_loader = rtab_loader
        self._rtab_all_points = np.asarray(rtab_points, dtype=np.float32)
        self._rtab_points_max = int(max_points)
        self._rtab_all_raw_obbs = kwargs.get("all_obbs", ObbTW(torch.zeros(0, 165)))
        self._rtab_all_raw_timed_obbs = dict(kwargs.get("timed_obbs", {}))
        self._rtab_all_raw_embeddings = None
        self._rtab_point_positions = None
        self._rtab_point_vbo = None
        self._rtab_point_vao = None
        self._rtab_point_count = 0
        self._rtab_highlight_vbo = None
        self._rtab_highlight_vao = None
        self._rtab_highlight_count = 0
        self.show_rtab_points = True
        self.rtab_point_size = float(point_size)
        self.rtab_point_alpha = float(point_alpha)
        self._rtab_fused_obbs = fused_obbs
        self._rtab_all_fused_instances = []
        self._rtab_all_raw_labels = []
        self._rtab_all_fused_labels = []
        self._rtab_label_options = ["<all>"]
        self._rtab_label_combo_idx = 0
        self._rtab_label_filter = ""
        self._rtab_label_only = False
        self._rtab_label_highlight = True
        self._rtab_label_match_count = 0
        self._rtab_initial_show_raw = bool(initial_show_raw)
        self._rtab_initial_show_fused = bool(initial_show_fused)
        super().__init__(**kwargs)

        self._rtab_all_raw_embeddings = self._semantic_embeddings
        self._rtab_all_raw_labels = self._labels_for_obbs(self._rtab_all_raw_obbs)
        self._rtab_all_fused_labels = self._labels_for_obbs(self._rtab_fused_obbs)
        self._rtab_label_options = self._build_label_options()
        self.show_raw_set = self._rtab_initial_show_raw
        self.show_tracked_all_set = self._rtab_initial_show_fused
        self.show_tracked_visible_set = True
        self.visible_line_width = max(1, self.tracked_all_line_width + 2)
        self.frustum_scale = 0.35

        if len(self._rtab_fused_obbs) > 0:
            self._rtab_all_fused_instances = [
                SimpleNamespace(obb=obb) for obb in _split_obbs(self._rtab_fused_obbs)
            ]
            self.tracked_all_instances = list(self._rtab_all_fused_instances)
            self._build_tracked_all_geometry()

        self._apply_label_filter()
        self._upload_rtab_point_cloud()
        if self._load_view_data is None:
            self._focus_on_rtab_scene()

    def _get_3d_viewport_size(self) -> tuple[int, int]:
        w, h = self.wnd.size
        return max(1, w - self.ui_panel_width), h

    def _load_rgb_for_timestamp(self, ts_ns: int):
        if getattr(self, "_data_source", None) != "oak_rtab":
            return super()._load_rgb_for_timestamp(ts_ns)
        if len(self._rgb_timestamps) == 0:
            return None

        idx = int(find_nearest2(self._rgb_timestamps, ts_ns))
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
        return cv2.resize(img, (target_w, target_h))

    def _downsample_rtab_points(self, points: np.ndarray) -> np.ndarray:
        if self._rtab_points_max <= 0 or len(points) <= self._rtab_points_max:
            return points
        rng = np.random.default_rng(0)
        idx = rng.choice(len(points), self._rtab_points_max, replace=False)
        return points[np.sort(idx)]

    def _point_color(self) -> np.ndarray:
        if getattr(self, "visual_theme_mode", 0) == 1:
            return np.array([0.65, 0.68, 0.70], dtype=np.float32)
        return np.array([0.18, 0.20, 0.22], dtype=np.float32)

    def _upload_rtab_point_cloud(self) -> None:
        points = self._downsample_rtab_points(self._rtab_all_points)
        self._rtab_point_positions = points
        self._rtab_point_count = len(points)
        if self._rtab_point_count == 0:
            return

        colors = np.tile(self._point_color()[None, :], (len(points), 1))
        vertex_data = np.hstack([points, colors]).astype(np.float32)

        if self._rtab_point_vbo is not None:
            self._rtab_point_vbo.release()
        if self._rtab_point_vao is not None:
            self._rtab_point_vao.release()
        self._rtab_point_vbo = self.ctx.buffer(vertex_data.tobytes())
        self._rtab_point_vao = self.ctx.vertex_array(
            self.point_prog,
            [(self._rtab_point_vbo, "3f 3f", "in_position", "in_color")],
        )
        print(f"==> Uploaded {self._rtab_point_count:,} OAK voxel points")

    def _apply_visual_theme(self) -> None:
        super()._apply_visual_theme()
        if (
            getattr(self, "_rtab_point_positions", None) is not None
            and getattr(self, "point_prog", None) is not None
        ):
            self._upload_rtab_point_cloud()

    def _focus_on_rtab_scene(self) -> None:
        points = self._rtab_point_positions
        if points is None or len(points) == 0:
            return
        lo = points.min(axis=0)
        hi = points.max(axis=0)
        center = (lo + hi) * 0.5
        extent = float(np.linalg.norm(hi - lo))
        if not np.isfinite(extent) or extent <= 0.0:
            return
        self.camera_target = center.astype(np.float32)
        self.camera_distance = max(1.0, extent * 0.9)
        self.camera_azimuth = 45.0
        self.camera_elevation = 35.0

    def _labels_for_obbs(self, obbs: ObbTW) -> list[str]:
        if obbs is None or len(obbs) == 0:
            return []
        labels = obbs.text_string()
        if isinstance(labels, str):
            return [labels]
        return [str(label) for label in labels]

    def _build_label_options(self) -> list[str]:
        labels = sorted(set(self._rtab_all_raw_labels + self._rtab_all_fused_labels))
        return ["<all>"] + labels

    def _label_term(self) -> str:
        return self._rtab_label_filter.strip().lower()

    def _matching_indices(self, labels: list[str]) -> list[int]:
        term = self._label_term()
        if not term:
            return list(range(len(labels)))
        return [i for i, label in enumerate(labels) if term in label.lower()]

    def _subset_obbs_by_indices(self, obbs: ObbTW, indices: list[int]) -> ObbTW:
        if len(obbs) == 0 or not indices:
            return self._empty_obbs_like(obbs)
        idx = torch.tensor(indices, dtype=torch.long)
        return ObbTW(obbs._data[idx])

    def _clear_raw_geometry(self) -> None:
        for attr in [
            "cached_instance_vbo",
            "cached_instance_vao",
            "axis_instance_vbo",
            "axis_instance_vao",
        ]:
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.release()
                setattr(self, attr, None)
        self.cached_instance_count = 0
        self.axis_instance_count = 0

    def _set_raw_display(self, indices: list[int] | None) -> None:
        if indices is None:
            self.all_obbs = self._rtab_all_raw_obbs
            self._semantic_embeddings = self._rtab_all_raw_embeddings
        else:
            self.all_obbs = self._subset_obbs_by_indices(
                self._rtab_all_raw_obbs, indices
            )
            if self._rtab_all_raw_embeddings is not None and indices:
                idx = torch.tensor(indices, dtype=torch.long)
                self._semantic_embeddings = self._rtab_all_raw_embeddings[idx]
            else:
                self._semantic_embeddings = None
        self.total_detections = len(self.all_obbs)
        self._cached_filtered_obbs = None
        self._cached_filtered_indices = None
        self._cached_prob_threshold = None
        self._clear_raw_geometry()
        if len(self.all_obbs) > 0:
            self._build_geometry_cache()

    def _filtered_timed_obbs(self, only_matching: bool) -> dict[int, ObbTW]:
        if not only_matching or not self._label_term():
            return dict(self._rtab_all_raw_timed_obbs)
        filtered = {}
        for ts, obbs in self._rtab_all_raw_timed_obbs.items():
            labels = self._labels_for_obbs(obbs)
            indices = self._matching_indices(labels)
            filtered[ts] = self._subset_obbs_by_indices(obbs, indices)
        return filtered

    def _clear_tracked_geometry(self) -> None:
        for attr in [
            "tracked_all_instance_vbo",
            "tracked_all_instance_vao",
            "outline_instance_vbo",
            "outline_instance_vao",
        ]:
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.release()
                setattr(self, attr, None)
        self.tracked_all_instance_count = 0
        self.outline_instance_count = 0
        self.tracked_all_text_labels = []
        self.tracked_all_label_positions = []
        self.tracked_all_label_colors = []

    def _clear_highlight_geometry(self) -> None:
        if self._rtab_highlight_vbo is not None:
            self._rtab_highlight_vbo.release()
            self._rtab_highlight_vbo = None
        if self._rtab_highlight_vao is not None:
            self._rtab_highlight_vao.release()
            self._rtab_highlight_vao = None
        self._rtab_highlight_count = 0

    def _build_highlight_geometry(self, obbs: ObbTW) -> None:
        self._clear_highlight_geometry()
        if len(obbs) == 0:
            return
        corners = obbs.bb3corners_world.cpu()
        starts = []
        ends = []
        for i, j in self._highlight_edges():
            starts.append(corners[:, i])
            ends.append(corners[:, j])
        start_pts = torch.stack(starts, dim=1)
        end_pts = torch.stack(ends, dim=1)
        color = torch.tensor([1.0, 0.78, 0.05], dtype=torch.float32)
        colors = color.reshape(1, 1, 3).expand(len(obbs), 12, 3)
        probs = torch.ones(len(obbs), 12, 1, dtype=torch.float32)
        data = (
            torch.cat([start_pts, end_pts, colors, probs], dim=2)
            .reshape(-1, 10)
            .numpy()
            .astype("f4")
        )
        self._rtab_highlight_count = len(data)
        self._rtab_highlight_vbo = self.ctx.buffer(data.tobytes())
        self._rtab_highlight_vao = self.ctx.vertex_array(
            self.line_prog,
            [
                (self.quad_vbo, "2f", "in_quad_pos"),
                (
                    self._rtab_highlight_vbo,
                    "3f 3f 3f 1f /i",
                    "start_pos",
                    "end_pos",
                    "line_color",
                    "line_prob",
                ),
            ],
        )

    def _highlight_edges(self):
        from utils.tw.obb import BB3D_LINE_ORDERS

        return BB3D_LINE_ORDERS

    def _apply_label_filter(self) -> None:
        term = self._label_term()
        raw_matches = self._matching_indices(self._rtab_all_raw_labels)
        fused_matches = self._matching_indices(self._rtab_all_fused_labels)
        self._rtab_label_match_count = len(raw_matches) + len(fused_matches)

        only = bool(term and self._rtab_label_only)
        self._set_raw_display(raw_matches if only else None)
        self.timed_obbs = self._filtered_timed_obbs(only)
        self.sorted_timestamps = sorted(self.timed_obbs.keys())
        self.total_frames = len(self.sorted_timestamps)
        self.current_frame_idx = min(self.current_frame_idx, max(0, self.total_frames - 1))

        if only:
            self.tracked_all_instances = [
                self._rtab_all_fused_instances[i] for i in fused_matches
            ]
        else:
            self.tracked_all_instances = list(self._rtab_all_fused_instances)
        self._clear_tracked_geometry()
        if self.tracked_all_instances:
            self._build_tracked_all_geometry()

        if term and self._rtab_label_highlight and not only:
            highlight_parts = []
            if raw_matches:
                highlight_parts.append(
                    self._subset_obbs_by_indices(self._rtab_all_raw_obbs, raw_matches)
                )
            if fused_matches:
                highlight_parts.append(
                    self._subset_obbs_by_indices(self._rtab_fused_obbs, fused_matches)
                )
            if highlight_parts:
                self._build_highlight_geometry(torch.cat(highlight_parts, dim=0))
            else:
                self._clear_highlight_geometry()
        else:
            self._clear_highlight_geometry()

        if self.total_frames > 0:
            self._step_to_frame(self.current_frame_idx)

    def render_3d(self, time_val: float, frame_time: float) -> None:
        super().render_3d(time_val, frame_time)
        if (
            not self.show_rtab_points
            or self._rtab_point_vao is None
            or self._rtab_point_count <= 0
        ):
            return

        full_w, h_full = self.wnd.size
        w, h = self._get_3d_viewport_size()
        vp_x = full_w - w
        self.ctx.viewport = (vp_x, 0, w, h)
        self.ctx.scissor = (vp_x, 0, w, h)

        mvp = self._last_render_mvp
        if mvp is None:
            _, _, mvp = self.get_camera_matrices()
        self.ctx.enable(self.ctx.PROGRAM_POINT_SIZE)
        self.point_prog["mvp"].write(np.asarray(mvp, dtype="f4").tobytes())
        self.point_prog["point_size"].write(
            np.array(self.rtab_point_size, dtype="f4").tobytes()
        )
        self.point_prog["alpha"].write(
            np.array(self.rtab_point_alpha, dtype="f4").tobytes()
        )
        self._rtab_point_vao.render(
            mode=self.ctx.POINTS, vertices=self._rtab_point_count
        )

        if self.show_frustum and self.frustum_instance_vao is not None:
            self.ctx.disable(self.ctx.DEPTH_TEST)
            self.line_prog["mvp"].write(np.asarray(mvp, dtype="f4").tobytes())
            self.line_prog["alpha"].write(np.array(1.0, dtype="f4").tobytes())
            self.line_prog["prob_threshold"].write(
                np.array(0.0, dtype="f4").tobytes()
            )
            self.line_prog["line_width"].value = 4.0
            self.line_prog["viewport_size"].write(
                np.array([w, h], dtype="f4").tobytes()
            )
            self.frustum_instance_vao.render(
                mode=self.ctx.TRIANGLES, instances=self.frustum_instance_count
            )

        if self._rtab_highlight_vao is not None and self._rtab_highlight_count > 0:
            self.ctx.disable(self.ctx.DEPTH_TEST)
            self.line_prog["mvp"].write(np.asarray(mvp, dtype="f4").tobytes())
            self.line_prog["alpha"].write(np.array(1.0, dtype="f4").tobytes())
            self.line_prog["prob_threshold"].write(
                np.array(0.0, dtype="f4").tobytes()
            )
            self.line_prog["line_width"].value = float(
                max(self.tracked_all_line_width + 3, 7)
            )
            self.line_prog["viewport_size"].write(
                np.array([w, h], dtype="f4").tobytes()
            )
            self._rtab_highlight_vao.render(
                mode=self.ctx.TRIANGLES, instances=self._rtab_highlight_count
            )

        self.ctx.viewport = (0, 0, full_w, h_full)
        self.ctx.scissor = None

    def _render_compact_rgb_panel(self, panel_x: int, panel_y: int, panel_w: int, panel_h: int) -> None:
        if self._rgb_texture is None or not self.show_rgb or panel_h <= 40:
            return
        tex_w, tex_h = self._rgb_tex_size
        imgui.set_next_window_position(panel_x, panel_y, imgui.ALWAYS)
        imgui.set_next_window_size(panel_w, panel_h, imgui.ALWAYS)
        expanded, _ = imgui.begin(
            "RGB View",
            flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE,
        )
        if expanded:
            avail_w, avail_h = imgui.get_content_region_available()
            if avail_w > 0 and avail_h > 0:
                img_scale = min(avail_w / tex_w, avail_h / tex_h)
                draw_w = tex_w * img_scale
                draw_h = tex_h * img_scale
                imgui.image(self._rgb_texture.glo, draw_w, draw_h)
                img_min = imgui.get_item_rect_min()
                scale_x = draw_w / tex_w * self._rgb_img_scale
                scale_y = draw_h / tex_h * self._rgb_img_scale
                draw_list = imgui.get_window_draw_list()

                if self.show_rgb_obbs and self.show_rgb_raw:
                    for edge_pts, edge_valid, color in self._rgb_projected_raw_lines:
                        col = imgui.get_color_u32_rgba(
                            float(color[0]), float(color[1]), float(color[2]), 1.0
                        )
                        for e in range(edge_pts.shape[0]):
                            for s in range(edge_pts.shape[1] - 1):
                                if edge_valid[e, s] and edge_valid[e, s + 1]:
                                    draw_list.add_line(
                                        img_min.x + edge_pts[e, s, 0] * scale_x,
                                        img_min.y + edge_pts[e, s, 1] * scale_y,
                                        img_min.x + edge_pts[e, s + 1, 0] * scale_x,
                                        img_min.y + edge_pts[e, s + 1, 1] * scale_y,
                                        col,
                                        self.rgb_obb_thickness,
                                    )

                if self.show_rgb_obbs and self.show_rgb_tracked_all:
                    for edge_pts, edge_valid, color in self._rgb_projected_tracked_all_lines:
                        col = imgui.get_color_u32_rgba(
                            float(color[0]), float(color[1]), float(color[2]), 1.0
                        )
                        for e in range(edge_pts.shape[0]):
                            for s in range(edge_pts.shape[1] - 1):
                                if edge_valid[e, s] and edge_valid[e, s + 1]:
                                    draw_list.add_line(
                                        img_min.x + edge_pts[e, s, 0] * scale_x,
                                        img_min.y + edge_pts[e, s, 1] * scale_y,
                                        img_min.x + edge_pts[e, s + 1, 0] * scale_x,
                                        img_min.y + edge_pts[e, s + 1, 1] * scale_y,
                                        col,
                                        self.rgb_obb_thickness,
                                    )

                if self.show_rgb_obbs and self.show_rgb_tracked_visible:
                    for edge_pts, edge_valid, color in self._rgb_projected_tracked_visible_lines:
                        col = imgui.get_color_u32_rgba(
                            float(color[0]), float(color[1]), float(color[2]), 1.0
                        )
                        for e in range(edge_pts.shape[0]):
                            for s in range(edge_pts.shape[1] - 1):
                                if edge_valid[e, s] and edge_valid[e, s + 1]:
                                    draw_list.add_line(
                                        img_min.x + edge_pts[e, s, 0] * scale_x,
                                        img_min.y + edge_pts[e, s, 1] * scale_y,
                                        img_min.x + edge_pts[e, s + 1, 0] * scale_x,
                                        img_min.y + edge_pts[e, s + 1, 1] * scale_y,
                                        col,
                                        self.rgb_obb_thickness + 1.0,
                                    )

                if self.show_rgb_tracked_visible:
                    self._rgb_projected_labels = list(
                        self._rgb_projected_tracked_visible_labels
                    )
                elif self.show_rgb_tracked_all:
                    self._rgb_projected_labels = list(
                        self._rgb_projected_tracked_all_labels
                    )
                else:
                    self._rgb_projected_labels = []

                if self.show_rgb_labels and self._rgb_projected_labels:
                    self._draw_projected_labels(
                        draw_list,
                        self._rgb_projected_labels,
                        img_min,
                        scale_x,
                        scale_y,
                    )
        imgui.end()

    def render_ui(self) -> None:
        if self.show_tracked_all_set:
            self._render_text_labels()

        _win_w, win_h = self.wnd.size
        rgb_visible = self._rgb_texture is not None and self.show_rgb
        if rgb_visible and win_h > 520:
            rgb_h = min(max(int(win_h * 0.48), 240), win_h - 260)
        else:
            rgb_h = 0
        control_h = win_h - rgb_h

        imgui.set_next_window_position(0, 0, imgui.ALWAYS)
        imgui.set_next_window_size(self.ui_panel_width, control_h, imgui.ALWAYS)
        imgui.begin("OBB Controls", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE)
        self._render_main_controls()
        imgui.end()

        self._render_compact_rgb_panel(0, control_h, self.ui_panel_width, rgb_h)

    def _render_main_controls(self):
        imgui.text(f"OAK RTAB: {self.seq_name}")
        imgui.text(f"Frames: {self.total_frames}")
        imgui.text(f"Voxel points: {self._rtab_point_count:,}")
        imgui.separator()

        self._render_common_visual_controls(
            tracked_all_checkbox_label="Show Fused 3DBBs",
            show_visible_line_width=True,
        )

        self._section_header("Frame")
        if self.total_frames > 0:
            imgui.push_item_width(200)
            changed, new_idx = imgui.slider_int(
                "Index", self.current_frame_idx, 0, self.total_frames - 1
            )
            imgui.pop_item_width()
            if changed:
                self._step_to_frame(int(new_idx))
            if imgui.button("<", width=45, height=28):
                self._step_to_frame(max(0, self.current_frame_idx - 1))
            imgui.same_line()
            if imgui.button(">", width=45, height=28):
                self._step_to_frame(
                    min(self.total_frames - 1, self.current_frame_idx + 1)
                )
            ts = self.sorted_timestamps[self.current_frame_idx]
            frame_obbs = self.timed_obbs.get(ts)
            n_obbs = len(frame_obbs) if frame_obbs is not None else 0
            imgui.text(f"{self.current_frame_idx + 1}/{self.total_frames}")
            imgui.text(f"Frame boxes: {n_obbs}")

        self._section_header("Point Cloud")
        _changed, self.show_rtab_points = imgui.checkbox(
            "Show Voxels", self.show_rtab_points
        )
        imgui.push_item_width(200)
        _changed, self.rtab_point_size = imgui.slider_float(
            "Voxel Size", self.rtab_point_size, 1.0, 8.0
        )
        _changed, self.rtab_point_alpha = imgui.slider_float(
            "Voxel Alpha", self.rtab_point_alpha, 0.02, 1.0
        )
        imgui.pop_item_width()

        self._section_header("Label Filter")
        imgui.push_item_width(200)
        edited, new_text = imgui.input_text("##rtab_label_filter", self._rtab_label_filter, 128)
        if edited:
            self._rtab_label_filter = new_text
            self._rtab_label_combo_idx = 0
            self._apply_label_filter()
        combo_changed, new_idx = imgui.combo(
            "Label", self._rtab_label_combo_idx, self._rtab_label_options
        )
        if combo_changed:
            self._rtab_label_combo_idx = new_idx
            self._rtab_label_filter = "" if new_idx == 0 else self._rtab_label_options[new_idx]
            self._apply_label_filter()
        imgui.pop_item_width()
        changed, self._rtab_label_only = imgui.checkbox(
            "Only Matching", self._rtab_label_only
        )
        if changed:
            self._apply_label_filter()
        changed, self._rtab_label_highlight = imgui.checkbox(
            "Highlight Matching", self._rtab_label_highlight
        )
        if changed:
            self._apply_label_filter()
        if imgui.button("Clear Label Filter", width=200, height=28):
            self._rtab_label_filter = ""
            self._rtab_label_combo_idx = 0
            self._apply_label_filter()
        if self._label_term():
            imgui.text(f"Matches: {self._rtab_label_match_count}")

        if len(self.all_obbs) > 0:
            self._render_fusion_controls()


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="View OAK RTAB-Map Boxer results")
    parser.add_argument("--input", type=str, default=DEFAULT_RTAB_EXPORT, help="OAK RTAB export directory")
    parser.add_argument("--output_dir", type=str, default=EVAL_PATH, help="Where Boxer CSVs live")
    parser.add_argument("--write_name", default="boxer", type=str, help="CSV prefix")
    parser.add_argument("--skip_n", type=int, default=1, help="subsample loaded raw OBB timestamps")
    parser.add_argument("--start_n", type=int, default=0, help="start from n-th raw OBB timestamp")
    parser.add_argument("--max_n", type=int, default=0, help="max raw OBB timestamps to load")
    parser.add_argument("--load_view", type=str, nargs="?", const="DEFAULT", default=None)
    parser.add_argument("--window_w", type=int, default=0, help="Initial window width (0 = default)")
    parser.add_argument("--window_h", type=int, default=0, help="Initial window height (0 = default)")
    parser.add_argument("--init_color_mode", type=str, default=None, help="Initial 3DBB color mode")
    parser.add_argument("--init_rgb_text_scale", type=float, default=None, help="Initial RGB label text scale")
    parser.add_argument("--init_image_panel_width", type=float, default=None, help="Initial image panel width fraction")
    parser.add_argument("--loader_skip", type=int, default=1, help="subsample OAK RGB frames for empty timeline/trajectory")
    parser.add_argument("--loader_start", type=int, default=1, help="1-based OAK RGB start frame")
    parser.add_argument("--loader_max", type=int, default=0, help="max OAK RGB frames to load (0 = all)")
    parser.add_argument("--oak_voxel_size", type=float, default=0.05, help="OAK RTAB voxel size in meters")
    parser.add_argument("--oak_hash_cell_size", type=float, default=1.0, help="OAK RTAB spatial hash cell size in meters")
    parser.add_argument("--oak_visibility_near", type=float, default=0.15, help="OAK RTAB near visibility cutoff in meters")
    parser.add_argument("--oak_visibility_far", type=float, default=6.0, help="OAK RTAB far visibility cutoff in meters")
    parser.add_argument("--oak_zbuffer_tolerance", type=float, default=None, help="OAK RTAB z-buffer visibility tolerance in meters")
    parser.add_argument("--oak_max_sdp_points", type=int, default=100000, help="OAK RTAB max visible SDP points per frame")
    parser.add_argument("--oak_zbuffer_grid", type=int, default=2, help="OAK RTAB z-buffer grid size in pixels")
    parser.add_argument("--max_points", type=int, default=250000, help="max voxel points rendered in 3D (0 = all)")
    parser.add_argument("--point_size", type=float, default=2.0, help="initial voxel point size")
    parser.add_argument("--point_alpha", type=float, default=0.35, help="initial voxel point alpha")
    parser.add_argument("--raw_csv", type=str, default="", help="override raw 3DBB CSV path")
    parser.add_argument("--fused_csv", type=str, default="", help="override fused 3DBB CSV path")
    parser.add_argument("--no_raw", action="store_true", help="start with raw 3DBBs hidden")
    parser.add_argument("--no_fused", action="store_true", help="start with fused 3DBBs hidden")
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
    view_path, load_view_data = load_view_file(log_dir, args.load_view)

    loader = OakRtabLoader(
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
    seq_ctx = _build_rtab_seq_ctx(loader)

    raw_csv = args.raw_csv or os.path.join(log_dir, f"{args.write_name}_3dbbs.csv")
    fused_csv = args.fused_csv or os.path.join(
        log_dir, f"{args.write_name}_3dbbs_fused.csv"
    )
    raw_timed_obbs = _load_timed_csv(raw_csv, "raw")
    raw_timed_obbs = subsample_timed_obbs(
        raw_timed_obbs, skip_n=args.skip_n, start_n=args.start_n, max_n=args.max_n
    )
    fused_timed_obbs = _load_timed_csv(fused_csv, "fused")

    timed_obbs = _timeline_from_loader(loader, raw_timed_obbs)
    all_obbs = _stack_timed_obbs(raw_timed_obbs)
    fused_obbs = _stack_timed_obbs(fused_timed_obbs)

    default_w, default_h = 2250 * scale_factor, 1100 * scale_factor
    init_w = args.window_w if args.window_w > 0 else default_w
    init_h = args.window_h if args.window_h > 0 else default_h

    class Viewer(RtabViewer):
        window_size = (init_w, init_h)

        def __init__(self, **kw):
            super().__init__(
                rtab_loader=loader,
                rtab_points=loader.voxel_centers,
                fused_obbs=fused_obbs,
                initial_show_raw=(not args.no_raw and len(all_obbs) > 0),
                initial_show_fused=(not args.no_fused and len(fused_obbs) > 0),
                point_size=args.point_size,
                point_alpha=args.point_alpha,
                max_points=args.max_points,
                all_obbs=all_obbs,
                root_path=log_dir,
                timed_obbs=timed_obbs,
                seq_ctx=seq_ctx,
                init_rgb_text_scale=args.init_rgb_text_scale,
                init_color_mode=args.init_color_mode,
                init_image_panel_width=args.init_image_panel_width,
                skip_precompute=(len(all_obbs) == 0),
                load_view_data=load_view_data,
                view_save_path=view_path,
                seq_name=seq_name,
                **kw,
            )

    launch_viewer(Viewer)


if __name__ == "__main__":
    main()
