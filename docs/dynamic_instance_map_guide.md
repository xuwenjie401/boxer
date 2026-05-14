# Boxer 源码导读：面向动态物体实例地图维护的改造指南

本文面向一个具体目标：阅读并复用现有 BoxerNet，把 Boxer 从“单帧 2D box lifting + 静态 3D box 融合”改造成“可维护动态物体实例地图”的系统。

核心判断：不建议优先改 BoxerNet 网络结构或 checkpoint。BoxerNet 当前已经能把单帧 2D 检测提升为 3D OBB；动态实例地图的主要工作应放在预处理、后处理和系统层。

## 1. 现有系统的主链路

主入口是 `run_boxer.py`。它把 loader、2D detector、BoxerNet、CSV 输出、在线 tracker 串在一个循环里。

```text
loader datum
  -> 2D detector / GT 2D / cached 2D
  -> BoxerNet forward
  -> 3D OBB confidence filter + label/text fill
  -> CSV output
  -> optional online tracker
  -> optional visualization
```

重点阅读位置：

- `run_boxer.py:260`：设备、labels、detector、BoxerNet、tracker 初始化。
- `run_boxer.py:421`：每帧 2D box 来源，包括 `--cache2d`、`--gt2d`、OWLv2。
- `run_boxer.py:498`：把 `datum["bb2d"]` 交给 BoxerNet 做 3D lifting。
- `run_boxer.py:535`：按 3D confidence 过滤并把 2D label 写进 `ObbTW`。
- `run_boxer.py:578`：现有 online tracker 调用点，是动态地图系统最自然的接入位置。

## 2. 每帧数据契约

所有 loader 的公共契约在 `loaders/base_loader.py:16`。

每帧 `datum` 至少应包含：

```python
{
    "img0": Tensor,          # (1, C, H, W), float32, [0, 1]
    "cam0": CameraTW,        # camera intrinsics/extrinsics
    "T_world_rig0": PoseTW,  # world-from-rig pose
    "sdp_w": Tensor,         # (N, 3), semi-dense world points
    "time_ns0": int,         # timestamp in ns
    "rotated0": Tensor,      # whether image was rotated
}
```

如果使用 GT 2D，还会用到：

```python
{
    "bb2d0": Tensor,  # (M, 4), Boxer 内部格式 [xmin, xmax, ymin, ymax]
    "obbs": ObbTW,
}
```

注意 2D box 格式有两个约定：

- Boxer 内部：`[xmin, xmax, ymin, ymax]`
- CSV/常见 detector：`[x1, y1, x2, y2]`

`run_boxer.py:437` 和 `run_boxer.py:558` 分别负责 CSV 格式和 Boxer 格式之间的转换。

## 3. BoxerNet 本体怎么工作

BoxerNet 在 `boxernet/boxernet.py`。

高层结构：

```text
RGB image
  -> DINOv3 patch features

semi-dense points in world
  -> project into current camera
  -> median depth per image patch

DINO feature + patch depth (+ optional ray encoding)
  -> input tokens
  -> self-attention

2D boxes
  -> normalized query tokens
  -> cross-attend to input tokens
  -> AleHead predicts center, size, yaw, uncertainty
  -> ObbTW in world coordinates
```

关键位置：

- `boxernet/boxernet.py:625`：`prepare_inputs()` 读取 `img0/cam0/T_world_rig0/sdp_w/bb2d`。
- `boxernet/boxernet.py:261`：`sdp_to_patches()` 把 world points 投影成 patch-level depth。
- `boxernet/boxernet.py:667`：DINO feature 与 SDP patch depth 拼接。
- `boxernet/boxernet.py:706`：2D box query cross-attention。
- `boxernet/boxernet.py:130`：`AleHead` 输出 7DoF OBB 和 aleatoric uncertainty。
- `boxernet/boxernet.py:197`：用 `prob = 1 / (1 + sigma2)` 把 uncertainty 转成 3D confidence。

动态系统里建议把 BoxerNet 视为“单帧 observation generator”，不要让它直接承担跨帧状态维护。

## 4. 2D 检测与预处理改造点

OWLv2 wrapper 在 `owl/owl_wrapper.py`。

重点位置：

- `owl/owl_wrapper.py:351`：resize + normalize。
- `owl/owl_wrapper.py:386`：classification score threshold。
- `owl/owl_wrapper.py:408`：过滤过大/过小 2D box。
- `owl/owl_wrapper.py:419`：per-class NMS。

现有代码默认过滤小于图像宽/高 5% 的 box：

```python
too_small = (x2 - x1 < 0.05 * WW) | (y2 - y1 < 0.05 * HH)
```

如果要维护动态物体，远处行人、手持物、移动小物体可能会被过滤掉。建议改成 class-specific threshold，例如：

- `person`、`bag`、`bottle` 等允许更小 box。
- 大件家具仍保留较严格的最小尺寸。
- 对动态类使用更高 temporal confirmation，而不是单帧强过滤。

建议预处理改造：

1. 优先通过 `--cache2d` 接入外部 2D detector / 2D tracker。
2. 扩展 `owl_2dbbs.csv` 的 `instance` 列，让 2D track id 进入 3D 关联。
3. 对动态类使用专门 prompt list，不要直接全量 `lvisplus`。
4. 对有深度的输入，尽量使用当前帧 depth 生成 `sdp_w`；SLAM semi-dense points 往往偏静态，不一定覆盖运动物体。

相关 CSV 读写在 `utils/file_io.py`：

- `save_bb2d_csv()`：`utils/file_io.py:1296`
- `load_bb2d_csv()`：`utils/file_io.py:1389`

目前 `save_bb2d_csv()` 会把 `instance` 写死为 `-1`，在 `utils/file_io.py:1369` 附近。这是接入 2D tracking id 的一个直接改造点。

## 5. OBB 数据结构

核心结构是 `utils/tw/obb.py` 里的 `ObbTW`。

`ObbTW` 是一个 165 维 packed tensor，主要字段包括：

```text
bb3_object       0:6
bb2_rgb          6:10
bb2_slaml        10:14
bb2_slamr        14:18
T_world_object   18:30
sem_id           30
inst_id          31
prob             32
moveable         33
color            34:37
text             37:165
```

创建入口在 `utils/tw/obb.py:98`：

```python
ObbTW.from_lmc(
    bb3_object=...,
    T_world_object=...,
    sem_id=...,
    inst_id=...,
    prob=...,
    moveable=...,
    text=...,
)
```

动态系统可以用 `moveable` 临时标记“可动/动态类别”，但不要把速度、轨迹、生命周期硬塞进 `ObbTW`。建议单独维护 `InstanceRecord`。

可复用的几何 API：

- `get_pseudo_bb2()`：`utils/tw/obb.py:742`，把 3D OBB 投影成 2D box。
- `batch_points_inside_bb3()`：`utils/tw/obb.py:800`，判断观测点是否落在 OBB 内。
- `iou_mc7()` / `iou_mc7_sparse()`：3D OBB IoU，tracker 和 fusion 都在用。

## 6. 静态 fusion 为什么不适合动态物体

静态后处理在 `utils/fuse_3d_boxes.py`。

`BoundingBox3DFuser` 的算法：

```text
all per-frame OBBs
  -> confidence filter
  -> pairwise 3D IoU
  -> connected components clustering
  -> confidence-weighted average
  -> min detection count filter
  -> optional NMS
```

入口位置：

- `utils/fuse_3d_boxes.py:276`：`BoundingBox3DFuser`
- `utils/fuse_3d_boxes.py:311`：`fuse()`
- `utils/fuse_3d_boxes.py:469`：IoU graph clustering

这个逻辑假设同一实例在世界坐标下基本静止。动态物体如果直接用 `--fuse`，跨帧位置会被聚类或平均，最终地图会产生拖影、错位或丢失轨迹信息。

结论：动态实例地图不要直接使用 `boxer_3dbbs_fused.csv` 作为最终输出。

## 7. 现有 online tracker 可复用什么

现有 tracker 在 `utils/track_3d_boxes.py`。

当前设计：

```text
per-frame OBB detections
  -> confidence filter
  -> visible tracks only
  -> V x N 3D IoU matrix
  -> Hungarian assignment
  -> matched tracks: confidence-weighted OBB average
  -> unmatched detections: create track
  -> unmatched tracks: visibility-aware aging
  -> duplicate track merge
```

重点位置：

- `utils/track_3d_boxes.py:48`：`TrackState`，只有 `TENTATIVE/ACTIVE/INACTIVE`。
- `utils/track_3d_boxes.py:68`：`TrackedInstance`，目前没有速度、动态性、完整轨迹历史。
- `utils/track_3d_boxes.py:175`：`update()`。
- `utils/track_3d_boxes.py:233`：3D IoU + Hungarian 关联。
- `utils/track_3d_boxes.py:379`：matched track 的 OBB 融合。
- `utils/track_3d_boxes.py:864`：visibility-aware aging。

可复用部分：

- 3D IoU 关联框架。
- Hungarian assignment。
- semantic embedding cache。
- 2D projection visibility。
- semidense points containment check。
- track promotion / aging 的基本框架。

需要改造部分：

- 增加 timestamp-aware motion model。
- 区分 static instance 和 dynamic instance。
- 动态对象不要做长期世界坐标加权平均。
- 关联 cost 不应只依赖 3D IoU。
- 输出应保留每帧状态或轨迹，而不是只输出 timestamp=0 的最终 box。

## 8. 建议的动态实例地图系统设计

建议新增一个系统层文件，例如：

```text
utils/dynamic_instance_map.py
```

推荐核心数据结构：

```python
@dataclass
class BoxObservation:
    obb: ObbTW
    timestamp_ns: int
    frame_idx: int
    label: str
    score2d: float
    score3d: float
    bb2d: torch.Tensor | None
    cam: CameraTW | None
    T_world_rig: PoseTW | None


@dataclass
class InstanceRecord:
    instance_id: int
    label_distribution: dict[str, float]
    state: str                 # tentative, active, lost, removed
    motion_state: str          # static, dynamic, unknown
    latest_obb: ObbTW
    velocity_world: torch.Tensor
    history: list[BoxObservation]
    confidence_mass: float
    first_seen_ns: int
    last_seen_ns: int
    missed_count: int
```

主类：

```python
class DynamicInstanceMap:
    def update(self, observations: list[BoxObservation], frame_ctx) -> list[InstanceRecord]:
        ...
```

关联代价建议从当前纯 3D IoU 改成混合 cost：

```text
cost = w_iou    * (1 - iou3d)
     + w_center * center_distance(predicted_center, det_center)
     + w_sem    * (1 - semantic_similarity)
     + w_2d     * (1 - projected_iou2d)
     + w_motion * motion_residual
```

gating 建议：

- semantic 不兼容：直接禁止匹配，除非在 override 表里。
- 3D center 距离超过上限：禁止匹配。
- 动态对象使用预测位置做 gating，而不是 last OBB。
- 有 2D track id 时，同 id 可降低 cost，不同 id 可提高 cost。

动态性判定可以先做启发式：

```text
if label in movable_classes:
    prior = dynamic_possible

if center displacement / dt > velocity_threshold:
    motion_state = dynamic

if many observations and velocity consistently small:
    motion_state = static
```

典型 movable classes：

```text
person, dog, cat, bag, backpack, cart, stroller, bicycle, chair, box
```

其中 `chair`、`box` 等应是 movable-but-currently-static，不要直接当 dynamic。

## 9. 静态和动态实例的更新策略

静态实例：

- 可以沿用现有 confidence-weighted fusion。
- 可以使用 `utils/fuse_3d_boxes.py` 里的 yaw/size alignment helper。
- 可输出到静态 instance map。

动态实例：

- 不做长期世界坐标平均。
- `latest_obb` 或短期 EMA 作为当前状态。
- 维护 `velocity_world`。
- 保存 history，用于轨迹和重识别。
- 输出每帧状态，而不是只写 timestamp=0。

一个保守更新策略：

```text
matched dynamic track:
  center = alpha * predicted_center + (1 - alpha) * observed_center
  velocity = beta * old_velocity + (1 - beta) * observed_velocity
  size/yaw = short-window robust average

matched static track:
  use current BoundingBox3DTracker confidence-weighted fusion

unmatched dynamic track:
  predict by velocity
  if visible and missed too long -> lost/removed

unmatched static track:
  use existing visibility-aware aging
```

## 10. 最小实现路径

### Phase 1：不改网络，只接入新系统层

1. 新增 `utils/dynamic_instance_map.py`。
2. 在 `run_boxer.py` 当前 `tracker.update(...)` 附近并行或替换为 `dynamic_map.update(...)`。
3. 把 `obb_pr_w` 转成 `BoxObservation` 列表，带上 `time_ns0`、`scores2d`、`scores3d`、`labels3d`、`bb2d`。
4. 先复用 `BoundingBox3DTracker` 的 IoU + Hungarian，输出结果保持和现有 tracker 类似。

### Phase 2：加 motion-aware association

1. `InstanceRecord` 中加入 `velocity_world`。
2. 用 `timestamp_ns` 计算 `dt`。
3. 用 `predicted_center = last_center + velocity * dt` 做 center gating。
4. cost 中加入 center distance 和 semantic cost。

### Phase 3：分离静态/动态更新策略

1. 增加 movable class prior。
2. 按速度和类别维护 `motion_state`。
3. static track 继续加权融合。
4. dynamic track 维护 latest/EMA + history。

### Phase 4：输出动态实例地图

不要复用 `boxer_3dbbs_fused.csv` 表示动态地图。建议新增：

```text
dynamic_instances.jsonl
dynamic_instances.csv
```

字段建议：

```text
time_ns, frame_idx, instance_id, name, motion_state,
tx, ty, tz, qw, qx, qy, qz, sx, sy, sz,
vx, vy, vz, prob, state
```

如果要兼容现有 viewer，可以额外输出当前时刻的 `ObbTW` CSV，但动态地图主结果应保留 timestamp。

## 11. 测试清单

现有测试可参考：

- `tests/test_track_3d_boxes.py`
- `tests/test_fusion.py`
- `tests/test_boxernet.py`

建议新增：

```text
tests/test_dynamic_instance_map.py
```

核心测试：

1. 静止实例连续观测后稳定为同一 id。
2. 匀速运动实例不会被 world-coordinate average 拉回中间。
3. 两个动态实例交叉经过时不互换 id。
4. 动态实例短时遮挡后恢复匹配。
5. 静态实例和动态实例重叠时，semantic + motion gating 不误合并。
6. 低置信度单帧 false positive 不进入 active map。
7. `timestamp_ns` 不均匀时速度估计仍合理。

## 12. 易踩坑

- `ObbTW.inst_id` 当前在 BoxerNet head 中只是 query index，`run_boxer.py` 之后没有真正表示全局实例 id。动态地图必须重新分配稳定 `instance_id`。
- `--fuse` 是静态融合，不适合动态对象。
- `tracker._get_active_tracks()` 会返回 `ACTIVE/INACTIVE`，不会返回 `TENTATIVE`。
- `sdp_w` 可能含 NaN padding；进入几何判断前要注意过滤。
- 2D box 格式在 CSV 和 Boxer 内部不同，改预处理时要反复确认。
- 现有 `BoundingBox3DTracker._update_track()` 会融合 translation/size/yaw，这对动态对象不是理想行为。
- `ObbCsvWriter2` 当前输出字段不足以表达速度、生命周期和动态状态。

## 13. 推荐优先阅读的文件顺序

```text
run_boxer.py
loaders/base_loader.py
boxernet/boxernet.py
owl/owl_wrapper.py
utils/tw/obb.py
utils/track_3d_boxes.py
utils/fuse_3d_boxes.py
utils/file_io.py
tests/test_track_3d_boxes.py
```

## 14. 推荐的第一版接口草案

```python
class DynamicInstanceMap:
    def __init__(
        self,
        iou_threshold: float = 0.2,
        center_gate_m: float = 1.0,
        dynamic_center_gate_m: float = 2.0,
        semantic_threshold: float = 0.6,
        min_hits: int = 3,
        max_missed: int = 30,
    ) -> None:
        ...

    def update(
        self,
        obbs: ObbTW,
        timestamp_ns: int,
        frame_idx: int,
        labels: list[str],
        scores2d: torch.Tensor | None = None,
        scores3d: torch.Tensor | None = None,
        bb2d: torch.Tensor | None = None,
        cam: CameraTW | None = None,
        T_world_rig: PoseTW | None = None,
        observed_points: torch.Tensor | None = None,
    ) -> list[InstanceRecord]:
        ...
```

这样 `run_boxer.py` 的接入点会非常直接：在 `obb_pr_w`、`labels3d`、`scores2d`、`scores3d` 都已准备好的位置调用 `dynamic_map.update(...)`。

