# boxer 测试
本地已经跑通了对自采数据的boxer，但希望对该自采数据，有一个 incremental 运行的demo视频

## 自采数据说明
是gravity方向align完成的 z-up点云，
以及 带位姿的rgb图像
- 整体路径: /home/wjxu22/Datasets/outputs/rtab/oak_stereo_imu_gravity_lossless_export/
- 点云文件: /home/wjxu22/Datasets/outputs/rtab/oak_stereo_imu_gravity_lossless_export/pointcloud/rtabmap_cloud_map_latest.ply
- 使用参考: /home/wjxu22/SLAMBot/rtab_ws/src/tools/view_oak_rtabmap_processed_export.py

## 自采数据跑通结果
- 路径 output/oak_stereo_imu_gravity_lossless_export
- 可视化 view_rtab.py

## Notes
- 环境: conda环境 jarvis

