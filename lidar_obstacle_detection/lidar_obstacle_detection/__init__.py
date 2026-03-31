"""
LiDAR ingress, spatial filtering, and obstacle detection.

Import the pure filter API from ``lidar_obstacle_detection.base_link_spatial_filter``
(e.g. ``SpatialFilterParams``, ``filter_and_downsample_xyz``) to avoid loading Open3D
when only other submodules are needed. Temporal merge: ``temporal_cloud_accumulator``.
Perception library: ``surface_obstacle_segmentation`` (normals, cosine split, DBSCAN).
ROS obstacle messages: ``lidar_obstacle_detection_msgs``; builders in ``obstacle_ros_msgs``.
"""
