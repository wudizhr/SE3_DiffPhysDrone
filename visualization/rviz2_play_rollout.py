#!/usr/bin/env python3
"""Play a saved DiffPhysDrone rollout in RViz2 without torch or CUDA imports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List


RVIZ2_DIR = Path(__file__).resolve().parent
if str(RVIZ2_DIR) not in sys.path:
    sys.path.insert(0, str(RVIZ2_DIR))

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray

from rviz2_common import (
    color_rgba,
    load_rollout_npz,
    load_yaml,
    make_float_image,
    make_line_strip_marker,
    make_marker,
    make_mono8_image,
    make_pointcloud2,
    quaternion_from_matrix,
    resolve_path,
    scene_to_marker_list,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to offline RViz2 playback YAML config")
    return parser.parse_args()


def require_rollout_path(config: Dict[str, Any], config_path: Path) -> Path:
    if "rollout_path" not in config:
        raise KeyError(f"Missing required YAML key 'rollout_path' in {config_path}")
    path = resolve_path(config["rollout_path"], config_path.parent)
    if not path.exists():
        raise FileNotFoundError(f"Configured rollout_path does not exist: {path}")
    return path


def merged_playback_config(config: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for nested_key in ("rollout", "playback"):
        nested_config = config.get(nested_key, {})
        if nested_config is None:
            nested_config = {}
        if not isinstance(nested_config, dict):
            raise ValueError(f"Config key '{nested_key}' must be a mapping when present")
        merged.update(nested_config)
    for key, value in config.items():
        if key not in ("rollout", "playback"):
            merged[key] = value
    return merged


def build_playback_config(
    config: Dict[str, Any],
    *,
    rollout_rate_hz: float,
    rollout_frame_id: str,
) -> Dict[str, Any]:
    merged = merged_playback_config(config)
    if "ctl_freq" in merged:
        rate_hz = float(merged["ctl_freq"])
    elif "rate_hz" in merged:
        rate_hz = float(merged["rate_hz"])
    elif "ctl_dt" in merged:
        ctl_dt = float(merged["ctl_dt"])
        if ctl_dt <= 0:
            raise ValueError(f"ctl_dt must be positive, got {ctl_dt}")
        rate_hz = 1.0 / ctl_dt
    else:
        rate_hz = float(rollout_rate_hz)
    playback_speed = float(merged.get("playback_speed", 1.0))
    if rate_hz <= 0:
        raise ValueError(f"rate_hz must be positive, got {rate_hz}")
    if playback_speed <= 0:
        raise ValueError(f"playback_speed must be positive, got {playback_speed}")
    return {
        "frame_id": str(merged.get("frame_id", rollout_frame_id)),
        "publish_prefix": str(merged.get("publish_prefix", "/diffphys")).rstrip("/"),
        "axis_length": float(merged.get("body_axis_length", 0.6)),
        "axis_radius": float(merged.get("body_axis_radius", 0.03)),
        "rate_hz": rate_hz,
        "playback_speed": playback_speed,
        "timer_period": 1.0 / (rate_hz * playback_speed),
        "loop": bool(merged.get("loop", False)),
        "exit_on_finish": bool(merged.get("exit_on_finish", True)),
    }


def make_odometry(position: np.ndarray, rotation: np.ndarray, *, frame_id: str, stamp, child_frame_id: str) -> Odometry:
    quat = quaternion_from_matrix(rotation)
    msg = Odometry()
    msg.header.frame_id = str(frame_id)
    msg.header.stamp = stamp
    msg.child_frame_id = str(child_frame_id)
    msg.pose.pose.position.x = float(position[0])
    msg.pose.pose.position.y = float(position[1])
    msg.pose.pose.position.z = float(position[2])
    msg.pose.pose.orientation.x = float(quat[0])
    msg.pose.pose.orientation.y = float(quat[1])
    msg.pose.pose.orientation.z = float(quat[2])
    msg.pose.pose.orientation.w = float(quat[3])
    return msg


class DiffPhysRolloutPlayer(Node):
    def __init__(self, config: Dict[str, Any], config_path: Path):
        super().__init__("diffphys_rviz2_rollout_player")
        rollout_path = require_rollout_path(config, config_path)
        self.rollout = load_rollout_npz(rollout_path)
        playback_config = build_playback_config(
            config,
            rollout_rate_hz=float(self.rollout["rate_hz"]),
            rollout_frame_id=str(self.rollout["frame_id"]),
        )
        self.frame_id = playback_config["frame_id"]
        self.publish_prefix = playback_config["publish_prefix"]
        self.axis_length = playback_config["axis_length"]
        self.axis_radius = playback_config["axis_radius"]
        self.rate_hz = playback_config["rate_hz"]
        self.playback_speed = playback_config["playback_speed"]
        self.loop = playback_config["loop"]
        self.exit_on_finish = playback_config["exit_on_finish"]
        self.step_idx = 0
        self.trajectory: List[np.ndarray] = []

        self.positions = np.asarray(self.rollout["positions"], dtype=np.float32)
        self.rotations = np.asarray(self.rollout["rotations"], dtype=np.float32)
        self.raw_depth = np.asarray(self.rollout["raw_depth"], dtype=np.float32)
        self.pooled_depth = np.asarray(self.rollout["pooled_depth"], dtype=np.float32)
        self.pooled_raw_depth = np.asarray(self.rollout["pooled_raw_depth"], dtype=np.float32)
        self.mid360_points = (
            np.asarray(self.rollout["mid360_points"], dtype=np.float32) if "mid360_points" in self.rollout else None
        )
        self.mid360_ranges = (
            np.asarray(self.rollout["mid360_ranges"], dtype=np.float32) if "mid360_ranges" in self.rollout else None
        )
        self.mid360_pseudo_image = (
            np.asarray(self.rollout["mid360_pseudo_image"], dtype=np.float32)
            if "mid360_pseudo_image" in self.rollout
            else None
        )
        self.mid360_pseudo_image_max_range = float(
            self.rollout.get("mid360_pseudo_image_max_range", self.rollout.get("mid360_max_range", 70.0))
        )
        self.target = np.asarray(self.rollout["target"], dtype=np.float32)

        self.odom_pub = self.create_publisher(Odometry, f"{self.publish_prefix}/odom", 1)
        self.mid360_pub = self.create_publisher(PointCloud2, f"{self.publish_prefix}/mid360_points", 1)
        self.mid360_pseudo_image_pub = self.create_publisher(Image, f"{self.publish_prefix}/mid360_pseudo_image", 1)
        self.mid360_pseudo_image_viz_pub = self.create_publisher(Image, f"{self.publish_prefix}/mid360_pseudo_image_viz", 1)
        self.markers_pub = self.create_publisher(MarkerArray, f"{self.publish_prefix}/markers", 1)
        self.raw_depth_pub = self.create_publisher(Image, f"{self.publish_prefix}/raw_depth", 1)
        self.pooled_depth_pub = self.create_publisher(Image, f"{self.publish_prefix}/pooled_depth", 1)
        self.pooled_raw_depth_pub = self.create_publisher(Image, f"{self.publish_prefix}/pooled_raw_depth", 1)
        self.raw_depth_viz_pub = self.create_publisher(Image, f"{self.publish_prefix}/raw_depth_viz", 1)
        self.pooled_depth_viz_pub = self.create_publisher(Image, f"{self.publish_prefix}/pooled_depth_viz", 1)
        self.pooled_raw_depth_viz_pub = self.create_publisher(Image, f"{self.publish_prefix}/pooled_raw_depth_viz", 1)
        self.timer = self.create_timer(playback_config["timer_period"], self.publish_step)
        self.get_logger().info(
            f"Loaded rollout {rollout_path}; playback={self.rate_hz:g} Hz x {self.playback_speed:g}"
        )

    def publish_step(self) -> None:
        if self.step_idx >= len(self.positions):
            if not self.loop:
                if self.exit_on_finish and rclpy.ok():
                    self.get_logger().info("Rollout finished; shutting down.")
                    rclpy.shutdown()
                return
            self.step_idx = 0
            self.trajectory.clear()

        stamp = self.get_clock().now().to_msg()
        position = self.positions[self.step_idx]
        rotation = self.rotations[self.step_idx]
        raw_depth = self.raw_depth[self.step_idx]
        pooled_depth = self.pooled_depth[self.step_idx]
        pooled_raw_depth = self.pooled_raw_depth[self.step_idx]
        self.trajectory.append(position.copy())

        markers = self.build_markers(position, rotation, stamp)
        self.odom_pub.publish(
            make_odometry(position, rotation, frame_id=self.frame_id, stamp=stamp, child_frame_id="base_link")
        )
        if self.mid360_points is not None:
            mid360_ranges = self.mid360_ranges[self.step_idx] if self.mid360_ranges is not None else None
            self.mid360_pub.publish(
                make_pointcloud2(
                    self.mid360_points[self.step_idx],
                    mid360_ranges,
                    frame_id=self.frame_id,
                    stamp=stamp,
                )
            )
        if self.mid360_pseudo_image is not None:
            mid360_pseudo_image = self.mid360_pseudo_image[self.step_idx]
            self.mid360_pseudo_image_pub.publish(
                make_float_image(mid360_pseudo_image, frame_id=self.frame_id, stamp=stamp)
            )
            self.mid360_pseudo_image_viz_pub.publish(
                make_mono8_image(mid360_pseudo_image, frame_id=self.frame_id, stamp=stamp, max_value=self.mid360_pseudo_image_max_range)
            )
        self.markers_pub.publish(MarkerArray(markers=markers))
        self.raw_depth_pub.publish(make_float_image(raw_depth, frame_id=self.frame_id, stamp=stamp))
        self.pooled_depth_pub.publish(make_float_image(pooled_depth, frame_id=self.frame_id, stamp=stamp))
        self.pooled_raw_depth_pub.publish(make_float_image(pooled_raw_depth, frame_id=self.frame_id, stamp=stamp))
        self.raw_depth_viz_pub.publish(make_mono8_image(raw_depth, frame_id=self.frame_id, stamp=stamp, max_value=24.0))
        self.pooled_depth_viz_pub.publish(
            make_mono8_image(pooled_depth, frame_id=self.frame_id, stamp=stamp, min_value=-0.6, max_value=9.4)
        )
        self.pooled_raw_depth_viz_pub.publish(
            make_mono8_image(pooled_raw_depth, frame_id=self.frame_id, stamp=stamp, max_value=24.0)
        )
        self.step_idx += 1

    def build_markers(self, position: np.ndarray, rotation: np.ndarray, stamp) -> List[Marker]:
        scene = {
            "balls": self.rollout["balls"],
            "voxels": self.rollout["voxels"],
            "cyl": self.rollout["cyl"],
            "cyl_h": self.rollout["cyl_h"],
        }
        markers = scene_to_marker_list(scene, frame_id=self.frame_id, stamp=stamp)
        base_id = 100000
        markers.append(
            make_marker(
                marker_id=base_id + 10,
                namespace="target",
                marker_type=Marker.SPHERE,
                frame_id=self.frame_id,
                stamp=stamp,
                position=self.target,
                scale=(0.35, 0.35, 0.35),
                color=color_rgba(0.0, 1.0, 0.2, 0.9),
            )
        )
        if len(self.trajectory) >= 2:
            markers.append(
                make_line_strip_marker(
                    self.trajectory,
                    marker_id=base_id + 20,
                    namespace="trajectory",
                    frame_id=self.frame_id,
                    stamp=stamp,
                    width=0.05,
                    color=color_rgba(1.0, 1.0, 0.0, 1.0),
                )
            )
        return markers


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)
    rclpy.init()
    node = DiffPhysRolloutPlayer(config, config_path)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
