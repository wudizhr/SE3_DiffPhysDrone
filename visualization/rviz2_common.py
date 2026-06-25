"""Helpers for offline RViz2 rollout playback."""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

VIS_DIR = Path(__file__).resolve().parent
if str(VIS_DIR) not in sys.path:
    sys.path.insert(0, str(VIS_DIR))

from visualization_common import ROLLOUT_REQUIRED_KEYS, load_yaml, resolve_path


def load_rollout_npz(path: str | Path) -> Dict[str, Any]:
    with np.load(Path(path).expanduser(), allow_pickle=False) as data:
        missing = [key for key in ROLLOUT_REQUIRED_KEYS if key not in data]
        if missing:
            raise ValueError(f"Rollout NPZ missing required keys: {missing}")
        return {key: data[key] for key in data.files}


def color_rgba(r: float, g: float, b: float, a: float = 1.0):
    from std_msgs.msg import ColorRGBA

    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))


def make_header(frame_id: str, stamp):
    from std_msgs.msg import Header

    return Header(frame_id=str(frame_id), stamp=stamp)


def make_float_image(image: np.ndarray, *, frame_id: str, stamp):
    from sensor_msgs.msg import Image

    arr = np.asarray(image, dtype=np.float32)
    msg = Image()
    msg.header = make_header(frame_id, stamp)
    msg.height = int(arr.shape[0])
    msg.width = int(arr.shape[1])
    msg.encoding = "32FC1"
    msg.is_bigendian = 0
    msg.step = int(arr.shape[1] * arr.dtype.itemsize)
    msg.data = arr.tobytes()
    return msg


def make_mono8_image(
    image: np.ndarray,
    *,
    frame_id: str,
    stamp,
    min_value: float = 0.0,
    max_value: float = 24.0,
):
    from sensor_msgs.msg import Image

    arr = np.asarray(image, dtype=np.float32)
    denom = max(float(max_value) - float(min_value), 1e-6)
    normalized = np.clip((arr - float(min_value)) / denom, 0.0, 1.0)
    mono = (normalized * 255).astype(np.uint8)

    msg = Image()
    msg.header = make_header(frame_id, stamp)
    msg.height = int(mono.shape[0])
    msg.width = int(mono.shape[1])
    msg.encoding = "mono8"
    msg.is_bigendian = 0
    msg.step = int(mono.shape[1])
    msg.data = mono.tobytes()
    return msg


def make_pointcloud2(points: np.ndarray, ranges: np.ndarray | None = None, *, frame_id: str, stamp):
    from sensor_msgs.msg import PointCloud2, PointField

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if ranges is not None:
        valid = np.asarray(ranges, dtype=np.float32).reshape(-1) > 0.0
        pts = pts[valid]
    pts = np.ascontiguousarray(pts, dtype=np.float32)

    msg = PointCloud2()
    msg.header = make_header(frame_id, stamp)
    msg.height = 1
    msg.width = int(pts.shape[0])
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = struct.pack("=I", 1) == struct.pack(">I", 1)
    msg.point_step = 12
    msg.row_step = int(msg.point_step * msg.width)
    msg.is_dense = True
    msg.data = pts.tobytes()
    return msg


def quaternion_from_matrix(rotation: np.ndarray) -> tuple[float, float, float, float]:
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(r)))
        if idx == 0:
            s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            qw = (r[2, 1] - r[1, 2]) / s
            qx = 0.25 * s
            qy = (r[0, 1] + r[1, 0]) / s
            qz = (r[0, 2] + r[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            qw = (r[0, 2] - r[2, 0]) / s
            qx = (r[0, 1] + r[1, 0]) / s
            qy = 0.25 * s
            qz = (r[1, 2] + r[2, 1]) / s
        else:
            s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            qw = (r[1, 0] - r[0, 1]) / s
            qx = (r[0, 2] + r[2, 0]) / s
            qy = (r[1, 2] + r[2, 1]) / s
            qz = 0.25 * s
    return float(qx), float(qy), float(qz), float(qw)


def make_marker(
    *,
    marker_id: int,
    namespace: str,
    marker_type: int,
    frame_id: str,
    stamp,
    position: Sequence[float],
    scale: Sequence[float],
    color,
    orientation: Sequence[float] | None = None,
):
    from visualization_msgs.msg import Marker

    marker = Marker()
    marker.header = make_header(frame_id, stamp)
    marker.ns = str(namespace)
    marker.id = int(marker_id)
    marker.type = int(marker_type)
    marker.action = Marker.ADD
    marker.pose.position.x = float(position[0])
    marker.pose.position.y = float(position[1])
    marker.pose.position.z = float(position[2])
    quat = orientation or (0.0, 0.0, 0.0, 1.0)
    marker.pose.orientation.x = float(quat[0])
    marker.pose.orientation.y = float(quat[1])
    marker.pose.orientation.z = float(quat[2])
    marker.pose.orientation.w = float(quat[3])
    marker.scale.x = float(scale[0])
    marker.scale.y = float(scale[1])
    marker.scale.z = float(scale[2])
    marker.color = color
    return marker


def make_line_strip_marker(
    points: Iterable[Sequence[float]],
    *,
    marker_id: int,
    namespace: str,
    frame_id: str,
    stamp,
    width: float,
    color,
):
    from geometry_msgs.msg import Point
    from visualization_msgs.msg import Marker

    marker = make_marker(
        marker_id=marker_id,
        namespace=namespace,
        marker_type=Marker.LINE_STRIP,
        frame_id=frame_id,
        stamp=stamp,
        position=(0.0, 0.0, 0.0),
        scale=(width, 0.0, 0.0),
        color=color,
    )
    marker.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in points]
    return marker


def make_body_axis_markers(
    position: Sequence[float],
    rotation: np.ndarray,
    *,
    frame_id: str,
    stamp,
    namespace: str,
    start_id: int,
    axis_length: float,
    axis_radius: float,
):
    from visualization_msgs.msg import Marker

    pos = np.asarray(position, dtype=np.float32)
    rot = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    axes = (
        ("forward", rot[:, 0], color_rgba(1.0, 0.0, 0.0, 1.0)),
        ("left", rot[:, 1], color_rgba(0.0, 1.0, 0.0, 1.0)),
        ("up", rot[:, 2], color_rgba(0.0, 0.2, 1.0, 1.0)),
    )
    markers = []
    for offset, (name, direction, color) in enumerate(axes):
        center = pos + direction * (axis_length * 0.5)
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
        v = np.cross(x_axis, direction)
        c = float(np.dot(x_axis, direction))
        if np.linalg.norm(v) < 1e-6:
            axis_rotation = np.eye(3, dtype=np.float32)
            if c < 0:
                axis_rotation[0, 0] = -1.0
                axis_rotation[1, 1] = -1.0
        else:
            vx = np.array(
                [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],
                dtype=np.float32,
            )
            axis_rotation = np.eye(3, dtype=np.float32) + vx + vx @ vx * (1.0 / (1.0 + c))
        markers.append(
            make_marker(
                marker_id=start_id + offset,
                namespace=f"{namespace}_{name}",
                marker_type=Marker.CYLINDER,
                frame_id=frame_id,
                stamp=stamp,
                position=center,
                scale=(axis_length, axis_radius, axis_radius),
                color=color,
                orientation=quaternion_from_matrix(axis_rotation),
            )
        )
    return markers


def scene_to_marker_list(scene: Mapping[str, Any], *, frame_id: str, stamp):
    from visualization_msgs.msg import Marker

    markers = []
    marker_id = 0
    for ball in np.asarray(scene.get("balls", []), dtype=np.float32).reshape(-1, 4):
        if ball[0] < -999:
            continue
        r = float(ball[3])
        markers.append(
            make_marker(
                marker_id=marker_id,
                namespace="balls",
                marker_type=Marker.SPHERE,
                frame_id=frame_id,
                stamp=stamp,
                position=ball[:3],
                scale=(2 * r, 2 * r, 2 * r),
                color=color_rgba(0.9, 0.25, 0.2, 0.65),
            )
        )
        marker_id += 1

    for voxel in np.asarray(scene.get("voxels", []), dtype=np.float32).reshape(-1, 6):
        if voxel[0] < -999:
            continue
        markers.append(
            make_marker(
                marker_id=marker_id,
                namespace="voxels",
                marker_type=Marker.CUBE,
                frame_id=frame_id,
                stamp=stamp,
                position=voxel[:3],
                scale=2 * voxel[3:6],
                color=color_rgba(0.25, 0.45, 0.95, 0.45),
            )
        )
        marker_id += 1

    for cyl in np.asarray(scene.get("cyl", []), dtype=np.float32).reshape(-1, 3):
        if cyl[0] < -999:
            continue
        r = float(cyl[2])
        markers.append(
            make_marker(
                marker_id=marker_id,
                namespace="cyl",
                marker_type=Marker.CYLINDER,
                frame_id=frame_id,
                stamp=stamp,
                position=(float(cyl[0]), float(cyl[1]), 1.0),
                scale=(2 * r, 2 * r, 4.0),
                color=color_rgba(0.95, 0.65, 0.1, 0.55),
            )
        )
        marker_id += 1

    for cyl_h in np.asarray(scene.get("cyl_h", []), dtype=np.float32).reshape(-1, 3):
        if cyl_h[0] < -999:
            continue
        r = float(cyl_h[2])
        markers.append(
            make_marker(
                marker_id=marker_id,
                namespace="cyl_h",
                marker_type=Marker.CYLINDER,
                frame_id=frame_id,
                stamp=stamp,
                position=(float(cyl_h[0]), 0.0, float(cyl_h[1])),
                scale=(2 * r, 2 * r, 18.0),
                color=color_rgba(0.6, 0.2, 0.9, 0.5),
                orientation=quaternion_from_matrix(
                    np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
                ),
            )
        )
        marker_id += 1
    return markers
