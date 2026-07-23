"""Minimal COLMAP text-model reader/writer.

Only the text format is handled (cameras.txt / images.txt / points3D.txt),
which is exactly what SplatKing emits for the LiDAR path and what LichtFeld can
ingest as a dataset. Writing supports a multi-camera model with per-image
camera assignment - the piece the GUI plugin cannot express.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from .intrinsics import ColmapCamera

COLMAP_HEADER = (
    "# Created by splatking_importer (SplatKing -> LichtFeld)\n"
)


@dataclass
class ColmapImage:
    image_id: int
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    camera_id: int
    name: str
    points2d: list[tuple[float, float, int]] = field(default_factory=list)

    def pose_line(self) -> str:
        return (
            f"{self.image_id} {self.qw:.9g} {self.qx:.9g} {self.qy:.9g} {self.qz:.9g} "
            f"{self.tx:.9g} {self.ty:.9g} {self.tz:.9g} {self.camera_id} {self.name}"
        )

    def points_line(self) -> str:
        if not self.points2d:
            return ""
        return " ".join(f"{x:.6g} {y:.6g} {pid}" for x, y, pid in self.points2d)


@dataclass
class ColmapPoint3D:
    point_id: int
    x: float
    y: float
    z: float
    r: int
    g: int
    b: int
    error: float = -1.0
    track: list[tuple[int, int]] = field(default_factory=list)

    def to_line(self) -> str:
        track_str = " ".join(f"{img} {p2d}" for img, p2d in self.track)
        base = (
            f"{self.point_id} {self.x:.9g} {self.y:.9g} {self.z:.9g} "
            f"{self.r} {self.g} {self.b} {self.error:.6g}"
        )
        return f"{base} {track_str}".rstrip()


@dataclass
class ColmapModel:
    cameras: list[ColmapCamera] = field(default_factory=list)
    images: list[ColmapImage] = field(default_factory=list)
    points3d: list[ColmapPoint3D] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def camera_by_id(self, cid: int) -> Optional[ColmapCamera]:
        for c in self.cameras:
            if c.camera_id == cid:
                return c
        return None

    def write(self, sparse_dir: str) -> None:
        os.makedirs(sparse_dir, exist_ok=True)
        self._write_cameras(os.path.join(sparse_dir, "cameras.txt"))
        self._write_images(os.path.join(sparse_dir, "images.txt"))
        self._write_points3d(os.path.join(sparse_dir, "points3D.txt"))

    def _write_cameras(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(COLMAP_HEADER)
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"# Number of cameras: {len(self.cameras)}\n")
            for c in self.cameras:
                f.write(c.to_line() + "\n")

    def _write_images(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(COLMAP_HEADER)
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            f.write(f"# Number of images: {len(self.images)}\n")
            for im in self.images:
                f.write(im.pose_line() + "\n")
                f.write(im.points_line() + "\n")

    def _write_points3d(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(COLMAP_HEADER)
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            f.write(f"# Number of points: {len(self.points3d)}\n")
            for p in self.points3d:
                f.write(p.to_line() + "\n")


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def _iter_data_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield line


def read_model(sparse_dir: str) -> ColmapModel:
    model = ColmapModel()

    cam_path = os.path.join(sparse_dir, "cameras.txt")
    if os.path.isfile(cam_path):
        for line in _iter_data_lines(cam_path):
            parts = line.split()
            model.cameras.append(
                ColmapCamera(
                    camera_id=int(parts[0]),
                    model=parts[1],
                    width=int(parts[2]),
                    height=int(parts[3]),
                    params=[float(x) for x in parts[4:]],
                )
            )

    img_path = os.path.join(sparse_dir, "images.txt")
    if os.path.isfile(img_path):
        pending: Optional[ColmapImage] = None
        for line in _iter_data_lines(img_path):
            parts = line.split()
            # Heuristic: a pose line has a non-numeric NAME as the last token.
            is_pose = len(parts) >= 10 and not _is_number(parts[-1])
            if is_pose:
                if pending is not None:
                    model.images.append(pending)
                pending = ColmapImage(
                    image_id=int(parts[0]),
                    qw=float(parts[1]), qx=float(parts[2]), qy=float(parts[3]), qz=float(parts[4]),
                    tx=float(parts[5]), ty=float(parts[6]), tz=float(parts[7]),
                    camera_id=int(parts[8]),
                    name=" ".join(parts[9:]),
                )
            elif pending is not None:
                nums = [float(x) for x in parts]
                for i in range(0, len(nums) - 2, 3):
                    pending.points2d.append((nums[i], nums[i + 1], int(nums[i + 2])))
        if pending is not None:
            model.images.append(pending)

    pts_path = os.path.join(sparse_dir, "points3D.txt")
    if os.path.isfile(pts_path):
        for line in _iter_data_lines(pts_path):
            parts = line.split()
            track = []
            rest = parts[8:]
            for i in range(0, len(rest) - 1, 2):
                track.append((int(rest[i]), int(rest[i + 1])))
            model.points3d.append(
                ColmapPoint3D(
                    point_id=int(parts[0]),
                    x=float(parts[1]), y=float(parts[2]), z=float(parts[3]),
                    r=int(parts[4]), g=int(parts[5]), b=int(parts[6]),
                    error=float(parts[7]) if len(parts) > 7 else -1.0,
                    track=track,
                )
            )

    return model


def _is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def find_sparse_dir(model_root: str) -> Optional[str]:
    """Locate the sparse/0 directory inside a COLMAP_Text_Model folder."""
    candidates = [
        os.path.join(model_root, "sparse", "0"),
        os.path.join(model_root, "sparse"),
        model_root,
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "cameras.txt")):
            return c
    return None
