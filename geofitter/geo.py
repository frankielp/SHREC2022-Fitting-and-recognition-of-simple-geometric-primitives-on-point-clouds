
from abc import ABC, abstractmethod

import numpy as np
import functools
import numbers
from typing import List, Union

import open3d as o3d

from .descriptor import Direction, Position, PositiveNumber
from .util import distance_line_point, distance_plane_point, distance_point_point

class GeometricShape(ABC):
    @abstractmethod
    def distance_to_point(self, point):
        """Calculates the smallest distance from a point to the shape"""

    # def project_point(self, point):
    # pass

class Line(GeometricShape):
    anchor_point = Position(3)
    direction = Direction(3)

    def __init__(self, anchor_point, direction):
        self.anchor_point = anchor_point
        self.direction = direction

    def __repr__(self):
        return f"Line(anchor_point={self.anchor_point.tolist()}, direction={self.direction.tolist()})"

    def distance_to_point(self, point):
        return distance_line_point(self.anchor_point, self.direction, point)


class Plane(GeometricShape):
    anchor_point = Position(3)
    normal = Direction(3)

    def __init__(self, anchor_point, normal):
        self.anchor_point = anchor_point
        self.normal = normal

    def __repr__(self):
        return f"Plane(anchor_point={self.anchor_point.tolist()}, normal={self.normal.tolist()})"

    def distance_to_point(self, point):
        return np.abs(distance_plane_point(self.anchor_point, self.normal, point))

class Sphere(GeometricShape):
    center = Position(3)
    radius = PositiveNumber()

    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def __repr__(self):
        return f"Sphere(center={self.center.tolist()}, radius={self.radius})"

    def distance_to_point(self, point):
        return np.abs(distance_point_point(point, self.center) - self.radius)


class Cylinder(Line):
    radius = PositiveNumber()

    def __init__(self, anchor_point, direction, radius):
        super().__init__(anchor_point, direction)
        self.radius = radius

    def __repr__(self):
        return f"Cylinder(anchor_point={self.anchor_point.tolist()}, direction={self.direction.tolist()}, radius={self.radius})"

    def distance_to_point(self, point):
        return np.abs(super().distance_to_point(point) - self.radius)

class Circle3D(GeometricShape):
    center = Position(3)
    direction = Direction(3)
    radius = PositiveNumber()

    def __init__(self, center, direction, radius):
        self.center = center
        self.direction = direction
        self.radius = radius

    def __repr__(self):
        return f"Circle3D(center={self.center.tolist()}, direction={self.direction.tolist()}, radius={self.radius})"

    def distance_to_point(self, point):
        delta_p = point - self.center
        x1 = np.matmul(
            np.expand_dims(np.dot(delta_p, self.direction), axis=-1),
            np.atleast_2d(self.direction),
        )
        x2 = delta_p - x1
        return np.sqrt(
            np.linalg.norm(x1, axis=-1) ** 2
            + (np.linalg.norm(x2, axis=-1) - self.radius) ** 2
        )


class Torus(Circle3D):
    minor_radius = PositiveNumber()

    def __init__(self, center, direction, major_radius, minor_radius):
        super().__init__(center, direction, major_radius)
        self.minor_radius = minor_radius

    def __repr__(self):
        return f"Torus(center={self.center.tolist()}, direction={self.direction.tolist()}, major_radius={self.major_radius}, minor_radius={self.minor_radius})"

    @property
    def major_radius(self):
        return self.radius

    def distance_to_point(self, point):
        return np.abs(super().distance_to_point(point) - self.minor_radius)


def vec2vec_rotation(unit_vec_1, unit_vec_2):
    angle = np.arccos(np.dot(unit_vec_1, unit_vec_2))
    if angle < 1e-8:
        return np.identity(3, dtype=np.float64)

    if angle > (np.pi - 1e-8):
        # WARNING this only works because all geometries are rotationaly invariant
        # minus identity is not a proper rotation matrix
        return -np.identity(3, dtype=np.float64)

    rot_vec = np.cross(unit_vec_1, unit_vec_2)
    rot_vec /= np.linalg.norm(rot_vec)

    return o3d.geometry.get_rotation_matrix_from_axis_angle(angle * rot_vec)


@functools.singledispatch
def to_open3d_geom(geom):
    return geom


@to_open3d_geom.register  # type: ignore[no-redef]
def _(points: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

@to_open3d_geom.register  # type: ignore[no-redef]
def _(geom: Line, length: numbers.Number = 1):
    points = (
        geom.anchor_point
        + np.stack([geom.direction, -geom.direction], axis=0) * length / 2
    )

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector([[0, 1]]),
    )
    return line_set


@to_open3d_geom.register  # type: ignore[no-redef]
def _(geom: Sphere):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=geom.radius)
    mesh.translate(geom.center)

    return o3d.geometry.LineSet.create_from_triangle_mesh(mesh)


@to_open3d_geom.register  # type: ignore[no-redef]
def _(geom: Plane, length: numbers.Number = 1):
    points = np.array([[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0]]) * length / 2

    mesh = o3d.geometry.TetraMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.tetras = o3d.utility.Vector4iVector(np.array([[0, 1, 2, 3]]))

    rotation = vec2vec_rotation([0, 0, 1], geom.normal)
    mesh.rotate(rotation)
    mesh.translate(geom.anchor_point)

    return o3d.geometry.LineSet.create_from_tetra_mesh(mesh)

@to_open3d_geom.register  # type: ignore[no-redef]
def _(geom: Cylinder, length: numbers.Number = 1):
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=geom.radius, height=length)

    mesh.remove_vertices_by_index([0, 1])

    rotation = vec2vec_rotation([0, 0, 1], geom.direction)
    mesh.rotate(rotation)
    mesh.translate(geom.anchor_point)

    return o3d.geometry.LineSet.create_from_triangle_mesh(mesh)


@to_open3d_geom.register  # type: ignore[no-redef]
def _(geom: Circle3D):
    mesh = o3d.geometry.TriangleMesh.create_torus(
        torus_radius=geom.radius, tube_radius=1e-6
    )
    rotation = vec2vec_rotation([0, 0, 1], geom.direction)
    mesh.rotate(rotation)
    mesh.translate(geom.center)

    return o3d.geometry.LineSet.create_from_triangle_mesh(mesh)


@to_open3d_geom.register  # type: ignore[no-redef]
def _(geom: Torus):
    mesh = o3d.geometry.TriangleMesh.create_torus(
        torus_radius=geom.major_radius, tube_radius=geom.minor_radius
    )
    rotation = vec2vec_rotation([0, 0, 1], geom.direction)
    mesh.rotate(rotation)
    mesh.translate(geom.center)

    return o3d.geometry.LineSet.create_from_triangle_mesh(mesh)

def plot(
    geometries_or_points: List[Union[GeometricShape, np.ndarray]],
    display_coordinate_frame: bool = False,
):
    geometries = [to_open3d_geom(g) for g in geometries_or_points]
    if display_coordinate_frame:
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame())
    o3d.visualization.draw_geometries(geometries)