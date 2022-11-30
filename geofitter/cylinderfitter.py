import numpy as np
import numpy.linalg as la
import open3d as o3d
import matplotlib.pyplot as plt
from scipy import optimize
from .geo import Line, Cylinder

def read_pc(file):
    #n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(',')])
    #print(n_verts, n_faces)
    verts = [[float(s) for s in line.strip().split(',')] for line in file]
    return verts

def centroid_fit(points, weights=None):
    """Calculates the weighted average of a set of points
    This minimizes the sum of the squared distances between the points
    and the centroid.
    """
    if points.ndim == 1:
        return points
    return np.average(points, axis=0, weights=weights)


def _check_input(points, weights) -> None:
    """Check the input data of the fit functionality"""
    points = np.asarray(points)
    if weights is not None:
        weights = np.asarray(weights)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Input data has the wrong shape, expects points to be of shape ('n', 3), got {points.shape}"
        )
    if weights is not None and (weights.ndim != 1 or len(weights) != len(points)):
        raise ValueError(
            "Shape of weights does not match points, weights should be a 1 dimensional array of len(points)"
        )

def cylinder_fit(points, weights=None, initial_guess: Cylinder = None):
    """Fits a cylinder trough a set of points"""
    _check_input(points, weights)
    if initial_guess is None:
        raise NotImplementedError(
            "Cylinder fit currently does support running without an intial guess."
        )

    def cylinder_fit_residuals(anchor_direction, points, weights):
        line = Line(anchor_direction[:3], anchor_direction[3:])
        distances = line.distance_to_point(points)
        radius = np.average(distances, weights=weights)
        if weights is None:
            return distances - radius
        return (distances - radius) * np.sqrt(weights)

    x0 = np.concatenate([initial_guess.anchor_point, initial_guess.direction])
    results = optimize.least_squares(
        cylinder_fit_residuals, x0=x0, args=(points, weights), ftol=1e-10
    )
    if not results.success:
        return RuntimeError(results.message)

    line = Line(results.x[:3], results.x[3:])
    distances = line.distance_to_point(points)
    radius = np.average(distances, weights=weights)
    return Cylinder(results.x[:3], results.x[3:], radius)


def Cylinderfitter(path):
  with open(path, 'r') as f:
    verts = read_pc(f)
  pointcloud = np.array(verts)
  initial_guess = Cylinder([0, 0, 0], [0, 0, 1], 1)
  cylinder = cylinder_fit(pointcloud, initial_guess=initial_guess)
  return cylinder