
import numpy as np
import numpy.linalg as la
import open3d as o3d
import matplotlib.pyplot as plt
from scipy import optimize
from .geo import Circle3D, Torus

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

def torus_fit(points, weights=None, initial_guess: Torus = None) -> Torus:
    """Fits a torus trough a set of points"""
    _check_input(points, weights)
    if initial_guess is None:
        raise NotImplementedError(
            "Toru fit currently does support running without an intial guess."
        )

    def torus_fit_residuals(circle_params, points, weights):
        circle = Circle3D(
            circle_params[:3], circle_params[3:], la.norm(circle_params[3:])
        )
        distances = circle.distance_to_point(points)
        radius = np.average(distances, weights=weights)
        weights = np.sqrt(weights) if weights is not None else 1.0
        return (distances - radius) * weights

    x0 = np.concatenate(
        [initial_guess.center, initial_guess.direction * initial_guess.major_radius]
    )

    results = optimize.least_squares(
        torus_fit_residuals,
        x0=x0,
        args=(points, weights),
    )

    if not results.success:
        return Torus([-1000,-1000,-1000], [-1000,-1000,-1000], 0, 0)
        raise RuntimeError(results.message)

    circle3D = Circle3D(results.x[:3], results.x[3:], la.norm(results.x[3:]))
    distances = circle3D.distance_to_point(points)
    minor_radius = np.average(distances, weights=weights)
    return Torus(
        results.x[:3], results.x[3:], la.norm(results.x[3:]), minor_radius
    )

def Torusfitter(path):
  with open(path, 'r') as f:
    verts = read_pc(f)
  pointcloud = np.array(verts)
  initial_guess = Torus([0, 0, 0], [0, 0, 1], 1, 0.1)
  torus = torus_fit(pointcloud, initial_guess=initial_guess)
  return torus