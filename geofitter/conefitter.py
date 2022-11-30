import random
import numpy as np
import numpy.linalg as la
import open3d as o3d
import matplotlib.pyplot as plt
from scipy import optimize

from .geo import Circle3D

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

def circle3D_fit(points, weights=None, initial_guess: Circle3D = None):
    """Fits a circle in three dimensions trough a set of points"""
    _check_input(points, weights)
    if initial_guess is None:
        raise NotImplementedError(
            "Circle3D fit currently does support running without an intial guess."
        )

    def circle_fit_residuals(circle_params, points, sqrt_w):
        circle = Circle3D(
            circle_params[:3], circle_params[3:], la.norm(circle_params[3:])
        )
        distances = circle.distance_to_point(points)
        return distances * sqrt_w

    x0 = np.concatenate(
        [initial_guess.center, initial_guess.direction * initial_guess.radius]
    )
    
    results = optimize.least_squares(
        circle_fit_residuals,
        jac="2-point",
        method="lm",
        x0=x0,
        args=(points, 1.0 if weights is None else np.sqrt(weights)),
    )
    

    results = optimize.minimize(
        lambda *args: np.sum(circle_fit_residuals(*args) ** 2),
        x0=results.x,
        args=(points, 1.0 if weights is None else np.sqrt(weights)),
    )

    if not results.success:
        return RuntimeError(results.message)

    return Circle3D(results.x[:3], results.x[3:], la.norm(results.x[3:]))

def plane_intersect(a, b):
    a_vec, b_vec = np.array(a[:3]), np.array(b[:3])

    aXb_vec = np.cross(a_vec, b_vec)

    A = np.array([a_vec, b_vec, aXb_vec])
    d = np.array([-a[3], -b[3], 0.]).reshape(3,1)
    p_inter = np.linalg.solve(A, d).T

    return p_inter[0], (p_inter + aXb_vec)[0]

def distance(x1, y1, z1, x2, y2, z2):
  d = 0.0
  d =(x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2
  return d

def cmp(t):
  return t[0]

def find_Denta(point):
  point.sort(key=lambda t: t[0])
  d=abs(point[-1][0]-point[0][0])
  point.sort(key=lambda t: t[1])
  d1=abs(point[-1][1]-point[0][1])
  if d1>d: d=d1
  point.sort(key=lambda t: t[2])
  d2=abs(point[-1][2]-point[0][2])
  if d2>d: d=d2
  return d/len(point)

def create_vector(point1,point2):
  return np.array([point2[i]-point1[i] for i in range(3)])

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def maximum_con(kt):
    res,max,pos=0,0,0
    for i in range(len(kt)):
      if (kt[i]==1): res+=1
      else: 
        if (res>max): 
          max=res
          pos=i
        res=0
    if (res>max): 
      max=res
      pos=len(kt)-1
    return [max,pos]

def Conefitter(path):
  pcd = o3d.io.read_point_cloud(path)
  print(pcd)
  size = (len(np.asarray(pcd.points)))
  print("Recompute the normal of the downsampled point cloud")
  pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2,max_nn=30))

  points = []
  for p in range(0,10000):
    i = random.randint(0, size-1)
    j = random.randint(0, size-1)
    k = random.randint(0, size-1)
    di = - (pcd.points[i][0] * pcd.normals[i][0] + pcd.points[i][1] * pcd.normals[i][1] + pcd.points[i][2] * pcd.normals[i][2])
    dj = - (pcd.points[j][0] * pcd.normals[j][0] + pcd.points[j][1] * pcd.normals[j][1] + pcd.points[j][2] * pcd.normals[j][2])
    dk = - (pcd.points[k][0] * pcd.normals[k][0] + pcd.points[k][1] * pcd.normals[k][1] + pcd.points[k][2] * pcd.normals[k][2])
    a = (pcd.points[i][0], pcd.points[i][1], pcd.points[i][2], di)
    b = (pcd.points[j][0], pcd.points[j][1], pcd.points[j][2], dj)
    try:
      c, d = plane_intersect(a,b)
    except:
        continue
    t =  -(c[0] + c[1] + c[2]) /((c[0]- d[0])* pcd.normals[k][0] + (c[1] - d[1]) * pcd.normals[k][1] + (c[2] - d[2])* pcd.normals[k][2])
    points.append([c[0] + (c[0] - d[0])* t, c[1] + (c[1] - d[1]) * t, c[2] + (c[2] - d[2]) * t])
  x_mean = 0
  y_mean = 0
  z_mean = 0
  for i in points:
    x_mean += i[0]
    y_mean += i[1]
    z_mean += i[2]
  x_mean = x_mean / 100000
  y_mean = y_mean / 100000
  z_mean = z_mean / 100000

  dist=[]
  for i in pcd.points:
    dist.append([distance(x_mean,y_mean,z_mean,i[0],i[1],[2]), i])

  dist.sort(key=cmp,reverse=True)
  exp=find_Denta(list(pcd.points))
  points_plane=[]
  kt=[0 for i in dist]

  for i in range(len(dist)-1):
    if abs(dist[i][0]-dist[i+1][0])<=exp: 
      kt[i]=1

  max,pos=maximum_con(kt)[0],maximum_con(kt)[1]
  for i in range(max):
    points_plane.append(dist[pos])
    pos-=1
  points_of_plane=[i[1] for i in points_plane]
  initial_guess = Circle3D([0, 0, 0], [0, 0, 1], 1)
  try:
    circle = circle3D_fit(points_of_plane, initial_guess=initial_guess)
  except:
    return (-1000,-1000,-1000,-1000,-1000,-1000,-1000)
  return angle_between(create_vector([x_mean,y_mean,z_mean],points_of_plane[0]),(circle.direction[0],circle.direction[1],circle.direction[2])), circle.direction[0],circle.direction[1],circle.direction[2], x_mean,y_mean,z_mean 