import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import random

def read_pc(file):
    #n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(',')])
    #print(n_verts, n_faces)
    verts = [[float(s) for s in line.strip().split(',')] for line in file]
    return verts

def Planefitter(path):
  xs = []
  ys = []
  zs = []

  with open(path, 'r') as f:
    points = read_pc(f)
  for point in points:
    xs.append(point[0])
    ys.append(point[1])
    zs.append(point[2])

  tmp_A = []
  tmp_b = []
  for i in range(len(xs)):
      tmp_A.append([xs[i], ys[i], 1])
      tmp_b.append(zs[i])
  b = np.matrix(tmp_b).T
  A = np.matrix(tmp_A)
  fit = (A.T * A).I * A.T * b
  errors = b - A * fit
  residual = np.linalg.norm(errors)

  const = fit[2]

  vector = []
  fit[2] = -1
  module = math.sqrt(fit[0]**2 + fit[1]**2 + 1)
  fit = fit/ module

  x = random.uniform(-50.55, 50.55)
  y = random.uniform(-50.55, 50.55)
  z = fit[0]*x + fit[1]*y + const

  return fit, x, y, z #vector phap tuyen, x, y, z