import os
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import math
#	fit a sphere to X,Y, and Z data points
#	returns the radius and center points of
#	the best fit sphere
def read_pc(file):
    #n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(',')])
    #print(n_verts, n_faces)
    verts = [[float(s) for s in line.strip().split(',')] for line in file]
    return verts

def Spherefitter(path):
    correctX=[]
    correctY=[]
    correctZ=[]
    with open(path, "r") as f:
      text=f.readlines()
      for i in text:
        s=i[:-2]
        s=s.split(',')
        correctX.append(float(s[0]))
        correctY.append(float(s[1]))
        correctZ.append(float(s[2]))

    spX = np.array(correctX)
    spY = np.array(correctY)
    spZ = np.array(correctZ)
    A = np.zeros((len(spX),4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX),1))
    f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
    C, residules, rank, singval = np.linalg.lstsq(A,f)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = math.sqrt(t)

    return radius, C[0], C[1], C[2]