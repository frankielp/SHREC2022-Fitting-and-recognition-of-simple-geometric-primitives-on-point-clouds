import pandas as pd
from tqdm import tqdm
import os

from geofitter.planefitter import Planefitter
from geofitter.spherefitter import Spherefitter
from geofitter.cylinderfitter import Cylinderfitter
from geofitter.conefitter import Conefitter
from geofitter.torusfitter import Torusfitter

path = "dataset/val/"

df_plane = pd.DataFrame(columns=["filename" , "nx", "ny", "nz", "x", "y", "z"])
df_cylinder = pd.DataFrame(columns=["filename" , 'rad', 'ax_x' , 'ax_y' , 'ax_z' , 'pt_x' , 'pt_y' , 'pt_z' ])
df_sphere = pd.DataFrame(columns=["filename" , 'rad', 'center_x', 'center_y' , 'center_z'])
df_torus = pd.DataFrame(columns=["filename" , 'b_rad' , 's_rad' , 'ax_x' , 'ax_y' , 'ax_z' , 'center_x' , 'center_y' , 'center_z'])
df_cone = pd.DataFrame(columns=["filename", 'aper', 'ax_x', 'ax_y', 'ax_z', 'v_x', 'v_y', 'v_z'])

def classify(num, path, df):
      for list_file in tqdm(os.listdir(path)):
        if num == 0:
          aper, ax_x, ax_y, ax_z, v_x, v_y, v_z = Conefitter(path+list_file)
          df = df.append({'filename' : list_file, 'aper' : aper, 'ax_x' : ax_x, 'ax_y' : ax_y, 'ax_z' : ax_z, 'v_x' : v_x, 'v_y': v_y, 'v_z': v_z}, ignore_index=True)

        if(num == 1):
          cylinder = Cylinderfitter(path+list_file)
          try:
            rad = cylinder.radius
            ax = cylinder.direction
            pt = cylinder.anchor_point
          except:
            rad=0
            ax=[-1000,-1000,-1000]
            pt = [-1000,-1000,-1000]
          
          df = df.append({'filename' : list_file, 'rad': rad, 'ax_x' : ax[0], 'ax_y' : ax[1], 'ax_z' : ax[2], 'pt_x' : pt[0], 'pt_y' : pt[1], 'pt_z' : pt[2]}, ignore_index=True)
          #print(2) 
        if(num == 2):
          n, x, y, z = Planefitter(path + list_file)
          df = df.append({'filename':  list_file , 'nx' : float(n[0,0]), 'ny' : float(n[1,0]), 'nz' : float(n[2,0]), 'x' : float(x), 'y' : float(y), 'z': float(z[0,0])}, ignore_index=True)
          print(n[0,0], n[1,0], n[2,0], x, y, z[0,0])
          print(df_plane)
          #print(1)
        if(num == 3):
          try:
            rad, x, y, z = Spherefitter(path+list_file)
          except:
            rad =0
            x= [-1000]
            y= [-1000]
            z= [-1000]
          df = df.append({'filename' : list_file, 'rad' : rad, 'center_x' : x[0], 'center_y' : y[0], 'center_z' : z[0]}, ignore_index=True)
          print(x[0],y[0],z[0])
          #print(3)
        if(num == 4):
          torus = Torusfitter(path+list_file)
          print(torus)
          try:
            b_rad = torus.major_radius
            s_rad = torus.minor_radius
            ax = torus.direction
            center = torus.center
          except:
            b_rad=0
            s_rad=0
            ax=[-1000,-1000,-1000]
            center=[-1000,-1000,-1000]
          df = df.append({'filename' : list_file, 'b_rad' : b_rad, 's_rad' : s_rad, 'ax_x' : ax[0], 'ax_y' : ax[1], 'ax_z' : ax[2], 'center_x' : center[0], 'center_y' : center[1], 'center_z' : center[2] }, ignore_index=True)
          #print(5)
      return df

for i in range(0,5):
  if i==0:
    df_cone = classify(0,path + "val_ply/", df_cone)
    df_cone.to_csv('geocsv/Cone.csv')
  elif i==1:
    df_cylinder = classify(1,path + "val_txt/", df_cylinder)
    df_cylinder.to_csv('geocsv/Cylinder.csv')
  elif i==2:
    df_plane = classify(2,path + "val_txt/", df_plane)
    df_plane.to_csv('geocsv/Plane.csv')
  elif i==3:
    df_sphere = classify(3,path + "val_txt/", df_sphere)
    df_sphere.to_csv('geocsv/Sphere.csv')
  elif i==4:
    df_torus = classify(4,path + "val_txt/", df_torus)
    df_torus.to_csv('geocsv/Torus.csv')