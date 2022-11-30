import torch
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pointnet.pointnet import PointCloudData, PointNet, Normalize, ToTensor, RandomNoise, RandRotation_z, pointnetloss
import pandas as pd

def sort_num(pc):
 return pc[2]

df_torus = pd.read_csv('geocsv/Torus.csv')
df_cone = pd.read_csv('geocsv/Cone.csv')
df_plane = pd.read_csv('geocsv/Plane.csv')
df_sphere = pd.read_csv('geocsv/Sphere.csv')
df_cylinder = pd.read_csv('geocsv/Cylinder.csv')

path = "dataset/test/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pointnet = PointNet()
pointnet.to(device);

pointnet.load_state_dict(torch.load('model/modelfull_9_save.pth'))

train_transforms = transforms.Compose([
                    Normalize(),
                    RandRotation_z(),
                    RandomNoise(),
                    ToTensor()
                    ])

valid_ds = PointCloudData(path, valid=True, folder='test', transform=train_transforms)
valid_loader = DataLoader(dataset=valid_ds, batch_size=1)

pointnet.eval()
all_preds = []
all_labels = []
with torch.no_grad():
  for i, data in enumerate(valid_loader):
        print('Batch [%4d / %4d]' % (i+1, len(valid_loader)))
        inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
        outputs, __, __ = pointnet(inputs.transpose(1,2))
        _, preds = torch.max(outputs.data, 1)
        all_preds.append([preds.detach().cpu().numpy(), data['filename']])
        all_labels.append(preds.detach().cpu().numpy())

res = []
for i in all_preds:
  res.append([i[0][0], i[1][0], int(i[1][0][10:i[1][0].find('.')])])
res.sort(key = sort_num)

path = "run/"
j = 0
for i in res:
  file = i[1]
  print(path + file[:file.find('.')] + "_prediction.txt")
  with open(path + file[:file.find('.')] + "_prediction.txt" , 'w') as f:
    #print(path + str(i[1][:i[1].find('.')] + "_prediction.txt")
    if(i[0] == 0):
      f.write('4' + "\n")
      name = i[1][:i[1].find(".")] + ".ply"
      id = df_cone.index[df_cone['filename'] == name]
      f.write(str(df_cone['aper'].values[id][0]) + '\n')
      f.write(str(df_cone['ax_x'].values[id][0]) + '\n')
      f.write(str(df_cone['ax_y'].values[id][0]) + '\n')
      f.write(str(df_cone['ax_z'].values[id][0]) + '\n')
      f.write(str(df_cone['v_x'].values[id][0]) + '\n')
      f.write(str(df_cone['v_y'].values[id][0]) + '\n')
      f.write(str(df_cone['v_z'].values[id][0]) + '\n')
    elif(i[0] == 1):
      f.write('2' + "\n")
      id = df_cylinder.index[df_cylinder['filename'] == str(i[1])]
      f.write(str(df_cylinder['rad'].values[id][0]) + '\n')
      f.write(str(df_cylinder['ax_x'].values[id][0]) + '\n')
      f.write(str(df_cylinder['ax_y'].values[id][0]) + '\n')
      f.write(str(df_cylinder['ax_z'].values[id][0]) + '\n')
      f.write(str(df_cylinder['pt_x'].values[id][0]) + '\n')
      f.write(str(df_cylinder['pt_y'].values[id][0]) + '\n')
      f.write(str(df_cylinder['pt_z'].values[id][0]) + '\n')
    elif(i[0] == 2):
      f.write('1' + "\n")
      id = df_plane.index[df_plane['filename'] == str(i[1])]
      f.write(str(df_plane['nx'].values[id][0]) + '\n')
      f.write(str(df_plane['ny'].values[id][0]) + '\n')
      f.write(str(df_plane['nz'].values[id][0]) + '\n')
      f.write(str(df_plane['x'].values[id][0]) + '\n')
      f.write(str(df_plane['y'].values[id][0]) + '\n')
      f.write(str(df_plane['z'].values[id][0]) + '\n')
    elif(i[0] == 3):
      f.write('3' + "\n")
      id = df_sphere.index[df_sphere['filename'] == str(i[1])]
      f.write(str(df_sphere['rad'].values[id][0]) + '\n')
      f.write(str(df_sphere['center_x'].values[id][0]) + '\n')
      f.write(str(df_sphere['center_y'].values[id][0]) + '\n')
      f.write(str(df_sphere['center_z'].values[id][0]) + '\n')
    elif(i[0] == 4):
      f.write('5' + "\n")
      id = df_torus.index[df_torus['filename'] == str(i[1])]
      f.write(str(df_torus['b_rad'].values[id][0]) + '\n')
      f.write(str(df_torus['s_rad'].values[id][0]) + '\n')
      f.write(str(df_torus['ax_x'].values[id][0]) + '\n')
      f.write(str(df_torus['ax_y'].values[id][0]) + '\n')
      f.write(str(df_torus['ax_z'].values[id][0]) + '\n')
      f.write(str(df_torus['center_x'].values[id][0]) + '\n')
      f.write(str(df_torus['center_y'].values[id][0]) + '\n')
      f.write(str(df_torus['center_z'].values[id][0]) + '\n')
    j+= 1

