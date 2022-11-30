
from tqdm import tqdm
import os 

path = "RAW_DATASET_PATH"
path_des = "UPSCALED_DATASET_PATH"
list_folder = ["Cone/", "Cylinder/", "Plane/", "Torus/", "Shpere/"]
N_MAX = 8100
exp = 10**-15

def read_all(file):
  file=file.split("\n")
  verts = [] 
  for line in file:
    if (len(line)<30): break
    l=line.split(',')
    verts.append([float(l[0]),float(l[1]),float(l[2])])
  return verts

def generate(v):
  cnt=-1
  ex=exp
  while(len(v)<N_MAX):
    cnt+=1
    if (cnt==3): ex+=exp
    cnt=cnt%3
    for i in v:
      if (len(v)==N_MAX):break
      i[cnt]+=ex
      v.append(i)
    for i in v:
      if (len(v)==N_MAX):break
      i[cnt]-=ex
      v.append(i)
  return v


for i in list_folder:
  this_path = path + i
  des_path=path_des+i
  j=0
  for list_file in tqdm(os.listdir(this_path)):
    v=[]
    with open(this_path + list_file, 'r') as f:
      try:
        data=f.read()
        v=read_all(data)
      except:
        print(list_file)
    v=generate(v)
    with open(des_path + list_file, 'w') as f:
      for i in v:
        f.write(str(i[0])+","+str(i[1])+","+str(i[2])+"\n")
    j+=1
    if (j%100==0): print(j)