import os 
from tqdm import tqdm

this_path = "TXT_DATA_PATH"
path_des="PLY_DATA_PATH"

def read_all(file):
  file=file.split("\n")
  verts = [] 
  for line in file:
    if (len(line)<30): break
    l=line.split(',')
    verts.append([float(l[0]),float(l[1]),float(l[2])])
  return verts


j = 0
for list_file in tqdm(os.listdir(this_path)):
    v=[]
    with open(this_path + list_file, 'r') as f:
      try:
        data=f.read()
        v=read_all(data)
      except:
        print(list_file)
    with open(path_des + list_file[:list_file.find('.')] + ".ply", 'w') as f:
      f.write("ply" + "\n")
      f.write("format ascii 1.0 " + "\n")
      f.write("element vertex 8100" + "\n")
      f.write("property float32 x" + "\n")
      f.write("property float32 y" + "\n")
      f.write("property float32 z" + "\n")
      f.write("end_header"+ "\n")
      for i in v:
        f.write(str(i[0])+" "+str(i[1])+" "+str(i[2])+"\n")
    j+=1
    if (j%100==0): print(j)