import torch
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pointnet.pointnet import PointCloudData, PointNet, Normalize, ToTensor, RandomNoise, RandRotation_z, pointnetloss

def read_pc(file):
    #n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(',')])
    #print(n_verts, n_faces)
    verts = [[float(s) for s in line.strip().split(',')] for line in file]
    return verts

def train(model, train_loader, val_loader=None,  epochs=10):
    j = 0
    for epoch in tqdm(range(epochs)): 
        pointnet.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = pointnet(inputs.transpose(1,2))

            loss = pointnetloss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:    # print every 10 mini-batches
                print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                    (epoch + 1, i + 1, len(train_loader), running_loss / 10))
                running_loss = 0.0
        # save the model
        torch.save(pointnet.state_dict(), "model/" + "save_" + str(j) + ".pth")
        j += 1

path = "dataset/train/"

os.environ['KMP_DUPLICATE_LIB_OK']='True'

train_transforms = transforms.Compose([
                    Normalize(),
                    RandRotation_z(),
                    RandomNoise(),
                    ToTensor()
                    ])

train_ds = PointCloudData(path, transform=train_transforms)

inv_classes = {i: cat for cat, i in train_ds.classes.items()};

print('Train dataset size: ', len(train_ds))
print('Number of classes: ', len(train_ds.classes))
print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())
print('Class: ', inv_classes[train_ds[0]['category']])

train_loader = DataLoader(dataset=train_ds, batch_size=2, shuffle=True)
valid_loader = DataLoader(dataset=train_ds, batch_size=2, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

pointnet = PointNet()
pointnet.to(device)

optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.00025)

train(pointnet, train_loader, valid_loader)

