import torch
import pandas as pd
import torch.nn as nn
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import random
import os
import time
import glob
import re
import matplotlib
import matplotlib.pyplot as  plt
matplotlib.use("TkAgg")  # or "Qt5Agg" if you have PyQt installed
#load data
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# NumPy
seed = 43
np.random.seed(seed)

# PyTorch CPU
torch.manual_seed(seed)
BATCHSIZE  = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH  = 200
LEARNING_RATE= 0.001
random.seed(seed)
# PyTorch GPU
torch.cuda.manual_seed(seed)         # seed current GPU
torch.cuda.manual_seed_all(seed)
DATAPATH = "C:/Users/dongs/Desktop/5 第五讲 机器学习训练模型的数学范式v16/cat_dog"
CATPATH = glob.glob(DATAPATH + "/cat/" + "*.jpg")
random.shuffle(CATPATH)
DOGPATH = glob.glob(DATAPATH + "/dog/" + "*.jpg")
random.shuffle(DOGPATH)
save_dir = "./models"
os.makedirs(save_dir, exist_ok=True) 
# Open the image file

TRANSFORMATIONS = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # mirror
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),      # ±15°
    ]
    
)

class AnimalDataset(Dataset):
    def __init__(self, path,typeo,stdsize,transformation = None):
        self.path = path
        self.type = typeo
        self.stdsize = stdsize
        self.transformation = transformation
    def __len__(self):
        return len(self.path)
    def __str__(self):
        return "".join(self.path)
    def __getitem__(self, index):
        im =  transforms.ToTensor()( transforms.Resize((self.stdsize,self.stdsize))(Image.open(self.path[index])))
        
        im = self.transformation(im) if self.transformation != None else im
        
        return im, 0 if self.type == 'cat' else 1
class MergedDataset(ConcatDataset):
    def __init__(self,dataset):
        self.__dict__ = dataset.__dict__.copy()

        self.dataset = dataset
    def __repr__(self):
       
            return (
                f"{type(self).__name__}(total_len={len(self)})\n" +
                "\n".join(
                    f"  [{i}] {type(d).__name__}: "
                    f"{len(getattr(str(d),'paths',[]))} files"
                    for i, d in enumerate(self.datasets[:5])
                )
            )

class CatvsDog(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=(3,3),stride = (1,1), padding = (1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2,stride = 2),
            nn.BatchNorm2d(16)

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),stride = (1,1), padding = (1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2,stride = 2),
            nn.BatchNorm2d(32)
            

        )
        self.conv3  = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride = (1,1), padding = (1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2,stride = 2),
            nn.Dropout2d(p = 0.5),

            nn.BatchNorm2d(64)

        )
        self.fc = nn.Sequential(nn.Linear(in_features=64,out_features=32),
                                 nn.Dropout(p = 0.5),
                                 nn.Sigmoid(),
                                 nn.Linear(in_features=32,out_features=1),
                                 nn.Sigmoid())
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.mean(dim=(2,3))
        x = torch.flatten(x,1)
        return self.fc(x)
        
  

maxCat = max([max(Image.open(PA).size) for PA in CATPATH])
maxDog = max([max(Image.open(PA).size) for PA in DOGPATH])
print(maxCat,"max")
split = int(0.1*len(CATPATH) )if int(0.1*len(CATPATH)) > 5 else 5
catdataset = AnimalDataset(CATPATH[:-split] ,'cat',maxCat,transformation=TRANSFORMATIONS)
dogdataset = AnimalDataset(DOGPATH[:-split] ,'dog',maxDog,transformation=TRANSFORMATIONS)
trainset = MergedDataset(ConcatDataset([catdataset,dogdataset]) )  
trainloader = DataLoader(dataset=trainset,shuffle = True,batch_size= BATCHSIZE)
print(trainset)

catdataset = AnimalDataset(CATPATH[-split:],'cat',maxCat)
dogdataset = AnimalDataset(DOGPATH[-split:],'dog',maxDog)
testset = MergedDataset(ConcatDataset([catdataset,dogdataset]) )  
testloader = DataLoader(dataset=testset,shuffle = False,batch_size= BATCHSIZE)
del catdataset, dogdataset
print(testset)
model = CatvsDog().to(DEVICE) 
# Automatically find the latest saved model
model_files = glob.glob(os.path.join(save_dir, "*.pth"))

if model_files:
    # Option 1: Pick the file with the largest epoch number
    latest_file = max(
        model_files,
        key=lambda x: int(re.findall(r"\d+", os.path.basename(x))[0])
    )
    start = int(re.findall(r"\d+", os.path.basename(latest_file))[0]) + 1
    model.load_state_dict(torch.load(latest_file))
    print(f"Resuming training from {latest_file}, starting at epoch {start}" if start !=  EPOCH + 1 else "Finished Training all EPOCH, continue:")
else:
    start = 1
    print("No saved model found, starting .")
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
#training starts
for i in range(start,EPOCH+1):
    start = time.perf_counter()
    total_correct = 0
    total_samples = 0
    model.train()
    totalLoss = 0
    for image, typ in trainloader:
        optimizer.zero_grad()
        image = image.to(DEVICE)
        typ = typ.to(DEVICE)
        y = model(image).to(DEVICE).squeeze() 
        los =  criterion(y, typ.float())
        optimizer.zero_grad()
        los.backward()
        optimizer.step()
        totalLoss += los.item()
    end = time.perf_counter()

    print(f"|EPOCH {i},loss {totalLoss/len(trainloader)}",end = '')
    print(f"| Execution took {end - start:.4f} seconds")

    if i% 10 ==0  or  i% 10 ==5:
        torch.save(model.state_dict(), os.getcwd()+f"/models/model{i}.pth")
        with torch.no_grad():
            for image, typ in testloader:
                image, typ = image.to(DEVICE), typ.to(DEVICE)
                y = model(image).squeeze(1)  # shape = (batch_size,)
                
                # BCE loss on raw probabilities
                loss = criterion(y, typ.float())
                
                # accuracy
                preds = (y > 0.5).int()
                total_correct += (preds == typ).sum().item()
                total_samples += typ.size(0)
        
    
                accuracy = total_correct / total_samples
                print("===================",accuracy,"=====================")
if start !=  EPOCH + 1 :
    print('training ends')
model_files = glob.glob(os.path.join(save_dir, "*.pth"))
model.eval()
loss_dict = {}

# iterate over all saved model files
for path in sorted(model_files):
    # load model
    model.load_state_dict(torch.load(path))
    model.to(DEVICE)
    
    # get epoch number from filename
    epoch_num = int(re.findall(r"\d+", os.path.basename(path))[0])
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for image, typ in testloader:
            image, typ = image.to(DEVICE), typ.to(DEVICE)
            y = model(image).squeeze(1)  # shape = (batch_size,)
            
            # BCE loss on raw probabilities
            loss = criterion(y, typ.float())
            total_loss += loss.item() * typ.size(0)
            
            # accuracy
            preds = (y > 0.5).int()
            total_correct += (preds == typ).sum().item()
            total_samples += typ.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    # store results
    loss_dict[epoch_num] = {'loss': avg_loss, 'accuracy': accuracy}

# sort by epoch
epochs = sorted(loss_dict.keys())
losses = [loss_dict[e]['loss'] for e in epochs]
accuracies = [loss_dict[e]['accuracy'] for e in epochs]

# plot
fig, ax = plt.subplots(1, 2, figsize=(12,5))

ax[0].plot(epochs, losses, marker='o')
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Test Loss")
ax[0].set_title("Test Loss vs Epoch")

ax[1].plot(epochs, accuracies, marker='o')
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Test Accuracy")
ax[1].set_title("Test Accuracy vs Epoch")
#plotting data
plt.tight_layout()
plt.savefig('RESULTS.pdf')
plt.show()



        