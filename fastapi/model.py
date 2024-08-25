# model.py
import utils

from utils import torch, torchvision, nn, F
from data import transforms, trainloader
class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.conv1 = nn.Conv2d(3,12, 5)
    self.pool = nn.MaxPool2d(2,2)

    self.conv2 = nn.Conv2d(12,32, 5)

    self.dropout = nn.Dropout(p=0.2)
    self.fc1 = nn.Linear(32*47*47,84)
    self.fc2 = nn.Linear(84,64)
    self.fc3 = nn.Linear(64,10)

  def forward(self,x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = torch.flatten(x,1)

    
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
  
model = Model()

print(model)

for images, lables in trainloader:
  break

y_pred = model(images)

print(y_pred.shape)

