# dataset
from utils import ImageFolder, Dataset, DataLoader

import torchvision.transforms as transforms
class ImageData(Dataset):
  def __init__(self, dir, transform=None):
    self.data = ImageFolder(dir, transform=transform)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]


transform = transforms.Compose([
    transforms.Resize((200,200)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

<<<<<<< HEAD
trainset = ImageData(dir=r"/home/abdullah/Desktop/celeb-classification/dataset/train", transform=transform)
validset = ImageData(dir=r"/home/abdullah/Desktop/celeb-classification/dataset/validation", transform=transform)
testset = ImageData(dir=r"/home/abdullah/Desktop/celeb-classification/dataset/test", transform=transform)
=======
trainset = ImageData(dir=r"C:\Users\D2\Desktop\celebrity-classification\dataset\train", transform=transform)
validset = ImageData(dir=r"C:\Users\D2\Desktop\celebrity-classification\dataset\validation", transform=transform)
testset = ImageData(dir=r"C:\Users\D2\Desktop\celebrity-classification\dataset\test", transform=transform)
>>>>>>> cd1e927ba8f4fc72f7d91b4e0998ff2d68d63ad2

trainloader = DataLoader(trainset, batch_size=8, shuffle=True)
validloader = DataLoader(validset, batch_size=8, shuffle=False)
testloader = DataLoader(testset, batch_size=8, shuffle=False)


print(trainloader)

print(validloader)

print(testloader)

for images, labels in trainloader:
  print(images.shape)
  print(labels.shape)
  break

