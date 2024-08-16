from utils import torch, torchvision
from data import trainloader, testloader, validloader
from model import model


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


for epoch in range(100):
  model.train() # start training CALL
  running_loss = 0.0
  for images, labels in trainloader:
    optimizer.zero_grad()
    # output predictions 
    outputs = model(images)
    loss = criterion(outputs,labels)
    # backward propagation
    loss.backward()
    # gradient descent
    optimizer.step()
    running_loss += loss.item() * labels.size(0)
  train_loss = running_loss / len(trainloader.dataset)

  if epoch % 1 == 0:
    print(f"Epoch: {epoch +1}/{100}     -- Training Loss = {train_loss}")

  # eval.py
  with torch.no_grad():
    model.eval()
    correct = 0
    total = 0
    for images, labels in validloader:
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  print(f"accuracy: {100 * correct // total} %")


# saving model
torch.save(model.state_dict(), "model.pth")
