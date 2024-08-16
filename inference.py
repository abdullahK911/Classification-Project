from utils import torch, torchvision, plt, Image
from model import Model
from data import testloader

classes = ["anne-hathaway", "blake-lively", "brad-pitt", "christian-bale", "emily-blunt", "james-bond",
"jennifer-garner", "leonardo-dicaprio", "margot-robbie", "robert-downey-jr"]

# loading model
model = Model()
model.load_state_dict(torch.load(r"C:\Users\D2\Desktop\celebrity-classification\model.pth"))
model.eval()


# test.py
correct_pred = {classname: 0 for classname in classes} # Initialize as dictionaries
total_pred = {classname: 0 for classname in classes}
with torch.no_grad():
  for images, labels in testloader:
    outputs = model(images)
    _, predictions = torch.max(outputs.data, 1)
    for prediction, label in zip(predictions, labels):
      if label == prediction:
        correct_pred[classes[label.item()]] += 1
      total_pred[classes[label.item()]] += 1

for classname, correct_count in correct_pred.items():
  accuracy = 100 * float(correct_count) / total_pred[classname]
  print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")