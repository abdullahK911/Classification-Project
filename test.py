# image.py
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt # Import matplotlib
from data import transforms
import requests
from io import BytesIO

from utils import torch
from inference import classes, model



def show_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    # img_path = r"C:\Users\D2\Desktop\celebrity-classification\test2.jpeg"
    # img = Image.open(img_path)

    tfsm = transforms.Compose([
        transforms.Resize((200,200)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    # Preprocess the image (resize, convert to tensor, normalize)


    img_tensor = tfsm(img).unsqueeze(0)  # Add batch dimension

    # Assuming 'output' is the model's raw output
    output = model(img_tensor)

    # Get the predicted class index
    _, predicted_class = torch.max(output, 1)
    
    result = classes[predicted_class.item()]
    
    # Visualize the input image (optional)
    # plt.imshow(img)
    # plt.title("Input Image")
    # plt.show()
    return f"Predicted class: {result}"


url = "https://images.hellomagazine.com/horizon/landscape/5774fe2559d3-daniel-craig.jpg"

if __name__ == "__main__":
    show_image(url)

