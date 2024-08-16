# FAST API

from fastapi import FastAPI
from pydantic import BaseModel
import matplotlib.pyplot as plt

from PIL import Image
from test import show_image


app = FastAPI()
class ImageURL(BaseModel):
    url: str



@app.get("/")
def root():
    return {"hello" : "world"}

@app.post("/predicted-image")
def predict_image(url:str):
    result = show_image(url)
    return result
