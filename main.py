# FAST API

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import logging
from PIL import Image
from test import show_image


app = FastAPI()
class ImageURL(BaseModel):
    url: str



@app.get("/")
def root():
    return {"hello" : "world"}

@app.post("/predicted-image")
def predict_image(image_url: ImageURL):
    try:
        url = image_url.url
        result = show_image(url)
        return result
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")