from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from utils import Agent, Worker
from PIL import Image
import io
import numpy as np

app = FastAPI()

a = Agent()

@app.get("/")
async def root():
    return {"message": "Hello World CIAo manuel"}

@app.post("/focus")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # Read image as opencv
        image = Image.open(io.BytesIO(contents))
        image = np.array(image)
        
        res, im = a.is_looking_at_the_screen(image)

        if res:
            return JSONResponse(content={"message": "Bravo"})
        else:
            return JSONResponse(content={"message": "Concentrati Guarda lo schermo"})
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)