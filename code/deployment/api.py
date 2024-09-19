from code.model import load_model, predict_tumor
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import io
from PIL import Image

model_version = "v1.0.0"
image_size = 256

app = FastAPI()

model = load_model(version=model_version)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    global model
    try:
        # Read the uploaded file and convert it into an OpenCV image
        image_stream = io.BytesIO(await file.read())
        image = Image.open(image_stream).convert("RGB")  
        image = np.array(image)

        # Resize the image to the required size
        img = cv2.resize(image, (image_size, image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=0)

        # Call the model's prediction function
        prediction = predict_tumor(img, model)

        return JSONResponse(content={"prediction": prediction})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@app.get("/")
def read_root():
    return {"message": "Welcome to the Brain Tumor Classification API"}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)