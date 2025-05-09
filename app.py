from fastapi import FastAPI, File, UploadFile
import pandas as pd
import joblib
from io import BytesIO

app = FastAPI()

# Загрузка обученной модели
model_path = "/content/drive/MyDrive/Colab Notebooks/laptop_price_model.pkl"
model = joblib.load(model_path)

@app.post("/content/drive/MyDrive/Colab Notebooks/")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(BytesIO(content))
    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}
