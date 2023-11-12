from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import test

app = FastAPI()

class InputData(BaseModel):
    heading: str
    text: str

@app.post("/predict")
async def predict(input_data: InputData):
    input_text = str(input_data.text)
    print(input_text)
    confidence_levels = test.predict(input_text)
    return confidence_levels

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.1", port=8000)
