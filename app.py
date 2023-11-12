from fastapi import FastAPI, HTTPException, JSONResponse
from pydantic import BaseModel
import test

app = FastAPI()

class InputData(BaseModel):
    heading: str
    text: str

@app.post("/predict", response_class=JSONResponse)
async def predict(input_data: InputData):
    input_text = input_data.heading + " " + input_data.text
    confidence_levels = test.predict(input_text)
    return confidence_levels

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
