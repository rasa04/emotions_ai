from fastapi import FastAPI
from pydantic import BaseModel

from model import predict_emotions, EmotionResponse, PredictionResponse


class TextRequest(BaseModel):
    text: str


app = FastAPI()


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    top_emotions = predict_emotions(request.text)
    response = PredictionResponse(top_emotions=[
        EmotionResponse(emotion=emotion['emotion'], probability=emotion['probability'])
        for emotion in top_emotions
    ])
    return response


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=5000)
