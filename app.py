from fastapi import FastAPI
from pydantic import BaseModel
from model import predict_emotions, EmotionResponse, PredictionResponse
from concurrent.futures import ThreadPoolExecutor
import asyncio


class TextRequest(BaseModel):
    text: str


app = FastAPI()


async def predict_emotions_async(text: str):
    return await (
        asyncio
        .get_event_loop()
        .run_in_executor(
            ThreadPoolExecutor(max_workers=10),
            predict_emotions,
            text
        )
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    return PredictionResponse(
        top_emotions=[
            EmotionResponse(
                emotion=emotion['emotion'],
                probability=emotion['probability']
            ) for emotion in await predict_emotions_async(request.text)
        ]
    )


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=5000)
