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
    result = await predict_emotions_async(request.text)
    return PredictionResponse(
        top_emotions=[
            EmotionResponse(
                emotion=data['emotion'],
                probability=data['probability']
            ) for data in result[0]
        ]
    )


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=5000)
