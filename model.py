import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from typing import List, Dict
from pydantic import BaseModel

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained('./model')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# List of emotions used in the GoEmotions dataset
emotions = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]


def predict_emotions(text: str) -> List[Dict[str, float]]:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)

    # Get probabilities using softmax
    probs = F.softmax(outputs.logits, dim=-1)[0]

    # Get top 5 emotions
    top_probs, top_indices = torch.topk(probs, 5)

    top_emotions = []
    for prob, idx in zip(top_probs, top_indices):
        top_emotions.append({
            'emotion': emotions[idx.item()],
            'probability': prob.item()
        })

    return top_emotions


# Pydantic models for FastAPI
class EmotionResponse(BaseModel):
    emotion: str
    probability: float


class PredictionResponse(BaseModel):
    top_emotions: List[EmotionResponse]
