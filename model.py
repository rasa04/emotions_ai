from typing import List, Any

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained('./model')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Check if GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# List of emotions used in the GoEmotions dataset
emotions = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]


# Pydantic models for FastAPI
class EmotionResponse(BaseModel):
    emotion: str
    probability: float


class PredictionResponse(BaseModel):
    top_emotions: List[EmotionResponse]


def count_tokens(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    return len(inputs['input_ids'][0])


def predict_emotions(text: str) -> tuple[list[dict[str, str | list[str | Any] | Any]], int, Any]:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
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

    return top_emotions, count_tokens(text), outputs.logits.size(1)
