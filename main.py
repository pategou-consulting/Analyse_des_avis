import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from tools import *
from motor.motor_asyncio import AsyncIOMotorClient

# Load the MongoDB connection string from the environment variable MONGODB_URI
CONNECTION_STRING = os.environ['MONGODB_URI']

# Create a MongoDB client
client = AsyncIOMotorClient(CONNECTION_STRING)
# FastAPI setup


app = FastAPI()

class ReviewInput(BaseModel):
    reviews: List[str]  # Changement pour accepter une liste d'avis

@app.post("/analyze_reviews")
def analyze_reviews(review_input: ReviewInput):
    results = []
    for review in review_input.reviews:
        label, confidence = analyze_review_bart(review)
        sentiment = analyze_sentiment(review)
        topic = get_topics(review)
        results.append({
            "review_text": review,
            "predicted_label": label,
            "confidence": confidence,
            "sentiment_score": sentiment,
            "topic": topic
        })
    return results