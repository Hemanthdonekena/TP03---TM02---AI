from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from src.inference import recommend_for_customer

app = FastAPI(title="Smart Retail Recommendation API")

class RecommendRequest(BaseModel):
    customer_id: int
    recent_category: Optional[str] = None
    top_k: int = 5

class RecommendResponseItem(BaseModel):
    category: str
    score: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend", response_model=List[RecommendResponseItem])
def recommend(req: RecommendRequest):
    recs = recommend_for_customer(req.customer_id, req.recent_category, req.top_k)
    return recs
