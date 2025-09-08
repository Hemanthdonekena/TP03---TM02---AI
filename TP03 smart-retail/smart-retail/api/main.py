from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
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
 
#NEW: Simple HTML view
@app.get("/recommend_ui", response_class=HTMLResponse)
def recommend_ui(customer_id: int = 1, recent_category: Optional[str] = None, top_k: int = 5):
    recs = recommend_for_customer(customer_id, recent_category, top_k)
    html = "<h2>Recommendations</h2><table border='1'><tr><th>Category</th><th>Score</th></tr>"
    for r in recs:
        html += f"<tr><td>{r['category']}</td><td>{r['score']:.2f}</td></tr>"
    html += "</table>"
    return html
 