from pydantic import BaseModel, HttpUrl
from typing import List, Optional


class RecommendationRequest(BaseModel):
    text: str
    top_k: int = 5


class FilmRecommendation(BaseModel):
    title: str
    genre: str
    overview: str
    poster_url: Optional[HttpUrl]
    final_score: float


class RecommendationResponse(BaseModel):
    recommendations: List[FilmRecommendation]
