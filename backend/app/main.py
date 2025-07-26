from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from .core.recommender import FilmRecommender, get_recommender
from .core.models import RecommendationRequest, RecommendationResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Emotion-Based Movie Recommendation API",
    description="Provides personalized movie recommendations based on user text.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    logger.info("Application is starting up...")
    get_recommender()
    logger.info("Models are loaded, application is ready to accept requests.")


@app.get("/", tags=["Status"])
def read_root():
    return {"status": "API is running", "version": "3.0.0"}


@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def recommend_films(
    request: RecommendationRequest,
    recommender: FilmRecommender = Depends(get_recommender),
):
    try:
        logger.info(f"Recommendation request received: '{request.text}'")
        recommendations = recommender.recommend(text=request.text, top_k=request.top_k)
        if not recommendations:
            raise HTTPException(status_code=404, detail="No suitable movies found.")

        return {"recommendations": recommendations}

    except Exception as e:
        logger.error(f"Error during recommendation: {e}")
        raise HTTPException(
            status_code=500, detail="An internal server error occurred."
        )
