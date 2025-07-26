import pandas as pd
import numpy as np
import faiss
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langdetect import detect, LangDetectException
import os
import logging
from ast import literal_eval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NLLB modelinin beklediği dil kodları için bir harita
# langdetect ('it', 'fr') -> nllb ('ita_Latn', 'fra_Latn')
LANG_CODE_MAP = {
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "it": "ita_Latn",
    "de": "deu_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "zh-cn": "zho_Hans",
}


class FilmRecommender:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(FilmRecommender, cls).__new__(cls)
        return cls._instance

    def __init__(self, data_path="app/data/"):
        if hasattr(self, "initialized"):
            return

        logger.info("FilmRecommender system is starting for the first time...")
        self.df_filmler = self._load_and_preprocess_data(data_path)

        logger.info("Loading AI models...")
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
        self.embedding_model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )

        logger.info("Loading translation model (NLLB)... This may take a while.")
        self.translator = pipeline(
            "translation", model="facebook/nllb-200-distilled-600M"
        )
        logger.info("All models loaded successfully.")

        logger.info("Creating corpus for semantic search...")
        self.df_filmler["corpus"] = (
            self.df_filmler["overview"] + " " + self.df_filmler["keywords"]
        )
        corpus = self.df_filmler["corpus"].tolist()

        logger.info("Encoding movies to vectors...")
        self.movie_vectors = self.embedding_model.encode(
            corpus, show_progress_bar=True
        ).astype("float32")

        logger.info("Creating FAISS vector index...")
        vector_dimension = self.movie_vectors.shape[1]
        self.faiss_index = faiss.IndexFlatL2(vector_dimension)
        self.faiss_index.add(self.movie_vectors)
        logger.info("FAISS index is ready.")

        self.sentiment_genre_map = {
            "POSITIVE": {
                "genres": [
                    "Comedy",
                    "Adventure",
                    "Family",
                    "Music",
                    "Romance",
                    "Fantasy",
                ],
                "bonus": 0.25,
            },
            "NEGATIVE": {
                "genres": ["Drama", "Action", "Thriller", "War", "History"],
                "bonus": 0.2,
            },
        }
        self.initialized = True
        logger.info("FilmRecommender system is fully ready!")

    def _load_and_preprocess_data(self, data_path):
        # Bu fonksiyonun içeriği aynı, değişiklik yok...
        logger.info("Loading and preprocessing movie data...")
        meta_path = os.path.join(data_path, "movies_metadata.csv")
        keywords_path = os.path.join(data_path, "keywords.csv")
        df_meta = pd.read_csv(meta_path, low_memory=False)
        df_keywords = pd.read_csv(keywords_path)
        df_meta = df_meta[df_meta["id"].str.isnumeric()]
        df_meta["id"] = df_meta["id"].astype(int)
        df = pd.merge(df_meta, df_keywords, on="id", how="inner")
        df = df[["id", "title", "overview", "genres", "keywords", "poster_path"]].copy()
        df.dropna(subset=["overview", "title"], inplace=True)
        df["overview"] = df["overview"].fillna("")

        def parse_json_column(data, key="name"):
            try:
                items = literal_eval(data)
                return (
                    ", ".join([item[key] for item in items])
                    if isinstance(items, list)
                    else ""
                )
            except:
                return ""

        df["genre"] = df["genres"].apply(parse_json_column)
        df["keywords"] = df["keywords"].apply(parse_json_column)
        base_poster_url = "https://image.tmdb.org/t/p/w500"
        df["poster_url"] = df["poster_path"].apply(
            lambda x: (
                base_poster_url + x
                if isinstance(x, str) and x.startswith("/")
                else None
            )
        )
        logger.info(f"Preprocessing complete. {len(df)} movies loaded into the system.")
        return df

    def recommend(self, text: str, top_k: int = 5):
        # ... (Adım 1, 2, 3 - Duygu Analizi, Arama ve Puanlama aynı kalıyor) ...
        sentiment_result = self.sentiment_analyzer(text)
        sentiment_label = sentiment_result[0]["label"]
        user_vector = self.embedding_model.encode([text]).astype("float32")
        distances, indices = self.faiss_index.search(user_vector, top_k * 3)
        similar_movies_idx = indices[0][indices[0] != -1]
        similar_movies = self.df_filmler.iloc[similar_movies_idx].copy()
        similar_movies["similarity_score"] = 1 / (1 + distances[0][distances[0] != -1])
        bonus_info = self.sentiment_genre_map.get(sentiment_label, {"bonus": 0})

        def score_function(row):
            score = row["similarity_score"]
            if "genres" in bonus_info and any(
                genre in row["genre"] for genre in bonus_info["genres"]
            ):
                score += bonus_info["bonus"]
            return score

        similar_movies["final_score"] = similar_movies.apply(score_function, axis=1)
        final_recommendations_df = similar_movies.sort_values(
            by="final_score", ascending=False
        ).head(top_k)

        # YENİ ADIM 4: DİL TESPİTİ VE GEREKİRSE İNGİLİZCE'YE ÇEVİRME
        logger.info("Checking and translating overviews to English if necessary...")

        translated_overviews = []
        for overview in final_recommendations_df["overview"]:
            if not isinstance(overview, str) or len(overview.strip()) < 20:
                translated_overviews.append(
                    overview
                )  # Kısa veya geçersiz metinleri atla
                continue

            try:
                lang = detect(overview)
                if lang == "en":
                    translated_overviews.append(overview)  # Zaten İngilizce, dokunma
                else:
                    # Dili NLLB modelinin anlayacağı koda çevir
                    src_lang_code = LANG_CODE_MAP.get(lang)
                    if src_lang_code:
                        logger.info(
                            f"Detected language '{lang}', translating to English..."
                        )
                        translated_text = self.translator(
                            overview, src_lang=src_lang_code, tgt_lang="eng_Latn"
                        )[0]["translation_text"]
                        translated_overviews.append(translated_text)
                    else:
                        logger.warning(
                            f"Detected language '{lang}' is not in the translation map, skipping translation."
                        )
                        translated_overviews.append(overview)  # Haritada yoksa dokunma
            except LangDetectException:
                logger.warning(
                    "Could not detect language for an overview, skipping translation."
                )
                translated_overviews.append(overview)  # Dil tespit edilemezse dokunma
            except Exception as e:
                logger.error(f"An unexpected error occurred during translation: {e}")
                translated_overviews.append(
                    overview
                )  # Herhangi bir hatada orjinalini koru

        final_recommendations_df["overview"] = translated_overviews

        return final_recommendations_df[
            ["title", "genre", "overview", "poster_url", "final_score"]
        ].to_dict(orient="records")


def get_recommender():
    return FilmRecommender()
