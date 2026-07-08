"""
Train Random Forest Model
1. Loads master data (검색량.csv)
2. Vectorize texts using SBERT (Cached)
3. Train Random Forest (MultiOutput)
4. Save model to C:/Keyword/model/
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from src.inference import SBERTVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from src.config import Config
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model():
    # 1. Config & Paths
    config = Config()
    DATA_PATH = config.MASTER_PATH
    MODEL_DIR = config.MODEL_DIR
    MODEL_PATH = config.MODEL_PATH
    VECTOR_CACHE = config.VECTOR_CACHE_PATH
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. Load Data
    logger.info("Loading master data...")
    df = pd.read_csv(DATA_PATH)
    item_count = len(df)
    logger.info(f"Loaded {item_count} rows.")
    
    # Preprocessing
    df['kw_low'] = df['Keyword'].astype(str).str.lower().str.strip()
    # Drop duplicates to ensure clean training
    df = df.drop_duplicates(subset=['kw_low'])
    logger.info(f"Unique keywords: {len(df)}")
    
    # Targets
    target_cols = ['Division', 'Product', 'Keyword Type', 'Brand', 'Type']
    # Fill NAs
    for col in target_cols:
        df[col] = df[col].fillna('Generic')
        
    y = df[target_cols].values
    
    # 3. Vectorization (SBERT)
    vectorizer = SBERTVectorizer()
    
    if VECTOR_CACHE.exists():
        logger.info("Loading cached vectors...")
        sbert_vectors = np.load(VECTOR_CACHE)
        if len(sbert_vectors) != len(df):
            logger.warning("Cache size mismatch! Re-encoding...")
            sbert_vectors = vectorizer.encode(df['kw_low'].tolist(), show_progress_bar=True)
            np.save(VECTOR_CACHE, sbert_vectors)
    else:
        logger.info("Encoding vectors (This may take a while)...")
        sbert_vectors = vectorizer.encode(df['kw_low'].tolist(), show_progress_bar=True)
        np.save(VECTOR_CACHE, sbert_vectors)
    
    # 3b. TF-IDF Vectorization (Optimal Settings from Ralph Optimization)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import hstack, csr_matrix
    
    logger.info("Generating TF-IDF vectors (max_features=1500, ngram_range=(2,3), analyzer=char_wb)...")
    tfidf = TfidfVectorizer(
        max_features=1500,
        ngram_range=(2, 3),
        analyzer='char_wb',
        lowercase=True
    )
    tfidf_vectors = tfidf.fit_transform(df['kw_low'].tolist())
    
    # Save TF-IDF vectorizer for inference
    TFIDF_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
    joblib.dump(tfidf, TFIDF_PATH)
    logger.info(f"TF-IDF vectorizer saved to {TFIDF_PATH}")
    
    # Combine SBERT + TF-IDF
    sbert_sparse = csr_matrix(sbert_vectors)
    X = hstack([sbert_sparse, tfidf_vectors])
    logger.info(f"Combined Feature shape: {X.shape}, Target shape: {y.shape}")
    
    # 4. Train Random Forest
    logger.info("Training Random Forest (MultiOutput)...")
    # [FINAL OPTIMIZED SETTINGS - Ralph Optimization Results]
    # TF-IDF: max_features=1500, ngram_range=(2,3), analyzer=char_wb -> 96.68% Accuracy
    # RF: n_estimators=150, max_depth=None -> Best efficiency (96.78%, 9.0s)
    rf = RandomForestClassifier(
        n_estimators=150,  # Optimal balance of speed & accuracy
        max_depth=None,    # Unlimited depth for max accuracy
        n_jobs=-1, 
        class_weight='balanced',
        random_state=42
    )
    multi_rf = MultiOutputClassifier(rf, n_jobs=-1)
    
    multi_rf.fit(X, y)
    logger.info("Training complete.")
    
    # 5. Save Model
    logger.info(f"Saving model to {MODEL_PATH}...")
    joblib.dump(multi_rf, MODEL_PATH)
    logger.info("DONE.")

if __name__ == "__main__":
    train_model()
