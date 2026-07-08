"""
Ralph-Style Optimizer for Random Forest Hyperparameters
Uses the optimal TF-IDF settings already found.
"""
import pandas as pd
import numpy as np
import logging
import time
from pathlib import Path
from itertools import product
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack, csr_matrix
from src.inference import SBERTVectorizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== Configuration Space ==========
RF_PARAM_GRID = {
    'n_estimators': [50, 100, 150, 200, 300],
    'max_depth': [10, 15, 20, 25, None],  # None = unlimited depth
}

# Fixed TF-IDF settings (from previous optimization)
TFIDF_SETTINGS = {
    'max_features': 1500,
    'ngram_range': (2, 3),
    'analyzer': 'char_wb'
}

def load_data():
    DATA_PATH = Path(r"C:\Keyword\master\검색량.csv")
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df['kw_low'] = df['Keyword'].astype(str).str.lower().str.strip()
    df = df.drop_duplicates(subset=['kw_low'])
    
    if len(df) > 10000:
        logger.info(f"Sampling 10000 from {len(df)} for fast iteration...")
        df = df.sample(n=10000, random_state=42)
    
    target_cols = ['Division', 'Product', 'Keyword Type', 'Brand', 'Type']
    for col in target_cols:
        df[col] = df[col].fillna('Generic')
    
    return df, df['kw_low'].tolist(), df[target_cols].values

def get_combined_features(texts):
    """Get SBERT + TF-IDF combined features."""
    cache_path = Path("C:/Keyword/model/sample_sbert_vectors.npy")
    
    # SBERT
    if cache_path.exists():
        logger.info("Loading cached SBERT vectors...")
        sbert_vectors = np.load(cache_path)
    else:
        logger.info("Encoding SBERT vectors...")
        vectorizer = SBERTVectorizer()
        sbert_vectors = vectorizer.encode(texts, show_progress_bar=True)
        np.save(cache_path, sbert_vectors)
    
    # TF-IDF
    logger.info("Generating TF-IDF vectors...")
    tfidf = TfidfVectorizer(**TFIDF_SETTINGS)
    tfidf_vectors = tfidf.fit_transform(texts)
    
    # Combine
    sbert_sparse = csr_matrix(sbert_vectors)
    X_combined = hstack([sbert_sparse, tfidf_vectors])
    
    return X_combined

def evaluate_rf_config(X, y, n_estimators, max_depth):
    """Evaluate a Random Forest configuration."""
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        class_weight='balanced',
        random_state=42
    )
    
    # Use first target (Division) for speed
    y_division = y[:, 0]
    
    try:
        scores = cross_val_score(rf, X, y_division, cv=3, scoring='accuracy', n_jobs=-1)
        return np.mean(scores), np.std(scores)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 0.0, 0.0

def ralph_optimize_rf():
    """Ralph-style optimization for Random Forest."""
    logger.info("="*60)
    logger.info("RALPH OPTIMIZATION: Random Forest Hyperparameters")
    logger.info("="*60)
    
    df, texts, y = load_data()
    X = get_combined_features(texts)
    
    configs = list(product(
        RF_PARAM_GRID['n_estimators'],
        RF_PARAM_GRID['max_depth']
    ))
    
    results = []
    best_score = 0
    best_config = None
    
    logger.info(f"Testing {len(configs)} configurations...")
    
    for i, (n_est, max_d) in enumerate(configs, 1):
        depth_str = str(max_d) if max_d else "None"
        logger.info(f"[{i}/{len(configs)}] Testing: n_estimators={n_est}, max_depth={depth_str}")
        
        start_time = time.time()
        mean_acc, std_acc = evaluate_rf_config(X, y, n_est, max_d)
        elapsed = time.time() - start_time
        
        logger.info(f"    -> Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f}) | Time: {elapsed:.1f}s")
        
        results.append({
            'n_estimators': n_est,
            'max_depth': depth_str,
            'accuracy': mean_acc,
            'std': std_acc,
            'time': elapsed
        })
        
        if mean_acc > best_score:
            best_score = mean_acc
            best_config = (n_est, max_d)
            logger.info(f"    *** NEW BEST CONFIG! ***")
    
    results_df = pd.DataFrame(results).sort_values('accuracy', ascending=False)
    
    print("\n" + "="*80)
    print("            RALPH OPTIMIZATION RESULTS: Random Forest")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    depth_str = str(best_config[1]) if best_config[1] else "None"
    print(f"\n🏆 BEST CONFIG: n_estimators={best_config[0]}, max_depth={depth_str}")
    print(f"   Accuracy: {best_score:.4f}")
    print("="*80)
    
    results_df.to_csv("C:/Keyword/output/ralph_rf_optimization_results.csv", index=False)
    logger.info("Results saved to C:/Keyword/output/ralph_rf_optimization_results.csv")
    
    return best_config

if __name__ == "__main__":
    ralph_optimize_rf()
