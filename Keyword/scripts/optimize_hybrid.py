"""
Ralph-Style Hyperparameter Optimizer for Hybrid Model (SBERT + TF-IDF)
Iterates through configurations, evaluates, and reports the best settings.
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
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.sparse import hstack, csr_matrix
from src.inference import SBERTVectorizer

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== Configuration Space ==========
PARAM_GRID = {
    'max_features': [500, 1000, 1500, 2000],
    'ngram_range': [(2, 3), (2, 4), (3, 5)],
    'analyzer': ['char_wb', 'char'],  # char_wb respects word boundaries
}

def load_data():
    """Load and preprocess master data."""
    DATA_PATH = Path(r"C:\Keyword\master\검색량.csv")
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df['kw_low'] = df['Keyword'].astype(str).str.lower().str.strip()
    df = df.drop_duplicates(subset=['kw_low'])
    
    # Sample for faster iteration (Ralph: iterate fast, refine later)
    if len(df) > 10000:
        logger.info(f"Sampling 10000 from {len(df)} for fast iteration...")
        df = df.sample(n=10000, random_state=42)
    
    target_cols = ['Division', 'Product', 'Keyword Type', 'Brand', 'Type']
    for col in target_cols:
        df[col] = df[col].fillna('Generic')
    
    return df, df['kw_low'].tolist(), df[target_cols].values

def get_sbert_vectors(texts):
    """Get SBERT vectors (cached if possible)."""
    cache_path = Path("C:/Keyword/model/sample_sbert_vectors.npy")
    if cache_path.exists():
        logger.info("Loading cached SBERT vectors...")
        return np.load(cache_path)
    
    logger.info("Encoding SBERT vectors...")
    vectorizer = SBERTVectorizer()
    vectors = vectorizer.encode(texts, show_progress_bar=True)
    np.save(cache_path, vectors)
    return vectors

def evaluate_config(sbert_vectors, texts, y, max_features, ngram_range, analyzer):
    """Evaluate a single TF-IDF configuration combined with SBERT."""
    # TF-IDF
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        analyzer=analyzer,
        lowercase=True
    )
    tfidf_vectors = tfidf.fit_transform(texts)
    
    # Combine: SBERT (dense) + TF-IDF (sparse)
    sbert_sparse = csr_matrix(sbert_vectors)
    X_combined = hstack([sbert_sparse, tfidf_vectors])
    
    # Use only first target (Division) for speed during optimization
    # Full multi-output will be trained after finding best params
    y_division = y[:, 0]
    
    # Random Forest with cross-validation
    rf = RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42)
    
    # Use 3-fold CV for speed
    try:
        scores = cross_val_score(rf, X_combined, y_division, cv=3, scoring='accuracy', n_jobs=-1)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 0.0, 0.0
    
    return mean_score, std_score

def ralph_optimize():
    """Ralph-style iterative optimization."""
    logger.info("="*60)
    logger.info("RALPH OPTIMIZATION: Starting iterative search...")
    logger.info("="*60)
    
    # Load data
    df, texts, y = load_data()
    sbert_vectors = get_sbert_vectors(texts)
    
    # Generate all combinations
    configs = list(product(
        PARAM_GRID['max_features'],
        PARAM_GRID['ngram_range'],
        PARAM_GRID['analyzer']
    ))
    
    results = []
    best_score = 0
    best_config = None
    
    logger.info(f"Testing {len(configs)} configurations...")
    
    for i, (max_feat, ngram, analyzer) in enumerate(configs, 1):
        logger.info(f"[{i}/{len(configs)}] Testing: max_features={max_feat}, ngram_range={ngram}, analyzer={analyzer}")
        
        start_time = time.time()
        mean_acc, std_acc = evaluate_config(sbert_vectors, texts, y, max_feat, ngram, analyzer)
        elapsed = time.time() - start_time
        
        logger.info(f"    -> Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f}) | Time: {elapsed:.1f}s")
        
        results.append({
            'max_features': max_feat,
            'ngram_range': ngram,
            'analyzer': analyzer,
            'accuracy': mean_acc,
            'std': std_acc,
            'time': elapsed
        })
        
        if mean_acc > best_score:
            best_score = mean_acc
            best_config = (max_feat, ngram, analyzer)
            logger.info(f"    *** NEW BEST CONFIG! ***")
    
    # Sort and report
    results_df = pd.DataFrame(results).sort_values('accuracy', ascending=False)
    
    print("\n" + "="*80)
    print("                    RALPH OPTIMIZATION RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    print(f"\n🏆 BEST CONFIG: max_features={best_config[0]}, ngram_range={best_config[1]}, analyzer={best_config[2]}")
    print(f"   Accuracy: {best_score:.4f}")
    print("="*80)
    
    # Save results
    results_df.to_csv("C:/Keyword/output/ralph_optimization_results.csv", index=False)
    logger.info("Results saved to C:/Keyword/output/ralph_optimization_results.csv")
    
    return best_config

if __name__ == "__main__":
    best = ralph_optimize()
