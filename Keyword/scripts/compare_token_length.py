"""
Confidence Comparison Script: Default vs Max Seq Length 64
"""
import pandas as pd
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from src.inference import SBERTVectorizer, MLModel
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_models():
    # 1. Load Data (Sample of long keywords)
    # Try multiple sources
    df_path = Path("C:/Keyword/input/SMP_UPdate.csv")
    if not df_path.exists():
        df_path = Path("C:/Keyword/master/검색량.csv")
    
    logger.info(f"Loading data from {df_path}")
    df = pd.read_csv(df_path)
    
    # Pick top 20 longest keywords
    df['len'] = df['Keyword'].astype(str).str.len()
    long_kws = df.sort_values('len', ascending=False).head(20)['Keyword'].tolist()
    
    # Add some manually crafted VERY long keywords for stress test
    # Limit test keywords to 5 for speed
    long_kws = long_kws[:5]
    logger.info(f"Test Keywords: {len(long_kws)}")

    # 2. Load RF Classifier
    model_path = Path("C:/Keyword/model/rf_classifier.joblib")
    if not model_path.exists():
        logger.error("RF Model not found!")
        return

    ml_model = MLModel(model_path) # Wrapper

    # 3. Model 1: Limited (64)
    logger.info("Encoding with Limited Model (64 tokens)...")
    vec_limited = SBERTVectorizer(model_name='all-MiniLM-L6-v2')
    vec_limited.model.max_seq_length = 64
    vectors_limit = vec_limited.encode(long_kws)
    
    res_limit = ml_model.predict(vectors_limit)
    conf_limit = [r['Confidence'] for r in res_limit]

    # 4. Model 2: Default (Unlimited / 128)
    logger.info("Encoding with Default Model (128 tokens)...")
    # Reuse model to save memory, just change property
    vec_limited.model.max_seq_length = 128
    vectors_default = vec_limited.encode(long_kws)
    
    res_default = ml_model.predict(vectors_default)
    conf_default = [r['Confidence'] for r in res_default]

    # 5. Compare & Report
    print("\n" + "="*80)
    print(f"{'Keyword (Truncated)':<50} | {'Conf(64)':<10} | {'Conf(128)':<10} | {'Delta':<10}")
    print("-" * 80)
    
    for i, kw in enumerate(long_kws):
        short_kw = (kw[:47] + '...') if len(kw) > 50 else kw
        delta = conf_default[i] - conf_limit[i]
        print(f"{short_kw:<50} | {conf_limit[i]:.4f}     | {conf_default[i]:.4f}     | {delta:.4f}")

    # Summary
    avg_delta = np.mean(np.abs(np.array(conf_default) - np.array(conf_limit)))
    print("-" * 80)
    print(f"Average Absolute Confidence Difference: {avg_delta:.5f}")
    print("="*80)

if __name__ == "__main__":
    compare_models()
