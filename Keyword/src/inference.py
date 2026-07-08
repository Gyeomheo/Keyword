import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Union
import logging

# Set up logging
logger = logging.getLogger(__name__)

class SBERTVectorizer:
    """
    Handles SBERT model loading and vectorization.
    Used for both training (batch) and inference (single/batch).
    
    [Optimizations]
    - FP16 Half Precision for 2x GPU speed
    - Optimized batch size
    - Embedding normalization for cosine similarity
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = None, use_fp16: bool = True):
        import torch
        
        try:
            logger.info(f"Loading SBERT model: {model_name}")
            self.model = SentenceTransformer(model_name, device=device)
            
            # [Optimization 1] Short sequence length for keywords
            self.model.max_seq_length = 64
            self.model_name = model_name
            
            # [Optimization 2] FP16 Half Precision (2x faster on GPU)
            self.use_fp16 = use_fp16 and torch.cuda.is_available()
            if self.use_fp16:
                logger.info("[TURBO] Enabling FP16 Half Precision for 2x GPU speed")
                self.model.half()  # Convert to FP16
                
            logger.info(f"SBERT loaded on: {self.model.device}, FP16: {self.use_fp16}")
            
        except Exception as e:
            logger.error(f"Failed to load SBERT model: {e}")
            raise

    def encode(self, texts: Union[str, List[str]], batch_size: int = 64, show_progress_bar: bool = False) -> np.ndarray:
        """
        Encode text(s) into vectors with optimized batching.
        Returns numpy array of shape (n_texts, 384)
        
        [Optimization] Default batch_size=64 for GPU efficiency
        """
        if isinstance(texts, str):
            texts = [texts]
            
        try:
            vectors = self.model.encode(
                texts, 
                batch_size=batch_size, 
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            # [Fix] Broad compatibility: cast to float32 (many libs like scipy/sklearn don't like float16)
            return vectors.astype('float32')
        except Exception as e:
            logger.error(f"Vectorization failed: {e}")
            return np.zeros((len(texts), 384)) # Fallback zero vectors


class MLModel:
    """
    Wrapper for Random Forest (or other) Classifier.
    Predicts Division/Product/etc. from HYBRID SBERT + TF-IDF vectors.
    """
    def __init__(self, model_path: Path):
        self.model = None
        self.tfidf = None
        self.model_path = model_path
        self.tfidf_path = model_path.parent / "tfidf_vectorizer.joblib"
        self.load_model()

    def load_model(self):
        if self.model_path.exists():
            try:
                logger.info(f"Loading ML model from {self.model_path}")
                self.model = joblib.load(self.model_path)
            except Exception as e:
                logger.error(f"Failed to load ML model: {e}")
        else:
            logger.warning(f"ML model not found at {self.model_path}. Inference will fail.")
        
        # Load TF-IDF Vectorizer
        if self.tfidf_path.exists():
            try:
                logger.info(f"Loading TF-IDF vectorizer from {self.tfidf_path}")
                self.tfidf = joblib.load(self.tfidf_path)
            except Exception as e:
                logger.error(f"Failed to load TF-IDF vectorizer: {e}")
        else:
            logger.warning(f"TF-IDF vectorizer not found at {self.tfidf_path}. Using SBERT only.")

    def predict(self, vectors: np.ndarray) -> List[Dict[str, str]]:
        """
        Predict categories for given vectors.
        Returns list of dicts with keys: Division, Product, Keyword Type, Brand, Type
        """
        if self.model is None:
            return [{'Division': 'Generic', 'Product': 'Generic'}] * len(vectors)
            
        try:
            # Predict
            # Expecting MultiOutputClassifier or similar capable of predicting all targets
            # prediction labels example: [['MX', 'SMP', 'Samsung', ...], ...]
            predictions = self.model.predict(vectors)
            
            # [Fix] Handle predict_proba for MultiOutputClassifier
            # predict_proba returns a LIST of arrays (one per target), and they have different shapes (different n_classes)
            # We cannot convert this list directly to a numpy array or take np.max without processing first.
            if hasattr(self.model, 'predict_proba'):
                try:
                    all_probs = self.model.predict_proba(vectors)
                    # Use a safe way to handle list of arrays
                    if isinstance(all_probs, list):
                        # For each target's probability array, take the max prob for each sample
                        # Result: list of (N,) arrays
                        max_probs_per_target = [np.max(probs, axis=1) for probs in all_probs]
                        # Average these max probs to get a single confidence score per sample
                        probabilities = np.mean(max_probs_per_target, axis=0) # Shape (N,)
                    else:
                        # Fallback for single output model logic (though we expect multi)
                        probabilities = np.max(all_probs, axis=1)
                except Exception as prob_err:
                    logger.warning(f"Confidence calculation failed: {prob_err}. Defaulting to 1.0")
                    probabilities = np.ones(len(vectors))
            else:
                probabilities = np.ones(len(vectors))
            
            results = []
            # Assuming the model was trained to output [Division, Product, Keyword Type, Brand, Type]
            # We need to ensure the columns align with training. 
            # For simplicity, let's assume predict() returns DataFrame or array in correct order.
            
            columns = ['Division', 'Product', 'Keyword Type', 'Brand', 'Type']
            
            for i, pred in enumerate(predictions):
                # Handle MultiOutput output format (list of arrays) vs Single output (array)
                # If MultiOutputClassifier, predict returns a list of arrays (one per target) NOT an array of lists
                # Wait, MultiOutputClassifier.predict() returns (n_samples, n_outputs) array. Correct.
                
                row_res = {
                    'Division': pred[0],
                    'Product': pred[1],
                    'Keyword Type': pred[2],
                    'Brand': pred[3],
                    'Type': pred[4],
                    'Confidence': float(probabilities[i]) if hasattr(probabilities, '__getitem__') else 0.8
                }
                results.append(row_res)
                
            return results
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return [{'Division': 'Error', 'Product': 'Error'}] * len(vectors)
