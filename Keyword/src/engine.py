import re
import pandas as pd
from typing import Dict, Any, List, Optional
from src.sitemap_mgr import SitemapManager
from src.domain_rules import apply_keyword_rules
from src.enhanced_rules import apply_enhanced_rules
from src.inference import SBERTVectorizer, MLModel

class UltimateHybridEngine:
    def __init__(self, 
                 master_df: pd.DataFrame,
                 sitemap_mgr: SitemapManager,
                 vectorizer: SBERTVectorizer,
                 ml_model: Optional[MLModel] = None):
        
        self.sitemap_mgr = sitemap_mgr
        self.vectorizer = vectorizer
        self.ml_model = ml_model
        
        # [Optimization] Memoization Cache for SBERT/ML results
        self.result_cache = {}
        self.max_cache_size = 50000 # Enough for most sessions
        
        # Exact Match Lookup (Tier 2/3)
        self.categories = ['Division', 'Product', 'Keyword Type', 'Brand', 'Type']
        unique_master = master_df.drop_duplicates(subset=['kw_low'])
        self.exact_lookup = unique_master.set_index('kw_low')[self.categories].to_dict('index')

    def _update_cache(self, kw, result):
        if len(self.result_cache) < self.max_cache_size:
            self.result_cache[kw] = result.copy()

    def process_chunk(self, keywords: List[str], original_keywords: List[str], config_allowed_brands: List[str], config_samsung_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Processes a chunk of keywords through the Tier 0 -> Tier 3 pipeline with mandatory Caching and AI Bypass.
        """
        results = [None] * len(keywords)
        needs_deep_idx = []
        
        # [Tier 0] Cache First
        for i, kw in enumerate(keywords):
            if kw in self.result_cache:
                results[i] = self.result_cache[kw].copy()
                results[i]['Method'] += " (Cached)"
                continue
                
        # [Tier 1-2] Fast Track & AI Bypass
        for i, kw in enumerate(keywords):
            if results[i] is not None: continue # Skip if already in Cache
            
            # Tier 0.5: Sitemap (Model Code) Match -> AI Bypass
            sitemap_result = self.sitemap_mgr.search(kw)
            if sitemap_result:
                kw_lower = kw.lower()
                final_type = 'Model'
                if bool(re.search(r'\bvs\b|\bvs\.\b', kw_lower)): final_type = 'Comparison'
                elif any(x in kw_lower for x in ['review', 'test', 'specs', 'features']): final_type = 'Research'
                elif any(x in kw_lower for x in ['buy', 'price', 'sale', 'cost', 'shop']): final_type = 'Sales'
                
                res = {
                    'Division': sitemap_result['Division'], 'Product': sitemap_result['Product'],
                    'Keyword Type': 'Samsung', 'Brand': 'Samsung', 'Type': final_type,
                    'Method': 'Sitemap Match (Bypass)', 'Confidence': 1.0
                }
                results[i] = res
                self._update_cache(kw, res)
                continue

            # Tier 1: Enhanced Rules -> AI Bypass
            enhanced_result = apply_enhanced_rules(kw)
            if enhanced_result:
                enhanced_result.update({'Confidence': 1.0, 'Method': enhanced_result.get('Method', 'Rule') + ' (Bypass)'})
                results[i] = enhanced_result
                self._update_cache(kw, enhanced_result)
                continue

            # Tier 2: Legacy Regex & Exact Match -> AI Bypass
            hard_res = apply_keyword_rules(kw, config_allowed_brands, config_samsung_ids)
            if hard_res:
                hard_res.update({'Method': 'Regex Rule (Bypass)', 'Confidence': 1.0})
                results[i] = hard_res
                self._update_cache(kw, hard_res)
                continue
                
            if kw in self.exact_lookup:
                res = self.exact_lookup[kw].copy()
                res.update({'Method': "Exact Match (Bypass)", 'Confidence': 1.0})
                results[i] = res
                self._update_cache(kw, res)
                continue
            
            # If still None, mark for deep processing (AI)
            needs_deep_idx.append(i)
            
        # [Tier 3] Fast ML Inference (SBERT + TF-IDF Hybrid -> Random Forest)
        if needs_deep_idx:
            # 1. Vectorize (Batch)
            batch_kws_clean = [keywords[idx] for idx in needs_deep_idx]
            
            # SBERT Vectorizer handles batch encoding efficiently
            sbert_vectors = self.vectorizer.encode(batch_kws_clean, batch_size=len(batch_kws_clean))
            
            # 2. TF-IDF Transformation (if available)
            if self.ml_model and self.ml_model.tfidf is not None:
                from scipy.sparse import hstack, csr_matrix
                # [Fix] scipy.sparse does not support float16. Cast to float32.
                sbert_sparse = csr_matrix(sbert_vectors.astype('float32'))
                tfidf_vectors = self.ml_model.tfidf.transform(batch_kws_clean)
                vectors = hstack([sbert_sparse, tfidf_vectors])
            else:
                vectors = sbert_vectors
            
            # 3. Predict (Batch) using Random Forest
            if self.ml_model:
                predictions = self.ml_model.predict(vectors)
                
                for j, real_idx in enumerate(needs_deep_idx):
                    kw_orig = original_keywords[real_idx]
                    pred = predictions[j]
                    
                    # Construct Result
                    res = pred.copy()
                    res['Method'] = 'ML Inference (SBERT+RF)'
                    # Confidence is already in pred dict
                    
                    # Post-processing for 'vs' (Comparison Type override)
                    if bool(re.search(r'\bvs\b|\bvs\.\b', kw_orig.lower())):
                        res['Type'] = 'Comparison'
                        
                    results[real_idx] = res
                    
                    # [Optimization] Populate Cache
                    self._update_cache(batch_kws_clean[j], res)
            else:
                # Fallback if no ML model loaded
                default_res = {
                    'Division': 'Generic', 'Product': 'Generic', 'Keyword Type': 'Generic', 
                    'Brand': 'Generic', 'Type': 'Category', 'Method': 'Fail (No ML)', 'Confidence': 0.0
                }
                for real_idx in needs_deep_idx:
                    results[real_idx] = default_res
                    # Do not cache fails

        # Fill None results if any (shouldn't happen logic-wise but good for safety)
        default_fail = {'Division': 'Error', 'Product': 'Error', 'Method': 'Fail', 'Confidence': 0.0}
        return [r if r is not None else default_fail for r in results]
