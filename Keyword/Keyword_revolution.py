import pandas as pd
import numpy as np
import os
import joblib
import glob
import torch
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# ==========================================
# [Step 0] 비즈니스 로직 및 제약 조건 설정
# ==========================================

ALLOWED_BRANDS = [
    'Samsung', 'Apple', 'Bissell', 'Bosch', 'Bose', 'Dyson', 'Ecovacs', 'Electrolux', 
    'Eufy', 'Fitbit', 'Generic', 'iHome', 'iRobot Roomba', 'LG', 'Miele', 'Oppo', 
    'Others', 'Roborock', 'Shark', 'Vivo', 'Whirlpool', 'Xiaomi'
]

SAMSUNG_IDENTIFIERS = [
    'samsung', 'galaxy', 'bespoke', 'jet bot', 'odyssey', 'neo qled', 'qled', 'uhd', 'frame tv', 
    's95', 's90', 's85', 'the frame', 'the serif', 'smart monitor', 'galaxy book'
]

def apply_keyword_rules(keyword_text):
    kw = keyword_text.lower().strip()
    is_comparison = bool(re.search(r'\bvs\b|\bvs\.\b', kw))
    final_brand, final_kw_type = 'Generic', 'Generic'
    
    if any(x in kw for x in SAMSUNG_IDENTIFIERS) or re.search(r'sm-[a-z]\d{3}', kw):
        final_brand, final_kw_type = 'Samsung', 'Samsung'
    else:
        for b in ALLOWED_BRANDS:
            if b == 'Samsung': continue
            if b.lower() in kw:
                final_brand, final_kw_type = b, 'Other Brands'
                break

    if re.search(r'sm-[a-z]\d{3,4}', kw) or re.search(r'\bnp\d{3}[a-z]', kw) or 'galaxy book' in kw:
        div, prod = 'MX', 'SMP'
        if any(x in kw for x in ['sm-x', 'sm-p', 'tab']): prod = 'Tab'
        elif any(x in kw for x in ['sm-r', 'watch']): prod = 'Watch'
        elif 'buds' in kw: prod = 'TWS'
        elif 'book' in kw or 'np' in kw: prod = 'Laptop'
        return create_res(div, prod, final_kw_type, final_brand, kw, is_comparison)

    vd_pattern = re.search(r'(qn|qa|ua|kq|un|sc|ls|s|lu|lc)\d{2}[a-z]{1,2}', kw)
    if vd_pattern:
        prefix = vd_pattern.group(1)
        div = 'VD'
        if prefix in ['qn', 'kq']: prod = 'QLED TV'
        elif prefix in ['sc', 's'] and re.search(r's[8-9]\d[a-z]', kw): prod = 'OLED TV'
        elif prefix in ['lu', 'lc', 'ls'] and 'monitor' in kw: prod = 'Monitor'
        else: prod = 'TV'
        return create_res(div, prod, final_kw_type, final_brand, kw, is_comparison)

    if re.search(r'\brf\d{2}|\brs\d{2}', kw) or 'refrigerator' in kw:
        return create_res('DA', 'REF', final_kw_type, final_brand, kw, is_comparison)
    if re.search(r'\bwf\d{2}|\bdv\d{2}|\bwa\d{2}', kw) or 'washer' in kw or 'dryer' in kw:
        return create_res('DA', 'Laundry', final_kw_type, final_brand, kw, is_comparison)
    return None

def create_res(div, prod, kw_type, brand, kw, is_comparison):
    if is_comparison: final_type = 'Comparison'
    else:
        if any(x in kw for x in ['review', 'test', 'specs']): final_type = 'Research'
        elif any(x in kw for x in ['buy', 'price', 'sale']): final_type = 'Sales'
        elif any(x in kw for x in ['s24', 's23', 'iphone', 'sm-', 'qn', 'rf']): final_type = 'Model'
        else: final_type = 'Category'
    return {'Division': div, 'Product': prod, 'Keyword Type': kw_type, 'Brand': brand, 'Type': final_type, 'Method': 'Hard Rule (Regex)'}

class UltimateHybridEngine:
    def __init__(self, master_path, model_path, vector_cache_path, sitemap_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f">>> [System] Device: {self.device.upper()}")
        
        # 1. Sitemap KB 로드 및 부분 일치 엔진 빌드
        self.sitemap_kb = self.load_sitemap_kb(sitemap_path)
        if self.sitemap_kb:
            # 긴 모델명부터 매칭되도록 정렬 (예: 'S24 Ultra'가 'S24'보다 먼저 잡히게)
            sorted_models = sorted(self.sitemap_kb.keys(), key=len, reverse=True)
            pattern_str = '|'.join(map(re.escape, sorted_models))
            self.sitemap_pattern = re.compile(f"({pattern_str})", re.IGNORECASE)
            print(f">>> [System] Partial Match Engine Armed: {len(sorted_models)} models.")
        else:
            self.sitemap_pattern = None

        # 2. Master DB 및 벡터 로드
        self.master_df = pd.read_csv(master_path)
        self.categories = ['Division', 'Product', 'Keyword Type', 'Brand', 'Type']
        self.master_df['kw_low'] = self.master_df['Keyword'].astype(str).str.lower().str.strip()
        self.master_df = self.master_df.drop_duplicates(subset=['kw_low'])
        self.exact_lookup = self.master_df.set_index('kw_low')[self.categories].to_dict('index')
        self.master_keywords = self.master_df['kw_low'].values
        self.master_records = self.master_df[self.categories].values

        # 3. SBERT 모델 로드
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        if os.path.exists(vector_cache_path):
            vecs = joblib.load(vector_cache_path)
        else:
            print(">>> Encoding Master Vectors...")
            vecs = self.sbert_model.encode(self.master_keywords.tolist(), show_progress_bar=True)
            joblib.dump(vecs, vector_cache_path)
        
        self.master_vectors = torch.tensor(vecs).to(self.device).to(torch.float32)
        self.rf_model = joblib.load(model_path) if os.path.exists(model_path) else None

    def load_sitemap_kb(self, path):
        kb = {}
        if not os.path.exists(path): return kb
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    model, div = parts[0].lower(), parts[1]
                    if div != "Unknown" and not model.isdigit(): kb[model] = div
        print(f">>> [System] Sitemap KB Loaded: {len(kb)} entries.")
        return kb

    def batch_process(self, input_df, semantic_threshold=0.85, batch_size=128):
        df = input_df.copy()
        df['Keyword_clean'] = df['Keyword'].astype(str).str.lower().str.strip()
        results = [None] * len(df)
        needs_deep_idx = []

        # [Tier 0-2] Fast Track Screening
        print(">>> Screening keywords with Fast Track (Sitemap/Regex/Exact)...")
        for i, kw in enumerate(df['Keyword_clean']):
            # [Tier 0] 부분 일치 사이트맵 매칭 (개선된 핵심 로직!)
            sitemap_match = None
            if self.sitemap_pattern:
                match = self.sitemap_pattern.search(kw)
                if match:
                    found_model = match.group().lower()
                    sitemap_match = self.sitemap_kb.get(found_model)

            if sitemap_match:
                results[i] = {
                    'Division': sitemap_match, 'Product': 'Model', 'Keyword Type': 'Samsung', 
                    'Brand': 'Samsung', 'Type': 'Model', 'Method': f'Sitemap Partial ({found_model})', 'Confidence': 1.0
                }
            else:
                # [Tier 1] Regex Rule
                hard_res = apply_keyword_rules(kw)
                if hard_res:
                    hard_res.update({'Confidence': 1.0}); results[i] = hard_res
                # [Tier 2] Exact Master DB Match
                elif kw in self.exact_lookup:
                    res = self.exact_lookup[kw].copy(); res.update({'Method': "Exact Match", 'Confidence': 1.0}); results[i] = res
                else:
                    # 모든 그물을 빠져나간 것만 BERT로
                    needs_deep_idx.append(i)

        # [Tier 3] Deep Processing (BERT)
        if needs_deep_idx:
            print(f">>> Deep processing {len(needs_deep_idx)} keywords (Reduced Workload)")
            for start in tqdm(range(0, len(needs_deep_idx), batch_size), desc="Deep Batches"):
                end = min(start + batch_size, len(needs_deep_idx))
                batch_indices = needs_deep_idx[start:end]
                batch_kws = [df['Keyword_clean'].iloc[idx] for idx in batch_indices]
                
                with torch.no_grad():
                    # 모델 내부 배치 사이즈도 지정하여 효율 극대화
                    batch_embeddings = self.sbert_model.encode(batch_kws, convert_to_tensor=True, batch_size=batch_size).to(torch.float32)
                    cos_sim = util.cos_sim(batch_embeddings, self.master_vectors)
                    max_sims, best_indices = torch.max(cos_sim, dim=1)
                    
                    max_sims = max_sims.cpu().numpy()
                    best_indices = best_indices.cpu().numpy()

                for j, real_idx in enumerate(batch_indices):
                    kw_orig = df['Keyword_clean'].iloc[real_idx]
                    if max_sims[j] >= semantic_threshold:
                        best_match_idx = best_indices[j]
                        res = dict(zip(self.categories, self.master_records[best_match_idx]))
                        res.update({'Method': f"Semantic ({self.master_keywords[best_match_idx]})", 'Confidence': float(max_sims[j])})
                    else:
                        if self.rf_model:
                            pred = self.rf_model.predict([kw_orig])[0]
                            res = dict(zip(self.categories, pred))
                            res.update({'Method': "ML Inference (RF)", 'Confidence': 0.0})
                        else:
                            res = {c: "N/A" for c in self.categories}; res.update({'Method': "Fail", 'Confidence': 0.0})
                    
                    if bool(re.search(r'\bvs\b|\bvs\.\b', kw_orig)): res['Type'] = 'Comparison'
                    results[real_idx] = res

                del batch_embeddings; del cos_sim
                torch.cuda.empty_cache()

        # [Final Step] 결과 정규화 및 저장
        print(">>> [System] Reformatting results for export...")
        default_fail = {c: "N/A" for c in self.categories}
        default_fail.update({'Method': 'Fail', 'Confidence': 0.0})
        
        final_res_list = [r if r is not None else default_fail for r in results]
        res_df = pd.DataFrame(final_res_list)
        
        df = df.reset_index(drop=True)
        for col in res_df.columns:
            df[col] = res_df[col]

        if 'Keyword_clean' in df.columns:
            df = df.drop(columns=['Keyword_clean'])
            
        return df

if __name__ == "__main__":
    BASE_DIR = r"C:\Keyword"
    paths = {
        'master': os.path.join(BASE_DIR, "master", "검색량.csv"),
        'model': os.path.join(BASE_DIR, "models", "ultimate_engine.joblib"),
        'cache': os.path.join(BASE_DIR, "master_vectors_v1.pkl"),
        'sitemap': os.path.join(BASE_DIR, "samsung_model_master.txt"),
        'input': os.path.join(BASE_DIR, "input"),
        'output': os.path.join(BASE_DIR, "output")
    }

    print(">>> [Check]")
    engine = UltimateHybridEngine(paths['master'], paths['model'], paths['cache'], paths['sitemap'])
    input_files = glob.glob(os.path.join(paths['input'], "*.csv"))
    
    for file_path in input_files:
        print(f"\n>>> Processing: {os.path.basename(file_path)}")
        try:
            input_df = pd.read_csv(file_path, encoding='utf-8-sig')
            final_result = engine.batch_process(input_df)
            
            if not os.path.exists(paths['output']): os.makedirs(paths['output'])
            out_name = os.path.join(paths['output'], f"RESULT_{os.path.basename(file_path)}")
            
            final_result.to_csv(out_name, index=False, encoding='utf-8-sig', chunksize=1000)
            print(f">>> [SUCCESS] Saved to {out_name}")
        except Exception as e:
            print(f"!!! [Error] {file_path} 처리 중 에러 발생: {e}")
