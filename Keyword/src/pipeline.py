import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List
from .engine import UltimateHybridEngine
from .config import Config

class Pipeline:
    def __init__(self, engine: UltimateHybridEngine, config: Config):
        self.engine = engine
        self.config = config

    def run_file(self, input_path: Path, output_path: Path, batch_size: int = 1024):
        print(f">>> Processing: {input_path.name}")
        try:
            # Chunk processing for memory efficiency
            # But the logic requires `process_chunk` which does SBERT.
            # We can read the whole file if it's not huge, or chunk with pandas.
            # Original script read whole DF.
            df = pd.read_csv(input_path, encoding='utf-8-sig')
            
            # Prepare columns
            if 'Keyword' not in df.columns:
                print(f"!!! Error: 'Keyword' column missing in {input_path}")
                return

            df['Keyword_clean'] = df['Keyword'].astype(str).str.lower().str.strip()
            all_keywords = df['Keyword_clean'].tolist()
            all_orig_keywords = df['Keyword'].astype(str).tolist()
            
            final_results = []
            
            # Save
            import time
            start_time = time.time()
            
            # Process in batches
            for start in tqdm(range(0, len(df), batch_size), desc="Pipeline Batch"):
                end = start + batch_size
                batch_kws = all_keywords[start:end]
                batch_orig = all_orig_keywords[start:end]
                
                results = self.engine.process_chunk(
                    batch_kws, 
                    batch_orig, 
                    self.config.ALLOWED_BRANDS, 
                    self.config.SAMSUNG_IDENTIFIERS
                )
                final_results.extend(results)
                
            # Merge results
            res_df = pd.DataFrame(final_results)
            df = df.reset_index(drop=True)
            for col in res_df.columns:
                df[col] = res_df[col]
            
            if 'Keyword_clean' in df.columns:
                df = df.drop(columns=['Keyword_clean'])

            df.to_csv(output_path, index=False, encoding='utf-8-sig', chunksize=1000)
            elapsed = time.time() - start_time
            print(f">>> [SUCCESS] Saved to {output_path} (Took {elapsed:.2f}s)")
            
        except Exception as e:
            print(f"!!! [Error] {input_path} processing failed: {e}")
            import traceback
            traceback.print_exc()

    def run_all(self):
        self.config.ensure_directories()
        input_files = list(self.config.INPUT_DIR.glob("*.csv"))
        
        if not input_files:
            print(">>> No input CSV files found.")
            return

        for f in input_files:
            out_name = self.config.OUTPUT_DIR / f"RESULT_{f.name}"
            self.run_file(f, out_name)
